from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

from advrole.config import load_config, resolve_path, REPO_ROOT
from advrole.data import (load_pool, scenarios_to_rewriter_verl_parquet,
                          scenarios_to_verl_parquet)
from advrole.gpu import parse_gpu_ids, resolve_gpu_ids
from advrole.server import VLLMServer, find_free_port, server_from_config, wait_for_endpoint
from advrole.utils import (find_hf_checkpoint, merge_verl_fsdp_checkpoint,
                           run_command, set_seed, setup_logging)

logger = logging.getLogger("advrole.train")

def sized_pool(scenarios: List[Dict], n_rows: int) -> List[Dict]:
    # verl 0.8 has no trainer.total_training_steps: one pass over
    # num_steps*batch rows == num_steps GRPO steps

    if not scenarios:
        raise ValueError("empty scenario pool")
    if len(scenarios) >= n_rows:
        return scenarios[:n_rows]
    reps = (n_rows + len(scenarios) - 1) // len(scenarios)
    return (scenarios * reps)[:n_rows]

def build_grpo_cmd(cfg: Dict, role: str, model_path: str, train_parquet: str,
                   val_parquet: str, num_steps: int, run_dir: str,
                   experiment: str, n_gpus: int, ckpt_dir: str) -> List[str]:
    g = cfg["grpo"]
    role_cfg = cfg[role]
    hw = cfg["hardware"]
    offload = bool(hw.get("fsdp_offload", False))
    attn = hw.get("attn_implementation", "flash_attention_2")
    unpad = bool(hw.get("use_remove_padding", True))
    fsdp_strategy = str(hw.get("fsdp_strategy", "fsdp"))
    compile_flags = bool(hw.get("vllm_torch_compile", True))

    cmd = [sys.executable, "-m", "verl.trainer.main_ppo",
           "algorithm.adv_estimator=grpo",
           f"data.train_files={train_parquet}",
           f"data.val_files={val_parquet}",
           f"data.train_batch_size={g['batch_size']}",
           f"data.max_prompt_length={g['max_prompt_length']}",
           f"data.max_response_length={role_cfg['max_new_tokens']}",
           "data.filter_overlong_prompts=True",
           f"actor_rollout_ref.model.path={model_path}",
           "actor_rollout_ref.model.enable_gradient_checkpointing=True",
           f"+actor_rollout_ref.model.override_config.attn_implementation={attn}",
           f"actor_rollout_ref.model.use_remove_padding={unpad}",
           f"actor_rollout_ref.actor.optim.lr={g['learning_rate']}",
           f"actor_rollout_ref.actor.ppo_mini_batch_size={g['ppo_mini_batch_size']}",
           f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={g['ppo_micro_batch_size_per_gpu']}",
           f"actor_rollout_ref.actor.clip_ratio={g['clip_epsilon']}",
           f"actor_rollout_ref.actor.grad_clip={g['max_grad_norm']}",
           "actor_rollout_ref.actor.use_kl_loss=True",
           f"actor_rollout_ref.actor.kl_loss_coef={g['kl_coeff']}",
           "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
           f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={g['ppo_micro_batch_size_per_gpu']}",
           f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={g['ppo_micro_batch_size_per_gpu']}",
           f"actor_rollout_ref.actor.fsdp_config.strategy={fsdp_strategy}",
           f"actor_rollout_ref.ref.fsdp_config.strategy={fsdp_strategy}",
           f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
           f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
           f"actor_rollout_ref.ref.fsdp_config.param_offload={offload}",
           f"actor_rollout_ref.actor.fsdp_config.use_torch_compile={compile_flags}",
           f"actor_rollout_ref.rollout.n={g['group_size']}",
           f"actor_rollout_ref.rollout.temperature={role_cfg['temperature']}",
           "actor_rollout_ref.rollout.top_p=0.95",
           f"actor_rollout_ref.rollout.tensor_model_parallel_size={hw.get('rollout_tp', 1)}",
           f"actor_rollout_ref.rollout.gpu_memory_utilization={hw.get('rollout_gpu_memory_utilization', 0.5)}",
           f"actor_rollout_ref.rollout.enforce_eager={bool(hw.get('rollout_enforce_eager', False))}",
           f"actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes={hw.get('update_weights_bucket_mb', 256)}",
           "actor_rollout_ref.rollout.name=vllm",
           "algorithm.kl_ctrl.kl_coef=0.0",
           "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
           f"custom_reward_function.path={os.path.join(REPO_ROOT, 'reward_functions', role + '_reward.py')}",
           "custom_reward_function.name=compute_score",
           f"trainer.n_gpus_per_node={n_gpus}",
           "trainer.nnodes=1",

           "trainer.total_epochs=1",
           f"trainer.save_freq={max(1, num_steps)}",
           "trainer.val_before_train=False",
           "trainer.test_freq=-1",
           "trainer.project_name=advrole",
           f"trainer.experiment_name={experiment}",

           # one dir per experiment: a shared dir + resume_mode=auto makes
           # the second run resume from the first one's checkpoint and exit
           f"trainer.default_local_dir={ckpt_dir}",
           "trainer.resume_mode=disable",
           "trainer.logger=[console]",
           ]
    if fsdp_strategy == "fsdp2" and offload:
        cmd.append("actor_rollout_ref.actor.fsdp_config.offload_policy=True")
    return cmd

class Endpoint:

    def __init__(self, name: str, cfg_block: Dict, model_fallback: str,
                 default_port: int, log_dir: str):
        self.name = name
        self.manage = bool(cfg_block.get("manage", False))
        self.api_base = cfg_block.get("api_base", f"http://localhost:{default_port}/v1")
        self.api_key = cfg_block.get("api_key", "EMPTY")
        self.model = cfg_block.get("served_name") or cfg_block.get("model") or name
        self._server: Optional[VLLMServer] = None
        self._cfg_block = cfg_block
        self._model_fallback = model_fallback
        self._log_dir = log_dir

    def start(self, model_path: Optional[str] = None,
              exclude_gpus: "set[int] | None" = None) -> None:
        if not self.manage:
            if not wait_for_endpoint(self.api_base, timeout=60, api_key=self.api_key):
                raise RuntimeError(
                    f"External endpoint '{self.name}' not reachable at {self.api_base}. "
                    f"Start it yourself or set servers.{self.name}.manage: true")
            return
        block = dict(self._cfg_block)
        if model_path:
            block["model"] = model_path
        else:
            block.setdefault("model", self._model_fallback)
        block["port"] = find_free_port()
        block["gpu_ids"] = resolve_gpu_ids(
            str(block.get("gpu_ids", "auto:1")),
            int(block.get("tensor_parallel_size", 1)),
            int(block.get("min_free_mb", 40000)),
            exclude=exclude_gpus)
        self._server = server_from_config(self.name, block, log_dir=self._log_dir)
        self._server.start()
        self.api_base = self._server.api_base
        self.model = self.name

    @property
    def gpu_ids_resolved(self) -> "set[int]":
        if self._server is not None:
            return parse_gpu_ids(self._server.gpu_ids)
        return set()

    def update_model(self, model_path: str,
                     exclude_gpus: "set[int] | None" = None) -> None:
        if self._server is not None:
            self._server.stop()
        self.start(model_path, exclude_gpus=exclude_gpus)

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None

def run(args) -> None:
    cfg = load_config(args.config, overrides=args.set)
    dataset = args.dataset
    lang = args.lang

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset}" + (f"_{lang}" if lang else "") + f"_{stamp}"
    run_dir = resolve_path(os.path.join(cfg["output"]["output_dir"], run_name))
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(os.path.join(run_dir, "train.log"))
    set_seed(args.seed)

    with open(os.path.join(run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg, f, allow_unicode=True)
    logger.info(f"Run dir: {run_dir}")

    data_cfg = cfg["data"][dataset]
    if dataset == "lanobe":
        lang = lang or "ja"
        lanobe_dir = resolve_path(data_cfg["data_dir"])
        train_path = os.path.join(lanobe_dir, f"{lang}_train.jsonl")
        val_path = os.path.join(lanobe_dir, f"{lang}_test.jsonl")
    else:
        train_path = resolve_path(data_cfg["train_path"])
        val_path = resolve_path(data_cfg.get("val_path") or data_cfg["train_path"])
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Prepared training pool not found: {train_path}\n"
            f"Run: python scripts/prepare_{dataset}.py"
            + (f" --lang {lang}" if lang else ""))
    original_pool = load_pool(train_path)
    current_pool = list(original_pool)
    logger.info(f"Original pool S0: {len(original_pool)} scenarios")

    proc_dir = os.path.join(run_dir, "parquet")
    os.makedirs(proc_dir, exist_ok=True)

    log_dir = os.path.join(run_dir, "logs")
    rm_ep = Endpoint("rm", cfg["servers"]["rm"],
                     cfg["model"]["reward_model"]["model"], 8000, log_dir)
    actor_ep = Endpoint("actor", cfg["servers"]["actor"],
                        cfg["model"]["actor"]["base_model"], 8001, log_dir)
    rewriter_ep = Endpoint("rewriter", cfg["servers"]["rewriter"],
                           cfg["model"]["rewriter"]["base_model"], 8002, log_dir)

    rm_env_base = {
        "RM_API_BASE": cfg["model"]["reward_model"].get("api_base", rm_ep.api_base),
        "RM_API_KEY": cfg["model"]["reward_model"].get("api_key", "EMPTY"),
        "RM_MODEL": cfg["model"]["reward_model"]["model"],
        "RM_DATASET": cfg["model"]["reward_model"].get("prompt_style") or dataset,
    }

    hw = cfg["hardware"]
    reserved: set = set()

    if args.dry_run:
        logger.info("[dry-run] skipping RM server startup")
    else:
        rm_ep.start()
        rm_env_base["RM_API_BASE"] = rm_ep.api_base
        rm_env_base["RM_MODEL"] = rm_ep.model
        reserved = rm_ep.gpu_ids_resolved

    actor_gpu_ids = resolve_gpu_ids(str(hw.get("actor_gpus", "auto:4")),
                                    int(hw.get("actor_gpus_n", 4)),
                                    int(hw.get("min_free_mb", 40000)),
                                    exclude=reserved)
    reserved |= parse_gpu_ids(actor_gpu_ids)
    n_gpus = max(1, len(actor_gpu_ids.split(","))) if actor_gpu_ids else int(os.environ.get("GPU_COUNT", "1"))
    logger.info(f"verl training will use GPUs: {actor_gpu_ids or '(all visible)'} x{n_gpus}")

    actor_model = resolve_path(cfg["model"]["actor"]["base_model"])
    rewriter_model = resolve_path(cfg["model"]["rewriter"]["base_model"])
    num_epochs = int(cfg["training"]["num_epochs"])

    try:
        for epoch in range(1, num_epochs + 1):
            logger.info("=" * 70)
            logger.info(f"EPOCH {epoch}/{num_epochs}  (pool size: {len(current_pool)})")
            logger.info("=" * 70)

            actor_steps = int(cfg["training"]["actor_steps_per_epoch"])
            if actor_steps > 0:
                pool_path = os.path.join(proc_dir, f"actor_pool_epoch{epoch}.jsonl")
                from advrole.data import save_jsonl
                save_jsonl(current_pool, pool_path)
                train_pq = os.path.join(proc_dir, f"actor_epoch{epoch}.parquet")
                val_pq = os.path.join(proc_dir, f"actor_val.parquet")
                n_rows = actor_steps * int(cfg["grpo"]["batch_size"])
                scenarios_to_verl_parquet(sized_pool(current_pool, n_rows),
                                          train_pq,
                                          data_source=f"advrole_{dataset}")
                val_pool = load_pool(val_path)[:max(4, cfg["grpo"]["batch_size"])]
                scenarios_to_verl_parquet(val_pool, val_pq,
                                          data_source=f"advrole_{dataset}")

                actor_ckpt_dir = os.path.join(run_dir, "ckpts", f"{dataset}_actor_epoch{epoch}")
                cmd = build_grpo_cmd(cfg, "actor", actor_model, train_pq, val_pq,
                                     actor_steps, run_dir,
                                     f"{dataset}_actor_epoch{epoch}", n_gpus,
                                     actor_ckpt_dir)
                env = dict(os.environ)
                env.update(rm_env_base)
                if actor_gpu_ids:
                    env["CUDA_VISIBLE_DEVICES"] = actor_gpu_ids
                env["TOKENIZERS_PARALLELISM"] = "false"
                env.update({str(k): str(v) for k, v in hw.get("extra_env", {}).items()})
                if not bool(hw.get("vllm_torch_compile", True)):
                    env["VLLM_TORCH_COMPILE_LEVEL"] = "0"

                if args.dry_run:
                    logger.info("[dry-run] " + " ".join(cmd))
                else:
                    rc = run_command(cmd, env=env,
                                     log_file=os.path.join(log_dir, f"verl_actor_epoch{epoch}.log"))
                    if rc != 0:
                        raise RuntimeError(f"Actor GRPO failed (exit {rc})")
                    ckpt = _locate_actor_ckpt(run_dir, f"{dataset}_actor_epoch{epoch}")
                    if ckpt:
                        actor_model = ckpt
                        logger.info(f"Actor checkpoint: {actor_model}")

            rewriter_steps = int(cfg["training"]["rewriter_steps_per_epoch"])
            if rewriter_steps > 0:
                if not args.dry_run:
                    actor_ep.update_model(actor_model, exclude_gpus=reserved)
                rw_train_pq = os.path.join(proc_dir, f"rewriter_epoch{epoch}.parquet")
                rw_val_pq = os.path.join(proc_dir, f"rewriter_val.parquet")
                n_rows = rewriter_steps * int(cfg["grpo"]["batch_size"])
                scenarios_to_rewriter_verl_parquet(
                    sized_pool(original_pool, n_rows), rw_train_pq)
                scenarios_to_rewriter_verl_parquet(
                    load_pool(val_path)[:max(4, cfg["grpo"]["batch_size"])], rw_val_pq)

                rw_ckpt_dir = os.path.join(run_dir, "ckpts", f"{dataset}_rewriter_epoch{epoch}")
                cmd = build_grpo_cmd(cfg, "rewriter", rewriter_model, rw_train_pq,
                                     rw_val_pq, rewriter_steps, run_dir,
                                     f"{dataset}_rewriter_epoch{epoch}", n_gpus,
                                     rw_ckpt_dir)
                env = dict(os.environ)
                env.update(rm_env_base)
                env.update({
                    "ACTOR_API_BASE": actor_ep.api_base,
                    "ACTOR_API_KEY": actor_ep.api_key,
                    "ACTOR_MODEL": actor_ep.model,
                    "NUM_ACTOR_SAMPLES": str(cfg["rewriter"]["num_actor_samples"]),
                    "ACTOR_TEMPERATURE": str(cfg["actor"]["temperature"]),
                    "ACTOR_MAX_TOKENS": str(cfg["actor"]["max_new_tokens"]),
                })
                if actor_gpu_ids:
                    env["CUDA_VISIBLE_DEVICES"] = actor_gpu_ids
                env["TOKENIZERS_PARALLELISM"] = "false"
                env.update({str(k): str(v) for k, v in hw.get("extra_env", {}).items()})
                if not bool(hw.get("vllm_torch_compile", True)):
                    env["VLLM_TORCH_COMPILE_LEVEL"] = "0"

                if args.dry_run:
                    logger.info("[dry-run] " + " ".join(cmd))
                else:
                    rc = run_command(cmd, env=env,
                                     log_file=os.path.join(log_dir, f"verl_rewriter_epoch{epoch}.log"))
                    if rc != 0:
                        raise RuntimeError(f"Rewriter GRPO failed (exit {rc})")
                    ckpt = _locate_actor_ckpt(run_dir, f"{dataset}_rewriter_epoch{epoch}")
                    if ckpt:
                        rewriter_model = ckpt
                        logger.info(f"Rewriter checkpoint: {rewriter_model}")
                actor_ep.stop()

            if args.dry_run:
                logger.info("[dry-run] skipping augmentation")
                continue

            rewriter_ep.update_model(rewriter_model, exclude_gpus=reserved)
            next_pool_path = os.path.join(proc_dir, f"actor_pool_epoch{epoch + 1}.jsonl")
            aug_cmd = [
                sys.executable, os.path.join(REPO_ROOT, "augment_pool.py"),
                "--rewriter-api-base", rewriter_ep.api_base,
                "--rewriter-api-key", rewriter_ep.api_key,
                "--rewriter-model", rewriter_ep.model,
                "--original-pool", train_path,
                "--current-pool", os.path.join(proc_dir, f"actor_pool_epoch{epoch}.jsonl"),
                "--output", next_pool_path,
                "--temperature", str(cfg["rewriter"]["temperature"]),
                "--max-tokens", str(cfg["rewriter"]["max_new_tokens"]),
                "--concurrency", str(cfg["rewriter"].get("augment_concurrency", 16)),
            ]
            if int(cfg["rewriter"].get("augment_limit", 0)) > 0:
                aug_cmd += ["--limit", str(cfg["rewriter"]["augment_limit"])]
            rc = run_command(aug_cmd, log_file=os.path.join(log_dir, f"augment_epoch{epoch}.log"))
            rewriter_ep.stop()
            if rc != 0:
                raise RuntimeError(f"Augmentation failed (exit {rc})")
            current_pool = load_pool(next_pool_path)
            logger.info(f"Pool after epoch {epoch}: {len(current_pool)} scenarios")

        logger.info("Training finished. Final actor model: %s", actor_model)
        with open(os.path.join(run_dir, "final_models.json"), "w", encoding="utf-8") as f:
            json.dump({"actor": actor_model, "rewriter": rewriter_model}, f, indent=2)
    finally:
        rm_ep.stop()
        actor_ep.stop()
        rewriter_ep.stop()

def _locate_actor_ckpt(run_dir: str, experiment: str) -> Optional[str]:

    exp_root = os.path.join(run_dir, "ckpts", experiment)
    hf = find_hf_checkpoint(exp_root)
    if hf:
        return hf
    import glob
    steps = sorted(glob.glob(os.path.join(exp_root, "global_step_*")),
                   key=lambda p: int(p.rsplit("_", 1)[-1]) if p.rsplit("_", 1)[-1].isdigit() else 0)
    for step_dir in reversed(steps):
        actor_dir = os.path.join(step_dir, "actor")
        if os.path.isdir(actor_dir):
            try:
                return merge_verl_fsdp_checkpoint(step_dir)
            except Exception as e:
                logger.warning(f"Checkpoint merge failed for {step_dir}: {e}")
    logger.warning(f"No checkpoint found under {exp_root}")
    return None

def parse_args():
    p = argparse.ArgumentParser(description="AdvRole training loop")
    p.add_argument("--dataset", required=True,
                   choices=["charactereval", "coser", "lanobe", "raiden"])
    p.add_argument("--lang", default=None, help="language for lanobe (e.g. ja)")
    p.add_argument("--config", default=None, help="extra config yaml")
    p.add_argument("--set", nargs="*", default=[],
                   help="dotted overrides, e.g. --set training.num_epochs=1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="build everything and print verl commands without running")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
