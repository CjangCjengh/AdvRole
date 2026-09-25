# AdvRole

[![Paper](https://img.shields.io/badge/Paper-arXiv%202609.28609-b31b1f)](https://arxiv.org/abs/2609.28609)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-2dd4a7)](https://cjangcjengh.github.io/AdvRole/)

This is the official implementation of the paper *Adversarial Closed-Loop Curriculum for Evolving Role-Playing Agents*.

AdvRole turns role-playing RL into a closed-loop curriculum: an **Actor** learns to role-play while a **Rewriter** is trained with a performance-gap reward to edit character profiles and dialogue contexts into actor-specific hard scenarios, so the training pool keeps evolving with the Actor. It is evaluated on CharacterEval, CoSER, RAIDEN (zero-shot), and **Lanobe**, a new multilingual role-playing benchmark covering Simplified/Traditional Chinese, Japanese, Korean, and Thai.

## Setup

```bash
conda create --prefix /path/to/advrole_env python=3.10 -y && conda activate $_
pip install vllm==0.12.0 verl==0.8.0 \
    openai pandas pyarrow datasets tqdm pyyaml scikit-learn nltk rouge-score jieba rich

# on machines where verl hits "CUDA error: invalid argument" during FSDP offload or "No module named 'flash_attn'":
bash scripts/patch_verl_offload.sh

# benchmarks (clone into data/raw/), then normalise to the unified schema
mkdir -p data/raw && cd data/raw
git clone --depth 1 https://github.com/morecry/CharacterEval.git
git clone --depth 1 https://github.com/FrontierLabs/RAIDEN.git
git clone --depth 1 https://github.com/Neph0s/CoSER.git
cd ../..
python scripts/prepare_charactereval.py   # 2000 train / 2564 test (paper split)
python scripts/prepare_coser.py           # 17,762 train (paper) / 200 test
python scripts/prepare_raiden.py          # 4470 test instances (zero-shot)
python scripts/prepare_lanobe.py --demo-dir <dir>   # demos from the paper supplement

cp configs/local.yaml.example configs/local.yaml   # machine-specific (gitignored)
```

## Training

```bash
python train.py --dataset charactereval          # also: coser | lanobe | raiden
python train.py --dataset lanobe --lang ja
python train.py --dataset coser --dry-run        # print verl commands only
python train.py --set training.num_epochs=1 ...  # dotted overrides
```

## Evaluation

```bash
bash scripts/serve_model.sh <run>/ckpts/<exp>/global_step_N/actor/huggingface actor 8001

python -m eval.eval_charactereval --model-endpoint http://localhost:8001/v1 \
    --model-name actor --rm-path /path/to/BaichuanCharRM --output results/ce.json
python -m eval.eval_coser    --model-endpoint ... --model-name actor \
    --judge-endpoint http://localhost:8003/v1 --judge-name judge --output results/coser.json
python -m eval.eval_raiden   --model-endpoint ... --model-name actor \
    --judge-endpoint ... --judge-name judge --output results/raiden.json
python -m eval.eval_lanobe generate|judge|summarize --model-name actor ...
python -m eval.diversity --original <S0.jsonl> --rewritten <rw.jsonl> \
    --embed-model <path-or-id> --output results/diversity.json
```

## Smoke test

```bash
bash scripts/run_smoke.sh     # 2 actor steps + 1 rewriter step + 16 rewrites
```

## Citation

```bibtex
@misc{zhang2026adversarialclosedloopcurriculumevolving,
      title={Adversarial Closed-Loop Curriculum for Evolving Role-Playing Agents},
      author={Zheng Zhang and Liu Liu and Qi Chai and Deheng Ye and Peilin Zhao and Mao Zheng and Hao Wang},
      year={2026},
      eprint={2609.28609},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.28609},
}
```
