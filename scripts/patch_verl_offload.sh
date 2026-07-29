#!/bin/bash
# Patches the installed verl package for machines where it fails:
# 1) FSDP param/optimizer offload uses non_blocking GPU->CPU copies that can
#    fail with "CUDA error: invalid argument" -> make them blocking;
# 2) the no-padding log-prob path imports flash_attn unconditionally ->
#    fall back to verl's own pure-torch npu_flash_attn_utils.
# Idempotent. Run inside the advrole env.
set -euo pipefail

PY=${PYTHON:-python}
F=$($PY -c "import verl, os; print(os.path.join(os.path.dirname(verl.__file__), 'utils', 'fsdp_utils.py'))")
echo "Patching: $F"

$PY - "$F" <<'EOF'
import sys
f = sys.argv[1]
src = open(f).read()
if 'AdvRole patch' in src:
    print("already patched, nothing to do")
    sys.exit(0)
repls = [
    ('        handle.flat_param_to(torch.device("cpu"), non_blocking=True)',
     '        torch.cuda.synchronize()  # AdvRole patch: flush pending async ops\n'
     '        handle.flat_param_to(torch.device("cpu"), non_blocking=False)  # AdvRole patch: blocking D2H'),
    ('                    state[key] = value.to("cpu", non_blocking=True)',
     '                    state[key] = value.to("cpu")  # AdvRole patch: blocking D2H'),
    ('                    state[key] = value.to(device_id, non_blocking=True)',
     '                    state[key] = value.to(device_id)  # AdvRole patch: blocking'),
    ('        handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"), non_blocking=True)',
     '        handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"))  # AdvRole patch: blocking'),
]
for old, new in repls:
    if old in src:
        src = src.replace(old, new)
    else:
        print(f"warning: pattern not found (verl layout changed?): {old[:50]}...", file=sys.stderr)
open(f, "w").write(src)
print("patched")
EOF

# --- patch 2: pure-torch fallback for flash_attn.bert_padding ---------------
# verl's no-padding log-prob path imports flash_attn unconditionally; the
# NPU utils module ships an identical pure-torch implementation.
F2=$($PY -c "import verl, os; print(os.path.join(os.path.dirname(verl.__file__), 'utils', 'attention_utils.py'))")
echo "Patching: $F2"
$PY - "$F2" <<'EOF'
import sys
f = sys.argv[1]
src = open(f).read()
old = """    else:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
"""
new = """    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:  # AdvRole patch: pure-torch fallback when flash_attn is absent
            from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input
"""
if 'AdvRole patch' in src:
    print("already patched, nothing to do")
elif old in src:
    open(f, "w").write(src.replace(old, new))
    print("patched")
else:
    print("target line not found; verl layout may have changed", file=sys.stderr)
    sys.exit(1)
EOF
