#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="$1"
shift
export PYTHONPATH="$ROOT/TWP:$ROOT/VLN-CE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export NCCL_P2P_DISABLE=1
if [[ "$ENTRY" == "VLN-CE/run.py" && "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
    exec python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-29500}" "$ROOT/$ENTRY" "$@"
fi
exec python "$ROOT/$ENTRY" "$@"
