#!/usr/bin/env bash
# Run serve.py exactly like the judge: offline, 4 CPUs, 6 GB RAM, read-only repo, writable /tmp only.
# Then run the kit's schema validator against it from a sibling container that shares its (empty) network.
# Usage: KIT=/path/to/yug-monsoon-kit bash scripts/judge_check.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && (pwd -W 2>/dev/null || pwd))"
KIT="${KIT:-$(cd "$REPO/../kits/yug-monsoon-kit" && (pwd -W 2>/dev/null || pwd))}"
NAME="monsoon-judge-run"
docker rm -f "$NAME" >/dev/null 2>&1 || true
T0=$(date +%s)
docker run -d --name "$NAME" --network none --cpus 4 --memory 6g --read-only --tmpfs /tmp \
  -v "$REPO:/sub:ro" -w /sub -e PYTHONDONTWRITEBYTECODE=1 \
  monsoon-judge python serve.py --port 8000 >/dev/null
docker run --rm --network "container:$NAME" -v "$KIT:/kit:ro" -e PYTHONPATH=/kit -e PYTHONDONTWRITEBYTECODE=1 \
  monsoon-judge python -m monsoon_r1.schema_validator http://127.0.0.1:8000
echo "--- ready+validated in $(( $(date +%s) - T0 ))s"
docker stats --no-stream --format 'mem={{.MemUsage}} cpu={{.CPUPerc}}' "$NAME"
docker logs "$NAME" 2>&1 | tail -5
docker rm -f "$NAME" >/dev/null
