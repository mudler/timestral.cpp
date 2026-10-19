#!/usr/bin/env bash
# Regenerate the baselines of the tiny checkpoints (all five modes, both variants).
# Usage: scripts/regen_tiny_baselines.sh
# Environment: TIMESTRAL_WORK (default ./work), VENV (default $TIMESTRAL_WORK/venv-ref),
#              TIMESTRAL_TEST_TINY (default tests/fixtures/tiny), CASES (default tests/cases.json).
# The tiny checkpoints must exist in $TIMESTRAL_WORK/tiny and $TIMESTRAL_WORK/tiny-bias
# (scripts/make_tiny_checkpoint.py). The modes run in the order the reruns need.
set -euo pipefail
TIMESTRAL_WORK=${TIMESTRAL_WORK:-$PWD/work}
VENV=${VENV:-${TIMESTRAL_REF_VENV:-$TIMESTRAL_WORK/venv-ref}}
TIMESTRAL_TEST_TINY=${TIMESTRAL_TEST_TINY:-$PWD/tests/fixtures/tiny}
CASES=${CASES:-tests/cases.json}
for v in tiny tiny-bias; do
  for m in f32 f32-math f64 f64-rerun f32-math-rerun; do
    "$VENV/bin/python" scripts/gen_reference_baseline.py --model "$TIMESTRAL_WORK/$v" \
        --cases "$CASES" --mode "$m" --tiny --output-dir "$TIMESTRAL_TEST_TINY/$v"
  done
done
