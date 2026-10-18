# AGENTS.md

Rules for humans and agents who work on timestral.cpp.

## What this project is

A C++17/ggml port of the Timestral-209M time-series forecaster. It runs one
self-contained GGUF file on CPU, with no Python at inference. The work order is
`docs/PLAN.md`.

## Rules

1. Parity first. A milestone is done only when its gate passes with the
   tolerances in section D of the plan. Do not loosen a tolerance to make a test
   pass.
2. Metadata-driven. Every dimension and setting comes from GGUF KV. No hardcoded
   model dimensions in code, no sidecar config.
3. A comparison helper that always passes is a bug. Every helper in
   `tests/parity.hpp` has tests with known-bad inputs in
   `tests/test_parity_helpers.cpp`. Keep them when you change a helper.
4. Scratch output (models, baselines, virtual environments, logs) goes under
   `$TIMESTRAL_WORK` (default `./work`, ignored by git). Never commit model
   weights or baselines made from the real model.
5. Test return codes: 0 pass, 1 fail, 77 skip (a missing input). A skip is not a
   pass for a milestone gate.
6. No absolute local paths, host names or private notes in the repository.
7. Do not add a `LICENSE` file until the code licence is chosen.

## Build and test

```bash
git submodule update --init
cmake -B build -DTIMESTRAL_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Layout

See the repository layout in `docs/PLAN.md`, section E.
