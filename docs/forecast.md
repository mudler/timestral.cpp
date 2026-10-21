# Native forecast API (M6b)

Include `timestral.h`, link `timestral`, and use C++17. The flat C ABI
([capi.md](capi.md)) and the CLI ([cli.md](cli.md)) both call this API.

```cpp
#include "timestral.h"
auto model = ts::Model::load("timestral-209m-f32.gguf");
ts::Forecaster forecaster(model);
ts::ForecastRequest request;
request.batch = 1;
request.targets = 1;
request.length = 5;
request.horizon = 48;
request.context = {1, 2, 3, 4, 5};
auto forecast = forecaster.forecast(request);
```

The result owns row-major `(batch, targets, horizon, quantile_levels.size())`
values and the GGUF probability levels. Request arrays use `(B,V,T)` for targets,
`(B,C,T)` for past covariates, and `(B,C,T+H)` for known covariates. Specify each
covariate row count. An empty observation mask means all finite target values
are observed; zero mask entries mark additional missing values. Categorical
codes use double precision and NaN for missing categories.

The model must outlive its forecaster. A forecaster is not reentrant. Independent
instances can run on independent calling threads. `ts::set_cpu_threads(N)` sets
the execution count for the calling thread, including existing forecasters;
the default is one. No Python runtime or sidecar configuration is required.

`max_context` and `per_shot_horizon_patches` use GGUF defaults when zero. The
shot size cannot exceed the trained block size. Optional averaging, clipping,
and raw-space quantile-cap overrides otherwise use GGUF metadata. Negative or
zero quantile cap disables that cap, as in the reference.

Native preparation reuses M5 assembly: finite target masks, forward filling,
left patch padding, context capping, categorical target encoding, and packed
batch groups. Each decode predicts a block and applies M6a advancing-frame
refinement. For subsequent blocks, targets receive sanitized median feedback,
known rows receive their future values and masks, and past-only rows receive
unobserved zeros. Context sliding retains complete patches. The final block is
trimmed to the requested horizon. The mirrored pass negates the assembled rows
and known future **after** categorical encoding, including indicator rows.

`ForecastTrace` optionally retains every native block's context, masks,
prediction, and refinement for audit. It is not required for ordinary inference.
`decode_state.hpp` supplies the pure scheduler: `D1-feedback` advances a pending
request, and `D2-done` completes it. Duplicate completion events are rejected.
The scheduler tests exhaust bounded horizon/shot combinations and check patch
conservation, transition rule IDs, and termination.

## Acceptance replay

Build with `TIMESTRAL_BUILD_TESTS=ON`. The test exits nonzero for missing inputs,
unknown selections, missing blocks, mismatched provenance, or numerical errors.
No missing artifact is silently counted as a passing case.

```sh
ctest --test-dir build -R '^test_forecast_units$' --output-on-failure
TIMESTRAL_TEST_TINY="$SUITE" build/tests/test_forecast
TIMESTRAL_TEST_REAL_BASELINES="$SUITE" \
TIMESTRAL_TEST_GGUF="$MODEL" build/tests/test_forecast --real
```

The suite contains `tiny`, `tiny-bias`, and `real` subdirectories. All five
established modes are mandatory for each selected case. `TIMESTRAL_TEST_REAL_CASES`
can select an explicit comma-separated development subset; `all` is the default.
Subset results cannot establish full acceptance. The real CTest is labelled
`slow`, including `ctx_cap` and `bench_32k` when selection is `all`.

G6.2 executes native prediction plus refinement on each block's dumped context
and mask, never native feedback. The prediction uses corresponding teacher
rerun `predict` tensors. Combined refinement uses the rerun `block_refined`
alias, not the isolated-refine rerun. G6.3 compares every native free-running
prediction/refinement and the final forecast with free-running companions.

Every comparison calls the reviewed `normalized-v1` metric. The fixed float32
anchor is the dumped block anchor (post known-row replacement). For final
quantiles it is the original positive-pass target anchor, held across the
horizon. Logs print these frames explicitly. Teacher floors are recomputed
from `f64-rerun` and `f32-math-rerun`; free-running floors are recomputed from
`f64` and `f32-math`. Stored raw-space `floor_tf.*` and `floor_fr.*` metadata is
neither reinterpreted nor modified. The rule remains `max(2*floor,16*2^-24)`.
Corrupt-output controls must fail the same comparator.

I6.4 checks bitwise sign symmetry with clipping disabled. I6.5 compares repeated
forecasts in one process and thread counts one and two. I6.6 checks exact output
shape and nondecreasing quantiles. These invariants are not numerical parity.
