# Head contraction precision

All three F32 `Head::project` contractions use binary64 products and ascending contracted-dimension summation, then one F32 rounding.
Hidden/output/residual projections follow the same policy for every model and shape.
Activation and residual addition retain their F32 operation boundaries; optional bias follows the rounded F32 dot.
This is an explicit arithmetic-contract change, not a bitwise optimization or universal MKL match.
No global linear, patch projection, attention, or scaler operation changes.
Reduced weights retain the existing GGML dispatch and quality-evaluation contract.

An internal legacy route retains the original head contraction arithmetic for its existing raw-head comparison.
No public runtime precision switch exists.
The legacy test's reference expectations remain unchanged; production head routing has a separate independent full-head oracle.
Direct contraction tests cover K1/7/8/31/32/191/192/193/768/1024, output tails, masks-free dots, cancellation, signed zeros, and nonzero subnormal products.
Actual concurrent callback partitions 1/2/3/6 check byte equality, sentinels, input immutability, and the F32 bias boundary.
The full production-route test independently computes hidden contraction, activation, output/residual contractions, and F32 residual addition.

A regression-first build of the new production-route oracle fails against the previous accepted library before implementation.
Reference artifacts, floors, and tolerances remain unchanged.
## Verified bounded implementation

The production-route test fails against the previous library before the implementation: exit 1 at its binary64-head oracle.
After implementation, 48 full `Head::project` cases and 640 direct contraction cases pass.
The direct tests exercise actual concurrent callback partitions 1/2/3/6, including optional bias after dot rounding.
The existing head test retains all reference expectations and passes 16 tiny cases, 32 blocks, and 96 comparisons through its internal legacy raw-head route.
Its end-to-end prediction checks still exercise production forecasting.

The component selection passes 37/37, including F16/Q8 storage/routing and synthetic K-quant writer tests.
Separate frozen-legacy and new attention oracles pass; no other arithmetic policy changes.
Full tiny forecasting passes 28 cases and 684 comparisons, maximum ratio 0.932042.
The exact 13-case 5M profile passes M3 with 240 local layers and 510 comparisons, no missing layers.
It passes M6b with 32 teacher blocks, 32 free blocks, 13 finals, and 141 comparisons, maximum ratio 0.947251.
Nineteen amplification diagnostics remain visible; the maximum adjacent amplification is 20.8664.
The 209M uni_short/known_cov/all_missing subset passes 27 comparisons, maximum ratio 0.547373.
Tiny and 209M 512/32 repeated/threaded determinism pass at threads 1 and 6.
These bounded results do not replace full current-aggregate replay or model-quality quantization acceptance.

Source archive SHA-256: `a3eb6d6d79c4cc1a028c4846c6c6988dbd410820c682612f5b5a09b28d30f456`.
Inference library SHA-256: `519ecc9c733040905ea80fdbd50e3e6b4ec8226b4666a558c8b3176f653bbe1e`.
Forecast executable SHA-256: `9d660a672dd97b7953f1dc380f516d8fbeba110976327c3790b79912ebea91f3`.
All 1962 vendor blobs match the pinned ggml source.
The candidate executable, model, and all 65 5M reference artifact hashes remain unchanged before/after real gates.
Source checks confirm global linear/attention/scaler/forecaster files are unchanged.
Execution is serial under 12 GiB; builds use at most two compiler jobs.

The optional timing comparison does not run because host load and CPU/I/O pressure fail the quiet-host screen.
No performance improvement, regression estimate, or formal M6c claim follows from this implementation.
Full-model/performance acceptance remains separate from the bounded implementation checks.
