# Parity notes

Measured facts about the parity gates. Tolerances are in `docs/PLAN.md`, section D; this file
records what was measured. It does not change a gate.

## G0.3: stored rope tables vs the tables the reference uses at run time

Preview made in M0b by `tests/python/check_convert.py` (the full gate is M0d). It runs the
reference `SinCosRoPE.forward` on run-time-like layouts and compares every value with the
stored table row of its position.

- Layouts: R rows in {1, 2, 3, 6, 17}; N patches in {1, 2, 3, 5, 8, 33, 100, and the largest
  position count the model can reach, capped at 1056}; three random `first_obs` draws per
  pair; positions `p - first_obs`, clamped to the table range.
- Timestral-209M (head_dim 64): 13,452,288 values compared, 0 differ in value (max 0 ulp),
  1,786,880 differ only in the sign of a zero.
- Tiny checkpoint (head_dim 8): 573,504 values, 0 differ in value, 88,500 differ only in the
  sign of a zero.
- An independent review scan found 0 value differences in 176,748,160 values; every
  difference was a signed zero.

The sign-of-zero differences are in the dims with `inv_freq = 0` (the identity rotation), where
`sin` is `+0` or `-0` depending on the sign of the position and on the layout. They are
counted apart from value differences, as the plan's D.1 requires.

An earlier, unconventional scan of the tiny checkpoint (one position per row, very few patches)
showed many bit differences. They were counted by raw bit pattern, so they were these same
signed zeros, not value differences. The gate is unchanged.

## Floor computation and publication

`compute_floors.py --dir DIR` processes every discovered f32 case in that directory.
It does not prove that the canonical suite is complete. Use `check_baseline.py` for
full milestone acceptance. Repeated `--case NAME` options select a partial run;
an unknown or missing selected case is an error. A partial run replaces the summary
with the selected cases only.

Every selected numerical case requires all five modes. The shared contract checks
case/model/source metadata, required records, shapes, precision, statistics,
reproducibility, and the f64 rerun self-check before publication. Exception artifacts
cannot supply numerical floors. Companion source identifiers must be present and agree;
this validates recorded provenance, not the authenticity of the source installation.

The C.4 denominators are always f64 and f64-rerun respectively. Both companion
numerators are mandatory. Required rerun outputs always receive `floor_tf` metadata.
For refined blocks, the G6.2 floor uses `block_refined`, not isolated `refined`.
A zero reference RMS produces zero only for zero error; a mismatch is an error.
Nonfinite evidence, statistics, floors, and caps are errors. Raw context copies may
retain NaNs only at matching comparison positions. Infinities are not accepted.

Caps remain warning-only. The gate tolerance remains `max(2 * floor, 16 * u)`.
Missing required statistics are errors, never `n/a`. Summary entries count cap
applicability separately as `bounded`, `copy`, or `na`; a zero bounded cap is not a copy.
The existing explicit exceptions in `EXTRA_NOCAP` cover raw copies (`input`, `args`,
`assemble`, `scale.in`, `refine.context`, `cap.rows`) and host/intermediate values
without a declared bound (normalization inputs, scaling, frames, refinement helpers,
masks, positions, rotary tables, and monotone outputs). Unmatched float names fail.

Final raw caps resolve through the matching `pos.b0` or `neg.b0` prefix.
For single-block `final.quantiles`, the bound is the larger of the two mirrored
single-block bounds. Final outputs with two or more blocks have no free-running cap.
Per-block `predict` and `refined` after block zero also have no free-running cap.
SDPA uses the stored key tensor's sequence axis for `S`, including variate attention.
For split time attention, the full key sequence bounds both the encoder prefix and
the decoder's full-key call; concatenated query length is not the definition of `S`.

The command preflights every selected case before writing any f32 metadata.
It reads cases again during publication to avoid retaining the whole suite in memory.
Do not run concurrent generators or writers against the directory. Concurrent input
changes between preflight and publication are outside this guarantee.

Each GGUF and JSON write uses a sibling temporary file, closes or flushes it,
fsyncs its contents, and replaces the destination atomically. Failure before replacement
preserves the original bytes and removes the temporary file. Metadata updates preserve
all stored tensor values and dtypes. Tests use generated synthetic files only.

This is not a directory transaction. An I/O failure or crash can leave some cases
updated and others unchanged, or leave an older summary. Abrupt process termination
can leave sibling temporary files. The code does not fsync the parent directory, so
power-loss durability of the rename depends on the filesystem. Repeat the command
after resolving the failure; no partially written destination file is published.
