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
