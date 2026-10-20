# M6b final replay evidence

Implementation revision: `c00921147f0884460d0beb0256811cddd7210522`.
The evidence commit changes documentation only, after replay completes.

Source archive SHA-256:
`90202d0bc07afee29dfa49d535a0d1abb9bb24d83426cfaf6172f4ca064debab`.

Archived executable SHA-256:
`a4b1099b78adc07586ed8197a5702a9a59661fa9ec83e921815de9320184578c`.

The archived executable is rebuilt from the committed worktree before replay.
All 100 tracked regular files, source archive, and executable pass their
before/after SHA-256 checks. The ggml submodule remains
`e705c5fed490514458bdd2eaddc43bd098fcce9b`. No source edits occur during replay.

## Final counts

| Suite | Cases | Teacher blocks | Free blocks | Finals | Comparisons | Negative controls | Worst ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny + tiny-bias | 28 | 164 | 164 | 28 | 684 | 684 | 0.618082 |
| 209M | 15 | 36 | 36 | 15 | 159 | 159 | 0.785359 |

Both executables return zero, selection `all`, no skipped or failed cases.
Both signs and every block run. All three models pass I6.4 bitwise sign symmetry
with clipping off, I6.5 one versus two threads and repeats in one process, and
I6.6 shape and quantile order. The candidate fast regression passes 17/17 tests.

The final tiny replay takes 24.39 seconds, peak RSS 648124 KiB. The final real
replay takes 3:00:56, peak RSS 3460812 KiB. Each runs serially under a 12 GiB
address-space limit, CPU only. `ctx_cap` and `bench_32k` both finish.

## Commands and scratch artifacts

Scratch paths below are relative to the work directory, not repository paths.
`timestral-native/m6b/test_forecast-pinned` is the archived executable.
`timestral-acceptance-v2/suite` is the unchanged five-mode accepted suite.
The model is `timestral-m0b/models/timestral-209m-f32.gguf`.

```sh
export TIMESTRAL_TEST_TINY="$WORK/timestral-acceptance-v2/suite"
export TIMESTRAL_TEST_REAL_BASELINES="$TIMESTRAL_TEST_TINY"
export TIMESTRAL_TEST_GGUF="$WORK/timestral-m0b/models/timestral-209m-f32.gguf"
export TIMESTRAL_TEST_REAL_CASES=all
/usr/bin/time -v python3 "$WORK/timestral-acceptance-v2/limit-as.py"   "$WORK/timestral-native/m6b/test_forecast-pinned"
/usr/bin/time -v python3 "$WORK/timestral-acceptance-v2/limit-as.py"   "$WORK/timestral-native/m6b/test_forecast-pinned" --real
```

| Artifact under `timestral-native/m6b` | Meaning |
|---|---|
| `red.log` | Initial missing-header compilation failure before implementation |
| `build-gates.log` | Transient missing thread-setting symbol during development |
| `build-gates2.log`, `unit-gates.log` | Rebuilt stale object and passing unit probe |
| `tiny-focused.log` | Six focused tiny cases, 486 comparisons, zero failures |
| `tiny-full-dev.log`, `tiny-candidate.log` | Complete tiny development/candidate replay |
| `real-focused.log` | Two real multiblock cases, 42 comparisons, zero failures |
| `fast-regression.log` | Candidate fast CTest: 17/17 pass |
| `committed-build.log` | Build from committed source before executable archive |
| `source-commit.txt`, `source.tar`, `source-archive.sha256` | Source pin |
| `binary-before.sha256`, `test_forecast-pinned` | Archived executable pin |
| `source-files-before.sha256`, `source-files-after-check.log` | 100 tracked files unchanged |
| `final-tiny.stderr` | Final tiny gates, controls, timing, RSS, exit status |
| `final-real.stderr` | Final real gates, controls, timing, RSS, exit status |
| `final-tiny.log`, `final-real.log` | Redirected stdout (gate output uses stderr) |

The focused real development invariant probe costs about 20 minutes before
the first case line because it uses a full trained block plus a remainder.
The final probe uses two one-patch blocks and logs each call. Negative controls
are deliberately corrupt and print FAIL; each final control is enclosed by
`G6_NEGATIVE_EXPECT_FAIL` and `G6_NEGATIVE_REJECTED`. None is a parity failure.

## Per-case and per-block refined error

Values use normalized-v1, with the explicitly logged fixed f32 block anchor.
G6.2 uses teacher `block_refined`; G6.3 uses free-running `refined`. Final rows
use the original positive-pass target anchor. Raw floor keys are not reused.
The full logs also contain every prediction-only comparison, frame, floor,
and tolerance. All rows below pass.

| Case | Sign/block | G6.2 error | G6.2 tolerance | G6.3 error | G6.3 tolerance |
|---|---|---:|---:|---:|---:|
| tiny/uni_short | pos.b0. | 0 | 3.97149722e-06 | 0 | 3.97149722e-06 |
| tiny/uni_short | neg.b0. | 0 | 3.49376727e-06 | 0 | 3.49376727e-06 |
| tiny/uni_short | final | — | — | 0 | 3.73760147e-06 |
| tiny/uni_one_patch | pos.b0. | 3.362739e-07 | 9.53674316e-07 | 3.362739e-07 | 9.53674316e-07 |
| tiny/uni_one_patch | neg.b0. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny/uni_one_patch | final | — | — | 0 | 9.53674316e-07 |
| tiny/multi | pos.b0. | 0 | 2.56585446e-06 | 0 | 2.56585446e-06 |
| tiny/multi | neg.b0. | 0 | 2.1183065e-06 | 0 | 2.1183065e-06 |
| tiny/multi | final | — | — | 0 | 2.83213486e-06 |
| tiny/past_cov | pos.b0. | 6.36464042e-09 | 9.81008394e-07 | 6.36464042e-09 | 9.81008394e-07 |
| tiny/past_cov | neg.b0. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny/past_cov | final | — | — | 0 | 9.70699905e-07 |
| tiny/known_cov | pos.b0. | 1.39912609e-07 | 1.34551379e-06 | 1.39912609e-07 | 1.34551379e-06 |
| tiny/known_cov | neg.b0. | 6.28206033e-07 | 1.36922334e-06 | 6.28206033e-07 | 1.36922334e-06 |
| tiny/known_cov | final | — | — | 2.44524372e-07 | 1.38368382e-06 |
| tiny/cat_uni | pos.b0. | 7.55659044e-07 | 1.34729398e-06 | 7.55659044e-07 | 1.34729398e-06 |
| tiny/cat_uni | neg.b0. | 8.37711685e-07 | 1.87000583e-06 | 8.37711685e-07 | 1.87000583e-06 |
| tiny/cat_uni | final | — | — | 4.46854841e-08 | 9.53674316e-07 |
| tiny/cat_multi | pos.b0. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny/cat_multi | neg.b0. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny/cat_multi | final | — | — | 0 | 9.53674316e-07 |
| tiny/long_h | pos.b0. | 0 | 4.99497196e-06 | 0 | 4.99497196e-06 |
| tiny/long_h | pos.b1. | 0 | 3.98246748e-06 | 0 | 4.01044442e-06 |
| tiny/long_h | pos.b2. | 0 | 4.54414594e-06 | 0 | 4.51109795e-06 |
| tiny/long_h | pos.b3. | 0 | 4.53911053e-06 | 0 | 4.55630386e-06 |
| tiny/long_h | pos.b4. | 0 | 3.88014339e-06 | 0 | 3.88274847e-06 |
| tiny/long_h | pos.b5. | 0 | 4.91440477e-06 | 0 | 4.93964015e-06 |
| tiny/long_h | pos.b6. | 0 | 5.28039189e-06 | 0 | 5.25973312e-06 |
| tiny/long_h | pos.b7. | 0 | 5.52681733e-06 | 0 | 5.51694753e-06 |
| tiny/long_h | pos.b8. | 0 | 5.562832e-06 | 0 | 5.55852009e-06 |
| tiny/long_h | pos.b9. | 0 | 5.04692278e-06 | 0 | 5.1046528e-06 |
| tiny/long_h | pos.b10. | 0 | 5.38323835e-06 | 0 | 5.43947042e-06 |
| tiny/long_h | pos.b11. | 0 | 5.96751837e-06 | 0 | 6.03034142e-06 |
| tiny/long_h | pos.b12. | 0 | 5.36682537e-06 | 0 | 5.30273816e-06 |
| tiny/long_h | pos.b13. | 0 | 4.53424482e-06 | 0 | 4.49122448e-06 |
| tiny/long_h | pos.b14. | 0 | 6.25177744e-06 | 0 | 6.21836012e-06 |
| tiny/long_h | pos.b15. | 0 | 6.65269817e-06 | 0 | 6.69128815e-06 |
| tiny/long_h | pos.b16. | 0 | 5.98780513e-06 | 0 | 6.04316124e-06 |
| tiny/long_h | pos.b17. | 0 | 6.33720376e-06 | 0 | 6.28002838e-06 |
| tiny/long_h | pos.b18. | 0 | 5.81062761e-06 | 0 | 5.85825328e-06 |
| tiny/long_h | pos.b19. | 2.52555148e-06 | 6.31313279e-06 | 2.52555148e-06 | 6.3983351e-06 |
| tiny/long_h | neg.b0. | 0 | 4.22838196e-06 | 0 | 4.22838196e-06 |
| tiny/long_h | neg.b1. | 0 | 4.33763934e-06 | 0 | 4.37089769e-06 |
| tiny/long_h | neg.b2. | 0 | 4.53323829e-06 | 0 | 4.49965666e-06 |
| tiny/long_h | neg.b3. | 0 | 4.89287921e-06 | 0 | 4.93858495e-06 |
| tiny/long_h | neg.b4. | 0 | 5.32436934e-06 | 0 | 5.3741941e-06 |
| tiny/long_h | neg.b5. | 0 | 4.42636676e-06 | 0 | 4.50514803e-06 |
| tiny/long_h | neg.b6. | 0 | 5.15712233e-06 | 0 | 5.07773806e-06 |
| tiny/long_h | neg.b7. | 0 | 4.54870461e-06 | 0 | 4.45547549e-06 |
| tiny/long_h | neg.b8. | 0 | 4.43394433e-06 | 0 | 4.5222528e-06 |
| tiny/long_h | neg.b9. | 0 | 5.98121167e-06 | 0 | 5.86748504e-06 |
| tiny/long_h | neg.b10. | 0 | 5.52907363e-06 | 0 | 5.62547586e-06 |
| tiny/long_h | neg.b11. | 0 | 5.28869949e-06 | 0 | 5.18439328e-06 |
| tiny/long_h | neg.b12. | 0 | 5.14393723e-06 | 0 | 5.2249899e-06 |
| tiny/long_h | neg.b13. | 0 | 4.81697666e-06 | 0 | 4.75262498e-06 |
| tiny/long_h | neg.b14. | 0 | 5.59366705e-06 | 0 | 5.67888488e-06 |
| tiny/long_h | neg.b15. | 0 | 6.30915381e-06 | 0 | 6.22583527e-06 |
| tiny/long_h | neg.b16. | 0 | 6.74650697e-06 | 0 | 6.66042025e-06 |
| tiny/long_h | neg.b17. | 0 | 6.25285145e-06 | 0 | 6.33253415e-06 |
| tiny/long_h | neg.b18. | 0 | 4.82699894e-06 | 0 | 4.74364384e-06 |
| tiny/long_h | neg.b19. | 0 | 5.3106111e-06 | 0 | 5.23942325e-06 |
| tiny/long_h | final | — | — | 0 | 5.91581175e-06 |
| tiny/long_h_known | pos.b0. | 1.42754951e-07 | 9.53674316e-07 | 1.42754951e-07 | 9.53674316e-07 |
| tiny/long_h_known | pos.b1. | 1.4326722e-07 | 9.53674316e-07 | 1.4326722e-07 | 9.53674316e-07 |
| tiny/long_h_known | pos.b2. | 2.31742669e-07 | 9.53674316e-07 | 2.31742669e-07 | 9.53674316e-07 |
| tiny/long_h_known | pos.b3. | 0 | 9.53674316e-07 | 2.39515926e-08 | 9.53674316e-07 |
| tiny/long_h_known | pos.b4. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny/long_h_known | pos.b5. | 2.24816914e-07 | 9.53674316e-07 | 2.24816914e-07 | 9.53674316e-07 |
| tiny/long_h_known | pos.b6. | 5.1589109e-08 | 9.53674316e-07 | 5.1589109e-08 | 9.53674316e-07 |
| tiny/long_h_known | pos.b7. | 1.51555524e-07 | 9.53674316e-07 | 1.51555524e-07 | 9.53674316e-07 |
| tiny/long_h_known | pos.b8. | 2.42249321e-07 | 9.53674316e-07 | 2.42249321e-07 | 9.53674316e-07 |
| tiny/long_h_known | neg.b0. | 1.39100208e-07 | 9.53674316e-07 | 1.39100208e-07 | 9.53674316e-07 |
| tiny/long_h_known | neg.b1. | 1.52297973e-07 | 9.53674316e-07 | 1.52297973e-07 | 9.53674316e-07 |
| tiny/long_h_known | neg.b2. | 1.43695938e-07 | 9.53674316e-07 | 1.43695938e-07 | 9.53674316e-07 |
| tiny/long_h_known | neg.b3. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny/long_h_known | neg.b4. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny/long_h_known | neg.b5. | 1.51971601e-07 | 9.53674316e-07 | 1.51971601e-07 | 9.53674316e-07 |
| tiny/long_h_known | neg.b6. | 4.57917356e-08 | 9.53674316e-07 | 1.41372575e-07 | 9.53674316e-07 |
| tiny/long_h_known | neg.b7. | 2.96852562e-07 | 9.53674316e-07 | 2.96852562e-07 | 9.53674316e-07 |
| tiny/long_h_known | neg.b8. | 1.40843209e-07 | 9.53674316e-07 | 1.40843209e-07 | 9.53674316e-07 |
| tiny/long_h_known | final | — | — | 1.52063488e-07 | 9.53674316e-07 |
| tiny/nans | pos.b0. | 0 | 9.93831256e-07 | 0 | 9.93831256e-07 |
| tiny/nans | neg.b0. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny/nans | final | — | — | 0 | 9.53674316e-07 |
| tiny/nonneg | pos.b0. | 0 | 0.000835365006 | 0 | 0.000835365006 |
| tiny/nonneg | neg.b0. | 0 | 0.000684540935 | 0 | 0.000684540935 |
| tiny/nonneg | final | — | — | 0 | 0.000986210089 |
| tiny/const | pos.b0. | 0 | 0.683685842 | 0 | 0.683685842 |
| tiny/const | neg.b0. | 0 | 0.683685842 | 0 | 0.683685842 |
| tiny/const | final | — | — | 0 | 0.642114172 |
| tiny/all_missing | pos.b0. | 2.15645168e-08 | 9.53674316e-07 | 2.15645168e-08 | 9.53674316e-07 |
| tiny/all_missing | neg.b0. | 7.95441572e-08 | 9.53674316e-07 | 7.95441572e-08 | 9.53674316e-07 |
| tiny/all_missing | final | — | — | 3.33845835e-08 | 9.53674316e-07 |
| tiny/ctx_cap | pos.b0. | 0 | 1.46779129e-06 | 0 | 1.46779129e-06 |
| tiny/ctx_cap | neg.b0. | 8.83457143e-07 | 1.42935326e-06 | 8.83457143e-07 | 1.42935326e-06 |
| tiny/ctx_cap | final | — | — | 8.78647616e-07 | 1.94108492e-06 |
| tiny-bias/uni_short | pos.b0. | 0 | 3.68689979e-06 | 0 | 3.68689979e-06 |
| tiny-bias/uni_short | neg.b0. | 0 | 3.54398561e-06 | 0 | 3.54398561e-06 |
| tiny-bias/uni_short | final | — | — | 0 | 4.01329097e-06 |
| tiny-bias/uni_one_patch | pos.b0. | 1.86477407e-07 | 9.53674316e-07 | 1.86477407e-07 | 9.53674316e-07 |
| tiny-bias/uni_one_patch | neg.b0. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny-bias/uni_one_patch | final | — | — | 0 | 9.53674316e-07 |
| tiny-bias/multi | pos.b0. | 0 | 2.23989855e-06 | 0 | 2.23989855e-06 |
| tiny-bias/multi | neg.b0. | 0 | 2.81371689e-06 | 0 | 2.81371689e-06 |
| tiny-bias/multi | final | — | — | 0 | 2.44272727e-06 |
| tiny-bias/past_cov | pos.b0. | 0 | 1.16276593e-06 | 0 | 1.16276593e-06 |
| tiny-bias/past_cov | neg.b0. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny-bias/past_cov | final | — | — | 0 | 1.09394621e-06 |
| tiny-bias/known_cov | pos.b0. | 1.41386664e-07 | 1.45663623e-06 | 1.41386664e-07 | 1.45663623e-06 |
| tiny-bias/known_cov | neg.b0. | 2.54141028e-07 | 1.19000917e-06 | 2.54141028e-07 | 1.19000917e-06 |
| tiny-bias/known_cov | final | — | — | 0 | 1.37700769e-06 |
| tiny-bias/cat_uni | pos.b0. | 7.53914442e-07 | 1.37654803e-06 | 7.53914442e-07 | 1.37654803e-06 |
| tiny-bias/cat_uni | neg.b0. | 7.4406475e-07 | 1.35215144e-06 | 7.4406475e-07 | 1.35215144e-06 |
| tiny-bias/cat_uni | final | — | — | 9.89962355e-08 | 9.53674316e-07 |
| tiny-bias/cat_multi | pos.b0. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny-bias/cat_multi | neg.b0. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny-bias/cat_multi | final | — | — | 0 | 9.53674316e-07 |
| tiny-bias/long_h | pos.b0. | 0 | 4.75576281e-06 | 0 | 4.75576281e-06 |
| tiny-bias/long_h | pos.b1. | 0 | 5.10904663e-06 | 0 | 5.11077353e-06 |
| tiny-bias/long_h | pos.b2. | 0 | 4.9349597e-06 | 0 | 4.95980364e-06 |
| tiny-bias/long_h | pos.b3. | 0 | 3.54066791e-06 | 0 | 3.49502087e-06 |
| tiny-bias/long_h | pos.b4. | 0 | 4.59427331e-06 | 0 | 4.64472638e-06 |
| tiny-bias/long_h | pos.b5. | 0 | 4.65192756e-06 | 0 | 4.58414944e-06 |
| tiny-bias/long_h | pos.b6. | 0 | 5.12709461e-06 | 0 | 5.16625519e-06 |
| tiny-bias/long_h | pos.b7. | 0 | 4.9407843e-06 | 0 | 5.01366285e-06 |
| tiny-bias/long_h | pos.b8. | 0 | 5.75521325e-06 | 0 | 5.67324153e-06 |
| tiny-bias/long_h | pos.b9. | 0 | 4.1830738e-06 | 0 | 4.10698243e-06 |
| tiny-bias/long_h | pos.b10. | 0 | 4.18012899e-06 | 0 | 4.10758487e-06 |
| tiny-bias/long_h | pos.b11. | 0 | 5.86134197e-06 | 0 | 5.9276554e-06 |
| tiny-bias/long_h | pos.b12. | 0 | 5.3500276e-06 | 0 | 5.26226403e-06 |
| tiny-bias/long_h | pos.b13. | 0 | 5.00990529e-06 | 0 | 4.93492085e-06 |
| tiny-bias/long_h | pos.b14. | 0 | 6.53068812e-06 | 0 | 6.61196622e-06 |
| tiny-bias/long_h | pos.b15. | 0 | 4.99953117e-06 | 0 | 4.91765261e-06 |
| tiny-bias/long_h | pos.b16. | 0 | 6.12546836e-06 | 0 | 6.05023175e-06 |
| tiny-bias/long_h | pos.b17. | 0 | 5.60647794e-06 | 0 | 5.53913508e-06 |
| tiny-bias/long_h | pos.b18. | 0 | 6.00871309e-06 | 0 | 5.93700268e-06 |
| tiny-bias/long_h | pos.b19. | 0 | 5.89302105e-06 | 0 | 5.82496333e-06 |
| tiny-bias/long_h | neg.b0. | 0 | 5.04216509e-06 | 0 | 5.04216509e-06 |
| tiny-bias/long_h | neg.b1. | 0 | 4.56490967e-06 | 0 | 4.57504943e-06 |
| tiny-bias/long_h | neg.b2. | 0 | 4.6137534e-06 | 0 | 4.6285065e-06 |
| tiny-bias/long_h | neg.b3. | 0 | 4.44589649e-06 | 0 | 4.46027403e-06 |
| tiny-bias/long_h | neg.b4. | 0 | 4.56799355e-06 | 0 | 4.59430848e-06 |
| tiny-bias/long_h | neg.b5. | 0 | 5.04095175e-06 | 0 | 4.96818696e-06 |
| tiny-bias/long_h | neg.b6. | 0 | 5.42182735e-06 | 0 | 5.49341216e-06 |
| tiny-bias/long_h | neg.b7. | 0 | 5.07507569e-06 | 0 | 4.99998371e-06 |
| tiny-bias/long_h | neg.b8. | 0 | 4.58647668e-06 | 0 | 4.53715514e-06 |
| tiny-bias/long_h | neg.b9. | 0 | 5.51037141e-06 | 0 | 5.58021851e-06 |
| tiny-bias/long_h | neg.b10. | 0 | 5.32805878e-06 | 0 | 5.43164344e-06 |
| tiny-bias/long_h | neg.b11. | 0 | 5.55990624e-06 | 0 | 5.66786235e-06 |
| tiny-bias/long_h | neg.b12. | 0 | 5.78211907e-06 | 0 | 5.90219077e-06 |
| tiny-bias/long_h | neg.b13. | 0 | 5.75747603e-06 | 0 | 5.89104773e-06 |
| tiny-bias/long_h | neg.b14. | 0 | 4.50185241e-06 | 0 | 4.80051085e-06 |
| tiny-bias/long_h | neg.b15. | 0 | 5.64068912e-06 | 0 | 5.50365183e-06 |
| tiny-bias/long_h | neg.b16. | 0 | 5.60323015e-06 | 0 | 5.40588641e-06 |
| tiny-bias/long_h | neg.b17. | 0 | 6.00096919e-06 | 0 | 5.81929658e-06 |
| tiny-bias/long_h | neg.b18. | 0 | 5.96153769e-06 | 0 | 6.14720978e-06 |
| tiny-bias/long_h | neg.b19. | 0 | 7.16114402e-06 | 0 | 7.35400649e-06 |
| tiny-bias/long_h | final | — | — | 0 | 6.54832648e-06 |
| tiny-bias/long_h_known | pos.b0. | 1.42546092e-07 | 9.53674316e-07 | 1.42546092e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | pos.b1. | 2.13612592e-07 | 9.53674316e-07 | 2.13612592e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | pos.b2. | 1.6576792e-07 | 9.53674316e-07 | 1.6576792e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | pos.b3. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny-bias/long_h_known | pos.b4. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny-bias/long_h_known | pos.b5. | 1.42800347e-07 | 9.53674316e-07 | 1.42800347e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | pos.b6. | 4.60915822e-08 | 9.53674316e-07 | 4.60915822e-08 | 9.53674316e-07 |
| tiny-bias/long_h_known | pos.b7. | 1.41667485e-07 | 9.53674316e-07 | 1.41667485e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | pos.b8. | 1.39439793e-07 | 9.53674316e-07 | 1.39439793e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | neg.b0. | 2.11697596e-07 | 9.53674316e-07 | 2.11697596e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | neg.b1. | 1.61076407e-07 | 9.53674316e-07 | 1.61076407e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | neg.b2. | 1.50321942e-07 | 9.53674316e-07 | 1.50321942e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | neg.b3. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny-bias/long_h_known | neg.b4. | 0 | 9.53674316e-07 | 0 | 9.53674316e-07 |
| tiny-bias/long_h_known | neg.b5. | 2.72566994e-07 | 9.53674316e-07 | 2.72566994e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | neg.b6. | 1.4086474e-07 | 9.53674316e-07 | 1.4086474e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | neg.b7. | 1.41778945e-07 | 9.53674316e-07 | 1.41778945e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | neg.b8. | 2.23968685e-07 | 9.53674316e-07 | 2.23968685e-07 | 9.53674316e-07 |
| tiny-bias/long_h_known | final | — | — | 1.77252674e-07 | 9.53674316e-07 |
| tiny-bias/nans | pos.b0. | 6.4119221e-09 | 9.53674316e-07 | 6.4119221e-09 | 9.53674316e-07 |
| tiny-bias/nans | neg.b0. | 1.18799589e-07 | 9.53674316e-07 | 1.18799589e-07 | 9.53674316e-07 |
| tiny-bias/nans | final | — | — | 0 | 9.60093551e-07 |
| tiny-bias/nonneg | pos.b0. | 0 | 0.000726896943 | 0 | 0.000726896943 |
| tiny-bias/nonneg | neg.b0. | 0 | 0.000856786919 | 0 | 0.000856786919 |
| tiny-bias/nonneg | final | — | — | 0 | 0.000931572842 |
| tiny-bias/const | pos.b0. | 0 | 0.694780635 | 0 | 0.694780635 |
| tiny-bias/const | neg.b0. | 0 | 0.694780635 | 0 | 0.694780635 |
| tiny-bias/const | final | — | — | 0 | 0.638962622 |
| tiny-bias/all_missing | pos.b0. | 2.72496515e-07 | 9.53674316e-07 | 2.72496515e-07 | 9.53674316e-07 |
| tiny-bias/all_missing | neg.b0. | 3.16757773e-07 | 9.53674316e-07 | 3.16757773e-07 | 9.53674316e-07 |
| tiny-bias/all_missing | final | — | — | 1.60093349e-07 | 9.53674316e-07 |
| tiny-bias/ctx_cap | pos.b0. | 0 | 1.61782136e-06 | 0 | 1.61782136e-06 |
| tiny-bias/ctx_cap | neg.b0. | 0 | 1.34053655e-06 | 0 | 1.34053655e-06 |
| tiny-bias/ctx_cap | final | — | — | 0 | 2.47619393e-06 |
| real/uni_short | pos.b0. | 1.76149306e-06 | 4.39094264e-06 | 1.76149306e-06 | 4.39094264e-06 |
| real/uni_short | neg.b0. | 1.92342676e-06 | 4.66463001e-06 | 1.92342676e-06 | 4.66463001e-06 |
| real/uni_short | final | — | — | 1.92532228e-06 | 4.44891896e-06 |
| real/uni_one_patch | pos.b0. | 5.51719499e-07 | 1.02807109e-06 | 5.51719499e-07 | 1.02807109e-06 |
| real/uni_one_patch | neg.b0. | 7.5591254e-07 | 1.28165727e-06 | 7.5591254e-07 | 1.28165727e-06 |
| real/uni_one_patch | final | — | — | 6.67956388e-07 | 1.09497469e-06 |
| real/multi | pos.b0. | 1.54498558e-06 | 2.3843325e-06 | 1.54498558e-06 | 2.3843325e-06 |
| real/multi | neg.b0. | 1.26435246e-06 | 2.58017401e-06 | 1.26435246e-06 | 2.58017401e-06 |
| real/multi | final | — | — | 1.32153982e-06 | 2.58798491e-06 |
| real/past_cov | pos.b0. | 1.39317148e-06 | 2.41178899e-06 | 1.39317148e-06 | 2.41178899e-06 |
| real/past_cov | neg.b0. | 7.19992433e-07 | 2.08524777e-06 | 7.19992433e-07 | 2.08524777e-06 |
| real/past_cov | final | — | — | 7.70938032e-07 | 1.40008854e-06 |
| real/known_cov | pos.b0. | 1.25490612e-06 | 3.26344163e-06 | 1.25490612e-06 | 3.26344163e-06 |
| real/known_cov | neg.b0. | 9.38503018e-07 | 2.62006873e-06 | 9.38503018e-07 | 2.62006873e-06 |
| real/known_cov | final | — | — | 8.63221183e-07 | 1.57702506e-06 |
| real/cat_uni | pos.b0. | 7.57498221e-06 | 9.98456274e-06 | 7.57498221e-06 | 9.98456274e-06 |
| real/cat_uni | neg.b0. | 3.0688182e-06 | 6.80266114e-06 | 3.0688182e-06 | 6.80266114e-06 |
| real/cat_uni | final | — | — | 5.65009318e-07 | 1.19043051e-06 |
| real/cat_multi | pos.b0. | 8.80751154e-07 | 1.74294024e-06 | 8.80751154e-07 | 1.74294024e-06 |
| real/cat_multi | neg.b0. | 1.34172594e-06 | 1.98845172e-06 | 1.34172594e-06 | 1.98845172e-06 |
| real/cat_multi | final | — | — | 5.05897978e-07 | 1.25821754e-06 |
| real/long_h | pos.b0. | 6.31775296e-06 | 2.40060247e-05 | 6.31775296e-06 | 2.40060247e-05 |
| real/long_h | pos.b1. | 4.44165393e-06 | 4.80500295e-05 | 1.59967163e-05 | 6.42940673e-05 |
| real/long_h | pos.b2. | 3.43202412e-06 | 8.36381896e-06 | 1.93237591e-05 | 2.64875635e-05 |
| real/long_h | neg.b0. | 4.79193673e-06 | 2.72825815e-05 | 4.79193673e-06 | 2.72825815e-05 |
| real/long_h | neg.b1. | 4.21341741e-06 | 2.56181016e-05 | 1.91031174e-05 | 8.39151024e-05 |
| real/long_h | neg.b2. | 3.93077756e-06 | 1.16416941e-05 | 2.34789284e-05 | 7.60433136e-05 |
| real/long_h | final | — | — | 1.22430088e-05 | 5.23428026e-05 |
| real/long_h_known | pos.b0. | 3.23549213e-06 | 1.0397325e-05 | 3.23549213e-06 | 1.0397325e-05 |
| real/long_h_known | pos.b1. | 1.32788644e-06 | 6.49657986e-06 | 4.30252469e-06 | 1.68193943e-05 |
| real/long_h_known | neg.b0. | 4.04797578e-06 | 7.12895743e-06 | 4.04797578e-06 | 7.12895743e-06 |
| real/long_h_known | neg.b1. | 1.15182545e-06 | 2.3837863e-06 | 1.1518255e-06 | 9.3874235e-06 |
| real/long_h_known | final | — | — | 1.43263928e-06 | 6.80408324e-06 |
| real/nans | pos.b0. | 1.06456644e-06 | 2.52479468e-06 | 1.06456644e-06 | 2.52479468e-06 |
| real/nans | neg.b0. | 1.36968466e-06 | 1.74546144e-06 | 1.36968466e-06 | 1.74546144e-06 |
| real/nans | final | — | — | 7.65210732e-07 | 1.09735487e-06 |
| real/nonneg | pos.b0. | 2.13718055e-05 | 6.33027518e-05 | 2.13718055e-05 | 6.33027518e-05 |
| real/nonneg | neg.b0. | 1.74706039e-05 | 5.08843884e-05 | 1.74706039e-05 | 5.08843884e-05 |
| real/nonneg | final | — | — | 2.17417026e-05 | 6.01512203e-05 |
| real/const | pos.b0. | 0 | 5.93866277 | 0 | 5.93866277 |
| real/const | neg.b0. | 0 | 5.93866277 | 0 | 5.93866277 |
| real/const | final | — | — | 0 | 5.4279981 |
| real/all_missing | pos.b0. | 1.25471241e-06 | 2.106658e-06 | 1.25471241e-06 | 2.106658e-06 |
| real/all_missing | neg.b0. | 1.25470234e-06 | 2.10664109e-06 | 1.25470234e-06 | 2.10664109e-06 |
| real/all_missing | final | — | — | 5.98397826e-07 | 1.9932665e-06 |
| real/ctx_cap | pos.b0. | 6.96119329e-07 | 1.08789029e-06 | 6.96119329e-07 | 1.08789029e-06 |
| real/ctx_cap | neg.b0. | 5.68404393e-07 | 9.53674316e-07 | 5.68404393e-07 | 9.53674316e-07 |
| real/ctx_cap | final | — | — | 4.15660689e-07 | 9.53674316e-07 |
| real/bench_32k | pos.b0. | 6.26482023e-06 | 3.13082312e-05 | 6.26482023e-06 | 3.13082312e-05 |
| real/bench_32k | neg.b0. | 4.52495614e-06 | 5.60149848e-05 | 4.52495614e-06 | 5.60149848e-05 |
| real/bench_32k | final | — | — | 2.66180932e-06 | 2.83527345e-05 |
