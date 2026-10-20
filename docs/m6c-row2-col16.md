# Two-row sixteen-output linear reuse: correctness WIP, performance HOLD

Root approves an isolated experiment against exact standalone-ON runtime68a5a53.
Only scratch source changes; the production kernel is untouched. The candidate
packs Kx16 and processes two cyclically owned rows, sharing each input broadcast
across two AVX output vectors. Four independent panel chains preserve ascending
K FMA order, positive-zero192-wide panels, separateF32 panel addition and bias.
The final single owned row, eight-column remainder, and scalar columns use prior
arithmetic. Eligibility and graph task policy are unchanged. Checked nothrow
RAII allocation falls back before writes; no shared/persistent packing state.

This is distinct from the reverted row4 naming rewrite and the unmeasured
four-row/16-column candidate. It does not optimize exp, SDPA, or vendor code.
No proprietary sampling, disassembly, interposition, or binary modification occurs.

## Actual completed correctness checks

- Optimized and forced-scalar oracle each pass **57024 comparisons /113729616
  output values**. The independent scalar arithmetic oracle remains unchanged.
- Widths15/16/17,31/32/33; row counts1/2/3 and31..39; partitions1/2/3/4/6/48
  exercise zero/one/two/three owned-row cases, idle workers, pair/tile tails.
  Independent ownership enumeration checks disjoint/exhaustive cyclic assignment.
  Existing actual-concurrent subset, bias/no-bias, cancellation/wide finite
  values, signed zero/subnormals, immutable inputs and sentinels remain covered.
- Component/tiny CTest **46/46**, separate loader **1/1**, zero failed tests.
- Full tiny forecast **28 cases /164 teacher blocks /164 free blocks /28 finals /
  684 comparisons**, zero failures, worst ratio **0.932042** for this ON profile.
- Actual forecast bitwise repeats pass all28 tiny cases and real512/32 with
  thread sequence1,6,6,1. Actual tiny output bytes match the accepted ON library.

Allocation-failure injection for the new wide helper is not implemented; its
checked nothrow prewrite fallback is source-inspected. No sanitizer/general
concurrency-completeness claim is made. Nonfinite NaN-payload portability is not
established by finite-input exact tests.

## Admission outcome: no full-shape candidate execution

After immutable pin capture, the bounded admission screen runs31 samples over
600seconds. It never reaches three quiet samples. The final sample records
load1 **13.96**, pressureavg10 **18.52%**, still outside approved thresholds.
`blocked.txt` records no quiet window. Exit0 indicates monitoring completion,
not performance success. No full32k byte-control run, candidate forecast, matched
baseline/candidate timing, or six-thread performance configuration launches.
Earlier real512 determinism is a completed correctness run, not a full32k run.

Performance is **UNDECIDED**. The approved retain-only-on-demonstrated-win rule
prevents adoption. Preserve the isolated candidate for root review; no production
runtime integration, optimization retry, or resource-screen relaxation follows.
The pending full32k byte comparison and warmed matchedN1repeat3 diagnostic need
a new explicit root grant/window. Existing formal M6c latency failure remains.

## Pins and evidence

Before/after-HOLD manifests match all **363 entries**, including155 reference
artifacts,199 SDK files, model, base/vendor/candidate source archives, candidate
CLI/harness/oracle and reviewed baseline CLI/library. A follow-up bounded hash
job closes pins after the main script's admission exit; no model is run during
that closure. No reference/model/SDK content changes.

Evidence under `timestral-m6c-row2-col16` relative to work root: `base.tar`,
`vendor.tar`, prepared/tested source archives, scratchsource, candidate patch,
`configure.log`, `build.log`, `fallback-build.log`, `oracle.log`,
`oracle-fallback.log`, `components.log`, `loader.log`, `tiny684.log`,
`tiny-det.log`, `real-det.log`, actualtiny byte files, `before.json`,
`after-HOLD.json`, screening/hostlogs, `blocked.txt`. The prepared patch/archive
precedes the final idle-partition guard/test extensions; `tested-source.tar`
is the authoritative actually built source. No candidate patch is applied to
the product worktree or committed as production code.

Jobs2958(build/exact),2960(controls),2964(pins/admissionHOLD),2974(closingpins)
are accounted and finished. Hard12GiB and serial-j2 constraints hold throughout.
Root owns any continuation; the heavy slot is released.

## Root-authorized frozen full-shape attempt2: admission HOLD

A new attempt preserves the original600-second HOLD and uses the exact same
frozen candidate/baseline binaries without edits or rebuild. All **363 entries**
are reverified before and after the attempt, including references, SDK, model,
source archives and executables. Independent bounded code review accepts the
finite-input arithmetic/ownership reasoning, but not performance adoption.
Actual concurrent focused coverage remains K193,N33,rows35; allocation failure
fallback remains source-inspected, not injected. Exact matrix count is57024,
not the earlier43200 baseline count.

The new admission screen completes **31 samples /approximately600seconds**,
with maximum quiet streak **zero**. Final load1 is **27.78**, pressureavg10
**56.59%**. It never satisfies the approved three-sample admission condition.
`HOLD.json` explicitly records no model runs. No full32k output files or timing
summary exist; the pending full-byte comparison and matchedN1warmrepeat3 remain
unexecuted. Exit0 means the bounded screening/pin closure completes, not a speed
or numerical gate pass. No new correctness claim is inferred from this monitor.

Evidence under `timestral-m6c-row2-col16/fullshape-attempt2`: `run.py`,
`before.json`, `after.json`, `screening.jsonl`, `admission-00.log` through
`admission-30.log`, and `HOLD.json`. Both attempts remain intact. Job2978 is
finished, no model child is running, and the heavy slot is released. No automatic
retry or production adoption follows; performance remains **UNDECIDED**.

## Separately authorized contended full-shape correctness pair

Root separates correctness progress from unavailable performance admission.
Using the exact frozen binaries with no edits/rebuild, one baseline and one
candidate N1 forecast run at32768/1024 dyadic-v1 under hard12GiB, serially.
No quiet screen, benchmark sampling, latency interpretation, affinity/topology
change, or unrelated process intervention is performed.

Both outputs contain **21504 finite F32 values /86016 bytes**, shape
`[1,1,1024,21]`, and match **byte-for-byte**. Shared output SHA256:
`ec17a1758dbefd78b2bae825464b02e8e78aeb9cf0fdd6a2862d3c54c8f2d4c6`.
This completes the pending full-shape numerical-identity control for this
single input, not a complete trained corpus or performance acceptance.

All **363 source/archive/reference/SDK/model/executable entries** match before/
after. Host snapshots confirm substantial contention (before load43.91,
pressureavg10 59.81%; after load41.67,pressure63.96%). Harness diagnostic clocks
remain in raw stderr but are not interpreted or promoted as timing results.
The candidate is still isolated: **performance UNDECIDED, no adoption**.
No six-thread model/performance run or matched warmed diagnostic is authorized
or launched by this correctness-only grant.

Evidence under `timestral-m6c-row2-col16/fullshape-correctness-only`:
`before.json`, `after.json`, `baseline.bin`, `candidate.bin`, process stdout/
stderr logs, `host-before.log`, `host-after.log`, and `result.json`.
Job2988 exits0 after actual comparison and pin closure. The heavy slot is
released; no retry or next scope starts automatically.
