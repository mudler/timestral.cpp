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
