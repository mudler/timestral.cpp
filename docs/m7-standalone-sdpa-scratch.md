# Private standalone SGEMM SDPA callback: bounded scratch result

## Formula and ownership

The callback computes its own QK, F32 scale/mask/max/shift/exp/sum, value SGEMM, and reciprocal normalization.
No stored probabilities, attention outputs, or fixture files are read by the callback.
Public Torch CPU flash loop structure at `5c4886908584029761b579af026dcfb627c84070` informs its ordering; no proprietary implementation is inspected or adapted.
The private scratch directory retains the complete PyTorch licence and scoped attribution alongside the source.
This is not a product dependency, redistribution, or production implementation.

Each context/decoder region uses its own original call geometry; variate attention uses the same canonical adapter with its own sequence axis.
Query tile is32 below192 queries,64 below768, otherwise256; key tile is at most512.
Ascending key chunks retain running maxima/sums and output rescaling; value GEMM uses beta0 then beta1.
Final reciprocal multiplication occurs only after all key chunks.
All C bases are64-byte aligned, with guards outside the write region and unchanged original ldc.
Checked LP64/product guards reject unsupported dimensions before calls.
No key packing, ISA setter, dynamic setter, persistent cache, or reference-selected branch is added.

## Mechanics coverage

Fifteen shapes cover query thresholds191/192/193 and767/768/769; key boundaries511/512/513; splits511/512; small dimensions/tails; multiple batches/heads; and context/decoder shapes.
Serial output matches actual concurrent ith/nth1/2/3/6 byte-for-byte.
All60 partition checks pass over169,032 checked output values; guard bytes and input memcmp checks pass.
All outputs are finite in the valid-mask corpus.
These checks establish deterministic mechanics, not an independently proven full arithmetic oracle.
The maximum reference-order F32 tile storage for hd64 is147,968 floats,591,872 bytes per active worker, plus small alignment/guard overhead.
No full sequence-square score allocation is used per worker.

## Synthetic actual-flash observations

Offline pinned Torch flash calls compare the same generated shapes with the private callback.
Thirteen of fifteen shapes are byte-exact.
S9/split8/hd64/H12 differs in three of6912 values, maximum `1.490116119e-8`.
S513/split512/hd7 differs in two of3591 values, maximum `1.047737896e-9`.
All other threshold/key-chunk observations are exact on this corpus, including S769 and S513 unsplit.
Residuals are report-only and retained; no arithmetic is tuned to remove them.
Exact policy, mathematical accuracy, and reference parity remain distinct claims; no independent mathematical-accuracy proof is inferred here.

## Actual stored model witnesses

The model harness runs the accepted native layer operations, replacing only computed attention boundaries with callback results from actual q/k/v.
For composite layers it recomputes downstream state before the second attention.
Free layers0–3 start from stored tf.in and feed each computed output into the next layer.
No model runtime is relinked to a new production callback; these are observational boundary interventions.

209M layer5, both signs: four standalone/composed comparisons pass.
Positive SDPA ratio is0.0052561445 and composed layer ratio0.0327352497.
Negative ratios are0.0047732697 and0.0148205977.
Small decoder normalization residuals remain; no exact-flash claim is made for those outputs.

5M multi layers0–3, both signs: all26 standalone/local/free comparisons pass and are byte-exact to their respective stored tensors.
The prior accumulated coherent-candidate failure is not reproduced by this reference-order SGEMM witness.
This does not establish all-model acceptance or authorize runtime integration.

## Dependency closure, integrity, and limits

Loaded-object enumeration shows private LP64/sequential/core, default dispatch/VML-default, and system libraries.
No Python, libtorch, OpenMP, or TBB is loaded by the mechanics executable.
Offline reference observation remains a separate Python process, not a callback dependency.
All extracted private runtime/header files match their verified inventory.
All90 canonical/extended references and all65 5M references/models match retained pins after the run.
Accepted library and encoder binaries remain unchanged; no golden/floor/tolerance changes occur.

Callback source SHA-256: `0a961648fea48b15d88e1a2e076acb8c8b5ba8b43d87fce61b2ecde990535e8f`.
Mechanics executable SHA-256: `17bbe8f119172ba981cf035ebe6fc259c49dbad1ed42ce8aad17dff9ebb71764`.
Model-witness executable SHA-256: `e63d5beea0cbd8b1601096cb337f93ffd4b6ffafa46ee0d19f6b1438e9dc925f`.
Execution is serial under12GiB; only small witnesses compile.
No tiny/full5M/full209M replay, performance benchmark, backend adoption, public API change, publication, or push follows.
A separately approved next stage can test isolated runtime integration on broader unchanged gates; this document grants none.
