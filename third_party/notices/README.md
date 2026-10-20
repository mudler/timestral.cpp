# Notices for adapted numerical helpers

These notices apply to the indicated portions of `src/layers.cpp`,
`src/head.cpp`, and `src/torch_sum.hpp`, not to
the entire project. No project-wide code licence is selected here.

## SLEEF (Boost Software License 1.0)

`activation_exp` (in both files) adapts the `xexpf` range reduction and polynomial and the
`L2Uf`, `L2Lf`, and `R_LN2f` constants to scalar C++ with explicit FMA.
The copyright notice and complete licence are retained in `SLEEF.txt`.

Source: SLEEF 3.9.0, commit
`906ca7512ee483296780a81a21b9ca715d40dfe1`:

- https://github.com/shibatch/sleef/blob/906ca7512ee483296780a81a21b9ca715d40dfe1/src/libm/sleefsimdsp.c
  (`xexpf`, including the file's copyright notice)
- https://github.com/shibatch/sleef/blob/906ca7512ee483296780a81a21b9ca715d40dfe1/src/common/misc.h
  (range-reduction constants)
- https://github.com/shibatch/sleef/blob/906ca7512ee483296780a81a21b9ca715d40dfe1/LICENSE.txt

Changes: scalar evaluation replaces vector intrinsics; standard-library
handling covers exceptional inputs and exponent-range edges.

## PyTorch / ATen (BSD-style licence)

`src/torch_sum.hpp::torch_sum_f32` adapts the contiguous F32 CPU reduction from
`multi_row_sum`, `row_sum`, and `vectorized_inner_sum` in the pinned `SumKernel.cpp` below.
Changes: scalar eight-lane arrays replace AVX2 vectors; a contiguous pointer/length API replaces TensorIterator dispatch.
It retains the four-level cascade, four partial groups, scalar tails, and lane-combination order.
`src/scaler.cpp::instance_norm_stats` calls this helper for weighted values, observation counts, and centered squares.
The complete upstream licence retained in `PyTorch.txt` applies to this adapted helper; no project-wide licence is selected.

`rms_square_mean` adapts the contiguous CPU cascade reduction from
`multi_row_sum`, `row_sum`, and `vectorized_inner_sum`. `attention_exp`
adapts the AVX2 `exp_u20` polynomial. The `sdpa_f32` exponential reduction,
normalization order, and context split follow the CPU flash kernel;
`sigmoid_f32` and `silu_f32` follow the activation operation boundaries.
The complete pinned upstream copyright statements, conditions, and
disclaimer are retained verbatim in `PyTorch.txt`.

Source: PyTorch commit `5c4886908584029761b579af026dcfb627c84070`:

- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SumKernel.cpp
- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cpu/vec/vec256/vec256_float.h
- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/Activation.cpp
- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE

Changes: scalar lanes and explicit FMA replace vector operations, and ggml
custom operations replace Torch tensor dispatch. No Torch runtime is linked.
The SLEEF-derived exponential retains its separate Boost licence above.

In `src/head.cpp`, `head_silu_f32` follows the pinned ATen SiLU operation
boundaries from `Activation.cpp`; `monotone_quantiles` follows the CPU cumsum
accumulator and per-prefix float32 cast in `ReduceOpsKernel.cpp` (same PyTorch
revision above). The cumsum traversal itself is the model's inward-to-outward
quantile order. These are scalar C++ adaptations, not a project-wide licence.

- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp

## NumPy (BSD 3-Clause)

`src/numpy_sum.hpp` adapts the contiguous float64 `pairwise_sum` traversal
from `numpy/_core/src/umath/loops_utils.h.src` (NumPy 2.5.4). The eight
accumulator lanes, 128-element leaves, and aligned recursive split retain
NumPy's evaluation order. C++ vectors replace strided NumPy dispatch.
The copyright, conditions, and disclaimer are retained in `NumPy.txt`.

- https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/umath/loops_utils.h.src
