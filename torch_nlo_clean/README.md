# nlo-torch

PyTorch implementation of NLO BK evolution and NLO DIS structure functions. The package runs on
CPU and has optimized CUDA float32 paths for the validated default BK/DIS configurations.

## Install

Python 3.12 or 3.13 is required.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

The CPU path needs no compiler. The optimized CUDA path also needs an NVIDIA CUDA toolkit with
`nvcc` compatible with the installed PyTorch build. CUDA extensions compile and cache on their
first use. The custom kernels were validated on an RTX 3090; PyTorch selects the architecture of
the visible GPU when it builds them.

## Dipole inputs

All DIS functions accept either an analytic `GBW` amplitude or a `BKDipole` made from a BK table.
Input tensors and table tensors must use the same device and dtype.

### Analytic GBW dipole

```python
import torch
from nlo_torch import DISConfig, DISOrder, F2, GBW, Quark, QuarkType

device = torch.device("cpu")
dtype = torch.float64

dipole = GBW(Qs0sqr=1.0, lambda_=0.3, gamma=1.0, x0=1.0)
config = DISConfig(
    order=DISOrder.NLO,
    quarks=(Quark(QuarkType.LIGHT), Quark(QuarkType.C, 1.4)),
    maxeval=1_000_000,
    epsrel=0.01,
)

result = F2(
    torch.tensor(9.0, device=device, dtype=dtype),
    torch.tensor(1e-3, device=device, dtype=dtype),
    dipole,
    config,
    seed=7,
    batch_size=65_536,
    quadrature_points=24,
)
print(result.value.item(), result.error.item(), result.converged)
```

### Existing legacy BK table

```python
import torch
from nlo_torch import BKDipole, DISConfig, DISOrder, F2, load_bk_table

table = load_bk_table("path/to/bk_table.dat", dtype=torch.float64, device="cpu")
dipole = BKDipole(table)
config = DISConfig(order=DISOrder.NLO)

Q2 = torch.tensor(9.0, dtype=table.r.dtype, device=table.r.device)
xbj = torch.tensor(1e-3, dtype=table.r.dtype, device=table.r.device)
result = F2(Q2, xbj, dipole, config, seed=7)
print(result.value.item(), result.error.item())
```

`load_bk_table` reads the original four-header `nlobk` text format and preserves its `x0` and
rapidity convention.

### Evolve a dipole, then use it in DIS

```python
import torch
from nlo_torch import BKConfig, BKDipole, DISConfig, F2, MV, solve_bk

device = torch.device("cuda")
dtype = torch.float32

bk_config = BKConfig()
table = solve_bk(
    MV(qs0sqr=0.10, x0=0.01, gamma=1.0),
    maxy=1.0,
    config=bk_config,
    device=device,
    dtype=dtype,
    seed=7,
)
dipole = BKDipole(table)

dis_config = DISConfig()
Q2 = torch.tensor(9.0, device=device, dtype=dtype)
xbj = torch.tensor(1e-3, device=device, dtype=dtype)
result = F2(Q2, xbj, dipole, dis_config, seed=7)
print(result.value.item(), result.error.item())
```

For a custom BK initial condition, replace `MV(...)` with
`ICDataFile("initial.dat", x0=0.01, device=device, dtype=dtype)`. The file contains at least ten
ordered rows with two columns: `r N(r)`.

## Command line

Evolve the default MV initial condition on CUDA and write a legacy BK table:

```bash
nlo-bk --device cuda --dtype float32 \
  --maxy 1.0 --seed 7 --output evolved.dat
```

Use a two-column initial condition by adding `--ic-data initial.dat`.

Calculate a DIS grid from a BK table:

```bash
nlo-dis --datafile evolved.dat --device cuda --dtype float32 \
  --C2 1.0 --charm-mass 1.4 \
  --proton-area 16.0 --rc-scheme SMALLEST --seed 7
```

Run `nlo-bk --help` or `nlo-dis --help` for all physics and integration options.

## Optimized-path scope

- Supported CUDA float32 BK calculations automatically use the fastest validated path: unrefined
  fixed K1, the fused K2/Kf custom producer, and one accepted three-derivative RK2(3) step per
  output interval. The generic GPU Vegas implementation is the fallback when the custom producer
  does not support the requested configuration. Other devices and dtypes use the portable path.
- K1 evaluates the two panels beside its singular corner in float64, while K2/Kf evaluates and
  accumulates its disjoint near-cancellation region in float64. Consecutive K2/Kf calculations
  reuse their learned Vegas grids; reused grids receive 25% of the ordinary warmup (5,000 points
  at the default 100,000-point production setting) before the unchanged production rounds.
- Python callers can set `CUDA_FUSION=False`, `K1_FIXED=False`, or `K1_FIXED_REFINE=True` when they
  explicitly need a fallback or comparison implementation; production use needs no backend flag.
- F2, FL, and FT automatically use the fastest validated DIS implementation for every sector.
  CUDA float32 evaluates the complete LO integrand and each NLO dipole expression with custom
  kernels for both `GBW` and `BKDipole`; cached tensor Gauss-Legendre grids handle the two-dimensional
  terms and fused GPU Vegas handles the higher-dimensional dipole terms.
- For the NLO qg sector, CUDA float32 with an analytic `GBW` dipole uses the fixed-Sobol I1+I2+I3
  path. Its common scrambled samples preserve the covariance of the summed qg result. A `BKDipole`,
  a nonstandard nested-point count, or a fixed-Sobol result that misses `epsrel` falls back to
  nested I2/I3 custom kernels with GPU Vegas. CPU and non-float32 calculations use the generic
  implementations for all sectors.
- `DISConfig` enables CUDA fusion and nested integration by default. Python callers may set
  either option to `False` for targeted comparisons, or both to `False` for a generic-Vegas run.
  The fixed-Sobol GBW workload uses 851,968 integrand evaluations per quark, so a smaller `maxeval`
  also selects the Vegas fallback.
- `benchmarks/dis_fixed_sobol_all_qg.py` exercises the same fixed-Sobol implementation used by the
  production cross section; it is a validation and timing entry point, not a separate fast path.
- The approximately 200x measurements are component/workload results: 186--189x for the complete
  six-term fixed-Sobol DIS evaluation and 206.7x for the BK `K2+Kf` radial grid, both against the
  matched 20-worker C++ runs. The fixed K1 production scope is 72.1x faster than adaptive CUDA;
  complete production BK evolution has not yet been benchmarked against C++ end to end.

Run the fixed-Sobol GBW validation workload with:

```bash
python -m benchmarks.dis_fixed_sobol_all_qg
```

## Design and file structure

The physics packages own equations, cross-section composition, and production dispatch. Custom
kernels are implementation details grouped by the mathematical expression they evaluate, never by
the implementation language. Shared build, quadrature, integration, interpolation, and data-mapping
code sits outside expression folders. A combined folder is used only for expressions implemented by
one compiled source because they share substantial mathematical machinery or one launch evaluates
both.

```text
src/nlo_torch/
├── bk/
│   ├── config.py             BK physics and numerical choices
│   ├── derivatives.py        K1, K2, and Kf derivative composition
│   └── evolution.py          production-path dispatch and rapidity stepping
├── dis/
│   ├── config.py             DIS physics and numerical choices
│   ├── longitudinal.py       longitudinal mathematical expressions
│   ├── transverse.py         transverse mathematical expressions
│   ├── observables.py        LO/NLO cross sections, F2/FL/FT, and fallback dispatch
│   └── fixed_sobol.py        production GBW qg sampling, adaptation, and error estimate
└── custom_kernels/
    ├── extension.py          shared extension build and cache management
    ├── quadrature.py         shared cached tensor Gauss-Legendre grids
    ├── integration.py/.cu    shared Vegas data-management kernels
    ├── integration_triton.py shared Vegas tensor kernels
    ├── interpolation.py/.cu  shared dipole-table interpolation
    ├── bk/
    │   ├── k1/               K1 kernel and Python boundary
    │   └── k2_kf/            fused K2/Kf kernel and Python/Triton boundaries
    └── dis/
        ├── i1/               I1 kernel and Python boundary
        ├── i2/               compiled I2 expression
        ├── i3/               compiled I3 expression
        ├── i2_i3/            fused nested I2/I3 kernel and Python boundary
        ├── lo_dipole/        LO and NLO dipole kernels sharing amplitude/Bessel machinery
        └── sampling/         DIS-only Sobol coordinate mapping
```

This separation keeps the readable formulas in `bk/` and `dis/`, keeps shared data management at
the custom-kernel root, and makes each optimized physics implementation discoverable from its
corresponding expression. There is deliberately no folder named after CUDA.

## Check the installation

```bash
python tests/smoke_test.py
```

This checks imports, analytic and table dipoles, legacy table I/O, and a small LO DIS calculation
without requiring CUDA.
