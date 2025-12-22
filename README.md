![NW-flex logo](nwflex_logo.svg)
# NW-flex


**A generalized Needleman-Wunsch alignment algorithm with Extra Predecessors (EP) for flexible repeat-region alignment.**

NW-flex extends classical sequence alignment to support flexible alignment of repetitive sequences, particularly short tandem repeats (STRs). By introducing configurable "extra predecessors" in the dynamic programming recurrence, NW-flex can skip over repeat units during alignment, enabling biologically meaningful alignments that standard algorithms cannot achieve.

> **Paper**: *[Placeholder for manuscript link once submitted]*

## Key Features

- **Extra Predecessor (EP) Framework**: Extend the standard Gotoh DP recurrence with configurable row-skip transitions
- **STR-Aware Alignment**: Phase-aware alignment for Short Tandem Repeats with arbitrary motifs
- **Single-Block Flex**: Flexible contraction over a designated block region
- **Multi-STR Support**: Handle compound STR loci with multiple repeat blocks
- **Cython Acceleration**: Drop-in fast implementation for production use
- **Educational Notebooks**: Step-by-step derivation from Needleman-Wunsch basics to NW-flex

## Installation

```bash
# Create and activate conda environment
conda create --name nwflex python=3.11
conda activate nwflex

# Clone and install
git clone https://github.com/nwflex/nwflex.git
cd nwflex
pip install -e .
```

This compiles the Cython extension automatically.

## Quick Start

```python
from nwflex.aligners import align_standard, align_single_block, align_STR_block
from nwflex.validation import get_default_scoring
from nwflex.repeats import STRLocus
from nwflex.plot import plot_flex_matrices
import matplotlib.pyplot as plt

# Get default scoring parameters
score_matrix, gap_open, gap_extend, a2i = get_default_scoring()

# X = A · Z  · B where Z  spans positions [s, e)
# Y = A · Z* · B where Z* is a substring of Z
X = "ACGT" + "ATAT" + "GCTAC"
Y = "ACGT" + "TA"   + "GCTAC"

# Standard Needleman-Wunsch/Gotoh alignment (EP = ∅)
result = align_standard(X, Y, score_matrix, gap_open, gap_extend, a2i)
print(f"Score: {result.score}")
print(f"X: {result.X_aln}")
print(f"Y: {result.Y_aln}")

# Single-block flex alignment
# end of A, start of B.
s, e = 4, 8  
result = align_single_block(X, Y, s, e, score_matrix, gap_open, gap_extend, a2i)
print(f"Flex score: {result.score}, Jumps: {result.jumps}")

# For an STR, 
# X is A·(R)^N·B where R is the repeat unit
# Build Y with M repeat units
strLocus = STRLocus(A="ACGT", B="GCTAC", R="AT", N=10)
Y = strLocus.build_locus_variant(a=0, b=0, M=5)
# X = ACGT + (AT)10 + GCTAC
# Y = ACGT + (AT)5 + GCTAC
result = align_STR_block(
    strLocus = strLocus,
    Y = Y,
    score_matrix = score_matrix,
    gap_open = gap_open,
    gap_extend = gap_extend,
    alphabet_to_index = a2i,
    return_data = True
)
## plot the matrices
plot_flex_matrices(
    result=result, 
    X= strLocus.X,
    Y= Y, 
    s = strLocus.s, 
    e = strLocus.e,
    figsize=(20, 10)
)
plt.show()

```

## Documentation

### Supplementary Notebooks

The `notebooks/` directory contains a pedagogical series deriving NW-flex from first principles:

| Notebook | Description |
|----------|-------------|
| `01_Alignment_Basics.ipynb` | Introduction to pairwise alignment and DAG representation |
| `02_NWflex_Core.ipynb` | The Extra Predecessor framework and core algorithm |
| `03_NWflex_Validation.ipynb` | Correctness validation against baseline implementations |
| `04_NWflex_STR.ipynb` | Application to Short Tandem Repeat alignment |
| `05_NWflex_Cython.ipynb` | Cython acceleration and performance |
| `06_NWflex_STR_locus.ipynb` | Simulating reads from STR loci and phase-aware alignment |

A merged PDF can genereated with the commmand 
```bash
cd notebooks && ./build_pdf.sh
```

### Module Reference

| Module | Purpose |
|--------|---------|
| `aligners.py` | User-facing alignment functions (`align_standard`, `align_single_block`, `align_STR_phase`, etc.) |
| `dp_core.py` | Three-state Gotoh DP with row-wise extra predecessors. Core dataclasses: `FlexInput`, `FlexData`, `RowJump`, `AlignmentResult` |
| `ep_patterns.py` | EP configuration builders: `build_EP_standard`, `build_EP_single_block`, `build_EP_STR_phase` |
| `fast.py` | Cython-accelerated DP via `run_flex_dp_fast()` |
| `repeats.py` | STR utilities: `phase_repeat`, `STRLocus`, `CompoundSTRLocus` |
| `validation.py` | Baseline implementations for testing: `nwg_global`, `sflex_naive` |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nwflex --cov-report=term-missing
```

See `tests/README.md` for detailed test documentation.

## Algorithm Overview

NW-flex modifies the standard Gotoh three-state recurrence by allowing additional predecessor rows beyond the baseline `i-1`:

```
Standard:  M[i,j] depends on M[i-1, j-1], Xg[i-1, j-1], Yg[i-1, j-1]
NW-flex:   M[i,j] depends on {i-1} ∪ E(i) predecessor rows
```

Where `E(i)` is the **extra predecessor set** for row `i`. This enables:

- **Single-block flex**: Skip over a designated block region
- **STR phase alignment**: Skip repeat units while maintaining phase consistency
- **Multi-STR**: Handle compound loci with multiple repeat blocks

The traceback records which predecessor was used, allowing reconstruction of the alignment path including any "jumps" over repeat regions.

## Project Structure

```
nwflex/
├── nwflex/            # Package source
│   ├── aligners.py    # User-facing API
│   ├── dp_core.py     # Core DP algorithm
│   ├── ep_patterns.py # EP configuration builders
│   ├── fast.py        # Cython interface
│   ├── repeats.py     # STR utilities
│   ├── plot/          # Visualization subpackage
│   └── _cython/       # Cython source
├── notebooks/         # Educational notebooks
├── scripts/           # Figure generation scripts
└── tests/             # pytest test suite
```

## Citation

*[Citation information will be added upon publication]*

## License

MIT License. See [LICENSE](LICENSE) for details.
