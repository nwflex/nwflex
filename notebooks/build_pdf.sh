#!/bin/bash
#
# Build PDF supplement from NW-flex notebooks
#
# Usage: ./build_pdf.sh [--skip-execute] [--vector]
#
# Options:
#   --skip-execute   Skip notebook execution (use existing outputs)
#   --vector         Use vector graphics (PDF format) instead of PNG
#

set -e  # Exit on error

# Check for required dependencies
check_dependencies() {
    local missing=()
    
    if ! command -v jupyter &> /dev/null; then
        missing+=("jupyter")
    fi
    
    if ! python -c "import nbmerge" &> /dev/null 2>&1; then
        missing+=("nbmerge")
    fi
    
    if ! python -c "import nbformat" &> /dev/null 2>&1; then
        missing+=("nbformat")
    fi
    
    if ! command -v pdflatex &> /dev/null; then
        echo "Warning: pdflatex not found. Install texlive for PDF generation."
        echo "  On Ubuntu/Debian: sudo apt-get install texlive-full"
        echo "  On macOS: brew install --cask mactex"
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo "Error: Missing required Python packages: ${missing[*]}"
        echo "Install them with: pip install -e '.[notebooks]'"
        exit 1
    fi
}

check_dependencies

# Configuration
NOTEBOOKS=(
    "01_Alignment_Basics.ipynb"
    "02_NWflex_Core.ipynb"
    "03_NWflex_Validation.ipynb"
    "04_NWflex_STR.ipynb"
    "05_NWflex_Cython.ipynb"
    "06_NWflex_STR_locus.ipynb"
)
OUTPUT_DIR="pdfs"
MERGED_NOTEBOOK="NW-flex_Notebook_Supplement.ipynb"
OUTPUT_PDF="NW-flex_Supplementary_Notebooks"
TITLE="Supplementary Notebooks for NW-flex"

# Parse arguments
SKIP_EXECUTE=false
USE_VECTOR=false

for arg in "$@"; do
    case $arg in
        --skip-execute)
            SKIP_EXECUTE=true
            ;;
        --vector)
            USE_VECTOR=true
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Change to notebooks directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Step 1: Merge notebooks (clearing outputs first)
echo ""
echo "==========================================================================="
echo "Step 1: Merging notebooks (without outputs)..."
echo "==========================================================================="

# Clear outputs from each notebook and merge
for nb in "${NOTEBOOKS[@]}"; do
    echo "  Clearing outputs: $nb"
    jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True \
        --output="$OUTPUT_DIR/temp_${nb}" "$nb"
done

# Merge the cleared notebooks
TEMP_NOTEBOOKS=()
for nb in "${NOTEBOOKS[@]}"; do
    TEMP_NOTEBOOKS+=("$OUTPUT_DIR/temp_${nb}")
done

if command -v nbmerge &> /dev/null; then
    nbmerge "${TEMP_NOTEBOOKS[@]}" -o "$OUTPUT_DIR/$MERGED_NOTEBOOK"
else
    python -c "
from nbmerge import merge_notebooks
import nbformat
import sys

nbs = []
for f in sys.argv[1:-1]:
    nbs.append(nbformat.read(f, as_version=4))

merged = merge_notebooks('.', nbs)
nbformat.write(merged, sys.argv[-1])
" "${TEMP_NOTEBOOKS[@]}" "$OUTPUT_DIR/$MERGED_NOTEBOOK"
fi

# Clean up temp files
rm -f "${TEMP_NOTEBOOKS[@]}"
echo "  Created: $OUTPUT_DIR/$MERGED_NOTEBOOK"

# Step 2: Inject configuration cell at the front (for vector graphics if requested)
echo ""
echo "==========================================================================="
echo "Step 2: Injecting configuration cell..."
echo "==========================================================================="

if [ "$USE_VECTOR" = true ]; then
    CONFIG_CODE="# PDF build configuration - vector graphics\n%config InlineBackend.figure_formats = ['pdf', 'png']\nimport matplotlib\nmatplotlib.rcParams['figure.dpi'] = 150"
else
    CONFIG_CODE="# PDF build configuration\nimport matplotlib\nmatplotlib.rcParams['figure.dpi'] = 150"
fi

python -c "
import nbformat
import sys

nb = nbformat.read('$OUTPUT_DIR/$MERGED_NOTEBOOK', as_version=4)

# Create config cell
config_cell = nbformat.v4.new_code_cell('''$CONFIG_CODE''')
config_cell['metadata'] = {'tags': ['injected-config']}

# Insert at position 1 (after the raw LaTeX cell)
nb.cells.insert(1, config_cell)

nbformat.write(nb, '$OUTPUT_DIR/$MERGED_NOTEBOOK')
print('  Injected config cell')
"

# Step 3: Execute merged notebook (optional)
if [ "$SKIP_EXECUTE" = false ]; then
    echo ""
    echo "==========================================================================="
    echo "Step 3: Executing merged notebook..."
    echo "==========================================================================="
    
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 \
        "$OUTPUT_DIR/$MERGED_NOTEBOOK"
    echo "  Execution complete"
else
    echo ""
    echo "==========================================================================="
    echo "Step 3: Skipping execution (--skip-execute)"
    echo "==========================================================================="
fi

# Step 4: Convert to PDF
echo ""
echo "==========================================================================="
echo "Step 4: Converting to PDF..."
echo "==========================================================================="

if [ "$USE_VECTOR" = true ]; then
    echo "  Using LaTeX conversion for vector graphics..."
    
    # Convert to LaTeX first (preserves vector graphics better)
    jupyter nbconvert --to latex "$OUTPUT_DIR/$MERGED_NOTEBOOK" \
        --output-dir="$OUTPUT_DIR" \
        --output="$OUTPUT_PDF" \
        --no-prompt \
        --LatexPreprocessor.title="$TITLE"
    
    # Compile LaTeX to PDF (run twice for TOC/references)
    cd "$OUTPUT_DIR"
    pdflatex -interaction=nonstopmode "$OUTPUT_PDF.tex" > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode "$OUTPUT_PDF.tex" > /dev/null 2>&1 || true
    
    # Clean up LaTeX auxiliary files
    rm -f "$OUTPUT_PDF.aux" "$OUTPUT_PDF.log" "$OUTPUT_PDF.out" "$OUTPUT_PDF.toc" 2>/dev/null
    rm -f "*.tex" 2>/dev/null  # Remove .tex file too
    cd ..
else
    # Direct PDF conversion (standard approach)
    jupyter nbconvert --to pdf "$OUTPUT_DIR/$MERGED_NOTEBOOK" \
        --output-dir="$OUTPUT_DIR" \
        --output="$OUTPUT_PDF" \
        --TemplateExporter.extra_template_basedirs="$(pwd)" \
        --template=custom_template \
        --no-prompt \
        --LatexPreprocessor.title="$TITLE"
fi

echo ""
echo "==========================================================================="
echo "Done!"
echo "==========================================================================="
echo "  Output: $OUTPUT_DIR/$OUTPUT_PDF.pdf"
echo ""
