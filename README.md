# KernelBench Analysis Tool

Automated analysis system for KernelBench Level 1 operations with support for FP32 tensor analysis, memory pattern detection, and visualization generation.

## Features

- 🔍 **Automatic Input Extraction**: Calls `get_inputs()` to analyze actual tensor data
- 📊 **FP32 Analysis**: Memory usage, tensor characteristics, and access patterns
- 📈 **Visualizations**: Generates 7 different charts and graphs
- 🏷️ **Categorization**: Organizes operations by type (GEMM, convolutions, activations, etc.)
- 📝 **Comprehensive Reports**: JSON and text summaries with detailed statistics
- 🚀 **CUDA Support**: Optimized for NVIDIA GPUs with CUDA 12.8

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA 12.x support (optional, but recommended)
- CUDA 12.8 toolkit installed (if using GPU)

### Quick Install (Windows)

```bash
# Run the installation batch file
install_requirements.bat
```

### Manual Install

```bash
# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy>=1.24.0 matplotlib>=3.7.0 psutil>=5.9.0
```

### Verify Installation

```bash
# Check if CUDA is properly configured
python verify_cuda.py
```

Expected output if CUDA is working:
```
PyTorch CUDA Verification
================================================================================
✓ PyTorch installed successfully
  Version: 2.1.0+cu121

CUDA Available: True
  CUDA Version: 12.1
  cuDNN Version: 8902
  Device Count: 1

  Device 0: NVIDIA GeForce RTX 4090
    Compute Capability: (8, 9)
    Total Memory: 24.00 GB

✓ Successfully created tensor on CUDA
  Tensor device: cuda:0
```

## Quick Start

```bash
# Run full analysis
python examples/run_analysis.py

# Analyze specific category
python examples/analyze_category.py

# Inspect single operation
python examples/quick_inspection.py

# Safe analysis with timeout protection
python examples/run_analysis_safe.py
```

## Usage Examples

### Full Analysis

```python
from kernelbench_analysis.main import KernelBenchAnalysisPipeline
from pathlib import Path

pipeline = KernelBenchAnalysisPipeline(
    dataset_path=Path("KernelBench-main/KernelBench/level1"),
    operation_types_path=Path("src/kernelbench_analysis/operation_types.json"),
    output_dir=Path("analysis_results"),
    timeout_seconds=10,  # Timeout per operation
    skip_actual_inputs=False  # Set True to skip get_inputs() calls
)

report = pipeline.run_full_analysis(create_visualizations=True)
```

### Command Line Options

```bash
# Run with custom timeout
python examples/run_analysis_safe.py --timeout 10

# Skip calling get_inputs() (faster but less accurate)
python examples/run_analysis_safe.py --skip-inputs

# Analyze only GEMM operations
python examples/run_analysis_safe.py --category gemm_matrix_operations

# Skip visualization generation
python examples/run_analysis_safe.py --no-viz
```

## Output

Analysis results are saved to `analysis_results/`:

- `analysis_report_*.json` - Complete analysis data
- `analysis_summary.txt` - Human-readable summary
- `visualizations/*.png` - Generated charts:
  - Category distribution
  - Memory usage by category
  - Tensor dimension distribution
  - Memory access patterns
  - Operation complexity
  - Memory usage histogram
  - Top memory operations

## Categories

Operations are categorized into:

- **GEMM Matrix Operations** (18 ops) - Matrix multiplication variants
- **Activation Functions** (15 ops) - ReLU, Sigmoid, GELU, etc.
- **Normalization Operations** (8 ops) - BatchNorm, LayerNorm, etc.
- **Pooling Operations** (6 ops) - MaxPool, AvgPool
- **Reduction Operations** (6 ops) - Sum, Mean, Max, etc.
- **Convolution Operations** (38 ops) - Standard, transposed, depthwise
- **Cumulative Operations** (5 ops) - Cumsum, cumprod
- **Loss Functions** (6 ops) - MSE, CrossEntropy, etc.
- **Attention Mechanisms** (1 op) - Scaled dot-product attention

## Memory Management

The tool includes aggressive memory management to prevent leaks when analyzing 100+ operations:

- Thread-based timeout for each operation (default: 10s)
- Automatic garbage collection every 20 operations
- CUDA cache clearing (if GPU is available)
- Progress monitoring with memory usage tracking

Status indicators during analysis:
- `✓` - Success with actual inputs extracted
- `○` - Success but only static analysis (no get_inputs())
- `✗` - Failed to analyze
- `⏱` - Operation timed out

## Troubleshooting

### CUDA Not Available

If `verify_cuda.py` shows CUDA is not available:

1. Check NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Memory Issues

If the system runs out of memory:

```bash
# Skip get_inputs() to use less memory
python examples/run_analysis_safe.py --skip-inputs

# Reduce timeout
python examples/run_analysis_safe.py --timeout 5

# Analyze one category at a time
python examples/run_analysis_safe.py --category activation_functions
```

### Operations Timing Out

Increase timeout for large operations:

```bash
python examples/run_analysis_safe.py --timeout 15
```

## System Requirements

- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA GPU with 24GB VRAM (e.g., RTX 4090)

## License

MIT License
