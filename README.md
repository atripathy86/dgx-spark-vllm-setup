# vLLM Setup for NVIDIA DGX Spark (Blackwell GB10)

**One-command installation** of vLLM for NVIDIA DGX Spark systems with GB10 GPUs (Blackwell architecture, sm_121).

This repository provides a dgx-spark tested, ready setup script that handles all the complexities of building vLLM on the DGX Spark platform, including:
- CUDA 13.0 support with Blackwell-specific optimizations
- Critical fixes for SM100/SM120 MOE kernel compilation
- Triton 3.5.0 from main branch (required for sm_121a support)
- PyTorch 2.9.0 with CUDA 13.0 bindings
- All necessary build fixes and workarounds

## Quick Start

**One-command installation** - installs to `./vllm-install` in your current directory:

```bash
curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash
```

Or specify a custom directory:

```bash
curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash -s -- --install-dir ~/my/custom/path
```

**Installation time:** ~20-30 minutes (mostly compilation)

### Alternative: Clone and Install

```bash
git clone https://github.com/eelbaz/dgx-spark-vllm-setup.git
cd dgx-spark-vllm-setup
./install.sh
```

### Installation Options

```bash
./install.sh [OPTIONS]

Options:
  --install-dir DIR    Installation directory (default: ./vllm-install)
  --vllm-version TAG   vLLM git tag/branch (default: v0.11.1rc3)
  --python-version VER Python version (default: 3.12)
  --skip-tests         Skip post-installation tests
  --help               Show help message
```

## System Requirements

- **Hardware:** NVIDIA DGX Spark with GB10 GPU (Blackwell sm_121)
- **OS:** Ubuntu 22.04+ (tested on Linux 6.11.0 ARM64)
- **CUDA:** 13.0 or later (driver 580.95.05+)
- **Disk Space:** ~50GB free
- **RAM:** 8GB+ recommended during build
- **Build Dependencies:** Python 3.12 development headers (script will auto-detect and offer to install)

## What Gets Installed

Installed to `./vllm-install` (or your custom directory):

- **Python 3.12** virtual environment at `.vllm/`
- **PyTorch 2.9.0+cu130** with full CUDA 13.0 support
- **Triton 3.5.0+git** from main branch (pre-release with Blackwell support)
- **vLLM 0.11.1rc3+** with all Blackwell-specific patches
- **Helper scripts** for managing vLLM server
- **Environment activation** script (`vllm_env.sh`)

## Usage

All examples assume you're in the installation directory (default: `./vllm-install`).

### Activate Environment

```bash
cd vllm-install
source vllm_env.sh
```

### Start vLLM Server

```bash
./vllm-serve.sh                                    # Default: Qwen2.5-0.5B on port 8000
./vllm-serve.sh "facebook/opt-125m" 8001          # Custom model and port
```

### Check Server Status

```bash
./vllm-status.sh
```

### Stop Server

```bash
./vllm-stop.sh
```

### Test API

```bash
# List models
curl http://localhost:8000/v1/models

# Generate completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True,
    gpu_memory_utilization=0.9
)

prompts = ["Tell me about DGX Spark"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

print(outputs[0].outputs[0].text)
```

## Critical Fixes Applied

This installer automatically applies the following critical fixes:

### 1. CMakeLists.txt SM100/SM120 MOE Kernel Fix

**Issue:** vLLM's MOE kernels for SM100/SM120 Blackwell architectures were incomplete
**Fix:** Added `12.0f` and `12.1a` to SCALED_MM_ARCHS in CMakeLists.txt

```cmake
# CUDA 13.0+ path (line ~671)
# Before
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
# After
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")

# Older CUDA path (line ~673)
# Before
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a" "${CUDA_ARCHS}")
# After
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;12.1a" "${CUDA_ARCHS}")
```

### 2. pyproject.toml License Field Format

**Issue:** Newer setuptools requires structured license format
**Fix:** Convert license string to dict format in both vLLM and flashinfer-python

```toml
# Before
license = "Apache-2.0"
license-files = ["LICENSE"]

# After
license = {text = "Apache-2.0"}
```

**Applied to:**
- vLLM's pyproject.toml
- flashinfer-python's pyproject.toml (patched during build)

### 3. GPT-OSS Triton MOE Kernels for Qwen3/gpt-oss Support

**Issue:** vLLM's GPT-OSS MOE kernel implementation uses deprecated Triton routing API
**Fix:** Update to new Triton kernel API (topk and SparseMatrix)

**Changes:**
- Replace deprecated `routing()` with `triton_topk()`
- Replace deprecated `routing_from_bitmatrix()` with `SparseMatrix()`
- Add support for `GatherIndx`, `ScatterIndx`, and new ragged tensor metadata

**Enables support for:**
- Qwen3 models with MOE architecture
- gpt-oss models using Triton kernels
- Latest Triton kernel optimizations for Blackwell

### 4. Triton Main Branch Requirement

**Issue:** Official Triton 3.5.0 release has bugs with sm_121a
**Fix:** Build Triton from main branch with latest Blackwell fixes

## Architecture-Specific Configuration

The installer sets these critical environment variables:

```bash
TORCH_CUDA_ARCH_LIST=12.1a                      # Blackwell sm_121
VLLM_USE_FLASHINFER_MXFP4_MOE=1                 # Enable FlashInfer MOE optimization
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas     # CUDA PTX assembler
TIKTOKEN_CACHE_DIR=$INSTALL_DIR/.tiktoken_cache # Cache tiktoken encodings locally
```

## Cluster Mode Setup

To set up multi-node vLLM cluster:

1. Run this installer on all nodes
2. Follow [CLUSTER.md](./CLUSTER.md) for configuration

## Troubleshooting

### Python Development Headers Missing

**Symptom:** Build fails with CMake error: `file STRINGS file "/usr/include/python3.12/patchlevel.h" cannot be read`
**Cause:** Python 3.12 development headers not installed
**Solution (Automatic):** The installer now detects this and offers to install it automatically
**Solution (Manual):**
```bash
sudo apt-get update
sudo apt-get install -y python3.12-dev
```

### Build Fails with "TypeError: can only concatenate str (not 'NoneType') to str"

This is a known Triton editable-mode build issue. The installer works around this by:
- Building Triton in non-editable mode
- Or copying pre-built Triton from another node

### Symbol Error: cutlass_moe_mm_sm100

**Symptom:** `ImportError: undefined symbol: _Z20cutlass_moe_mm_sm100`
**Solution:** Ensure CMakeLists.txt fix is applied (done automatically by installer)

### PyTorch CUDA Capability Warning

**Symptom:** Warning about GPU capability 12.1 vs PyTorch max 12.0
**Status:** Harmless warning - PyTorch 2.9.0+cu130 works correctly with GB10

### ImportError: No module named 'vllm'

**Solution:**
```bash
source vllm-install/vllm_env.sh
python -c "import vllm; print(vllm.__version__)"
```

## File Structure

```
vllm-install/
├── .vllm/                  # Python virtual environment
├── vllm/                   # vLLM source (editable install)
├── triton/                 # Triton source
├── vllm_env.sh            # Environment activation script
├── vllm-serve.sh          # Start server
├── vllm-stop.sh           # Stop server
├── vllm-status.sh         # Check status
└── vllm-server.log        # Server logs
```

## Manual Installation

If you prefer to understand each step:

```bash
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Create installation directory and Python virtual environment
mkdir -p vllm-install && cd vllm-install
uv venv .vllm --python 3.12
source .vllm/bin/activate

# 3. Install PyTorch with CUDA 13.0
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 4. Clone and build Triton from main
git clone https://github.com/triton-lang/triton.git
cd triton
uv pip install pip cmake ninja pybind11
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas python -m pip install --no-build-isolation .

# 5. Install additional dependencies
uv pip install xgrammar setuptools-scm apache-tvm-ffi==0.1.0b15 --prerelease=allow

# 6. Clone vLLM
cd ..
git clone --recursive https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.11.1rc3

# 7. Apply fixes (see scripts/apply-fixes.sh)
# 8. Build vLLM (see install.sh for full process)
```

## Version Information

- **vLLM:** 0.11.1rc4.dev6+g66a168a19.d20251026
- **PyTorch:** 2.9.0+cu130
- **Triton:** 3.5.0+git4caa0328
- **CUDA:** 13.0
- **Python:** 3.12.3
- **Target Architecture:** sm_121 (Blackwell GB10)

## Contributing

Issues and pull requests welcome! This installer is maintained by the DGX Spark community.

## References

- [NVIDIA Forum Discussion](https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Triton GitHub](https://github.com/triton-lang/triton)

## License

MIT License - See [LICENSE](./LICENSE)

## Acknowledgments

Developed and tested on NVIDIA DGX Spark systems. Special thanks to the vLLM and Triton communities.
