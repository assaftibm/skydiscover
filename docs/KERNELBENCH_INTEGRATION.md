# KernelBench Integration

SkyDiscover now supports kernel optimization using the KernelBench dataset from Hugging Face. This allows you to evolve and evaluate CUDA kernels across multiple difficulty levels.

## Overview

KernelBench contains 270 kernel optimization problems organized into 4 difficulty levels:

- **Level 1**: 100 single-kernel operators (convolutions, matrix multiplies, layer normalization)
- **Level 2**: 100 fusion patterns (e.g., Conv+Bias+ReLU combined operations)
- **Level 3**: 50 full ML architectures (MobileNet, VGG, MiniGPT, Mamba)
- **Level 4**: 20 HuggingFace model architectures

Dataset: [ScalingIntelligence/KernelBench on Hugging Face](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)

## Installation

Install the kernelbench optional dependencies:

```bash
uv sync --extra kernelbench
```

This installs:
- `datasets>=4.5.0` - For loading kernels from Hugging Face
- `torch>=2.0.0` - For PyTorch kernel definitions

## Usage

### Basic Usage

Use the `--kernelbench-level` and `--kernelbench-problem-id` arguments:

```bash
skydiscover-run \
  --kernelbench-level 1 \
  --kernelbench-problem-id 0 \
  --config config.yaml \
  -i 5
```

### Parameters

- `--kernelbench-level` (required): Difficulty level (1-4)
- `--kernelbench-problem-id` (required): Problem identifier within the level

Both parameters must be specified together. The CLI will:

1. Download the kernel from Hugging Face
2. Save it to a temporary file as `initial_program.py`
3. Create an evaluator script that checks:
   - **Correctness**: Output matches the reference implementation (tolerance: 1e-2)
   - **Performance**: Speedup ratio vs PyTorch baseline
4. Run SkyDiscover to evolve the kernel

### Examples

```bash
# Simple level 1 kernel (10 iterations)
skydiscover-run \
  --kernelbench-level 1 \
  --kernelbench-problem-id 5 \
  --config config.yaml \
  -i 10

# More complex level 2 fusion pattern (20 iterations)
skydiscover-run \
  --kernelbench-level 2 \
  --kernelbench-problem-id 25 \
  --config config.yaml \
  -i 20

# Level 3 architecture (with specific model)
skydiscover-run \
  --kernelbench-level 3 \
  --kernelbench-problem-id 10 \
  --config config.yaml \
  --model gpt-4 \
  -i 30
```

### Configuration

You still need to provide a SkyDiscover configuration file with LLM settings. Example `config.yaml`:

```yaml
llm:
  models:
    - name: gpt-4
      weight: 1.0
  api_base: https://api.openai.com/v1

max_iterations: 10

search:
  type: beam_search
```

## How It Works

### Kernel Loading

The CLI loads kernels directly from the Hugging Face dataset:

1. Fetches the `level_<N>` split from `ScalingIntelligence/KernelBench`
2. Finds the kernel matching the `problem_id`
3. Saves it to a temporary file

### Evaluation

The generated evaluator:

1. Executes the generated kernel code to extract the `Model` class
2. Loads the reference implementation from Hugging Face
3. Creates test inputs using `Model.get_inputs()`
4. Compares output correctness with tolerance (1e-2 absolute/relative)
5. Returns a result dictionary with `correct` and `speedup` metrics

### Kernel Format

Each kernel follows this PyTorch format:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """Description of the kernel operation."""

    def __init__(self, *args):
        super(Model, self).__init__()
        # Initialize any parameters

    def forward(self, *inputs):
        """Implement the kernel computation."""
        return result

def get_inputs():
    """Returns list of input tensors for testing."""
    return [tensor1, tensor2, ...]

def get_init_inputs():
    """Returns initialization parameters (optional)."""
    return [param1, param2, ...]
```

## Error Handling

### Problem ID Not Found

If you specify an invalid `problem_id`, the CLI will list available problem IDs:

```
Error: Problem ID 999 not found in KernelBench level 1.
Available problems: [0, 1, 2, 3, ..., 99]
```

### Invalid Level

```
Error: Level must be 1-4, got 5
```

### Missing KernelBench Dependencies

If `datasets` or `torch` are not installed:

```bash
uv sync --extra kernelbench
```

### Both Parameters Required

Both `--kernelbench-level` and `--kernelbench-problem-id` must be specified:

```
Error: Both --kernelbench-level and --kernelbench-problem-id must be specified
```

## Metrics

The evaluator returns:

- **`correct`** (bool): Whether the generated kernel's output matches the reference
- **`speedup`** (float): Relative speedup vs PyTorch baseline (1.0 = same speed, 2.0 = 2x faster)
- **`error`** (str, if applicable): Error message if evaluation failed

## Performance Tips

1. **Start small**: Begin with level 1 kernels to verify setup, then progress to higher levels
2. **Increase iterations for complex kernels**: Level 3-4 problems may need more iterations
3. **Use appropriate models**: For kernel optimization, specialized code generation models perform better
4. **Monitor evaluation time**: More complex kernels take longer to evaluate

## Example Workflow

```bash
# 1. Try a simple level 1 kernel
skydiscover-run \
  --kernelbench-level 1 \
  --kernelbench-problem-id 0 \
  --config config.yaml \
  -i 5

# 2. If successful, try a level 2 fusion pattern
skydiscover-run \
  --kernelbench-level 2 \
  --kernelbench-problem-id 10 \
  --config config.yaml \
  -i 10

# 3. Move to level 3 architectures
skydiscover-run \
  --kernelbench-level 3 \
  --kernelbench-problem-id 5 \
  --config config.yaml \
  -i 20
```

## Troubleshooting

### Network Issues

If you can't download from Hugging Face, ensure:

1. Internet connection is available
2. You can access `huggingface.co`
3. Your Hugging Face token is set (if required): `huggingface-cli login`

### Memory Issues

If you encounter OOM errors:

1. Close other applications
2. Use smaller kernels (lower problem IDs in level 1)
3. Reduce the number of iterations

### Slow Evaluation

Kernel evaluation can be slow, especially for level 3-4. Expected times:

- **Level 1**: 0.5-2 seconds per iteration
- **Level 2**: 1-5 seconds per iteration
- **Level 3**: 5-30 seconds per iteration
- **Level 4**: 30+ seconds per iteration

## API Reference

See `skydiscover.extras.kernelbench` module:

```python
from skydiscover.extras.kernelbench import (
    load_kernelbench_kernel,
    save_kernelbench_kernel_to_file,
    create_kernelbench_evaluator,
)

# Load a kernel directly
code, name = load_kernelbench_kernel(level=1, problem_id=5)

# Save to a file
path = save_kernelbench_kernel_to_file(
    level=1,
    problem_id=5,
    output_path="my_kernel.py"
)

# Create an evaluator
evaluator_path = create_kernelbench_evaluator(
    level=1,
    problem_id=5
)
```

## References

- [KernelBench Paper](https://arxiv.org/abs/2502.10517)
- [KernelBench GitHub](https://github.com/ScalingIntelligence/KernelBench)
- [KernelBench Leaderboard](https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/)
