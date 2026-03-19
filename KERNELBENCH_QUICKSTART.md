# KernelBench Quick Start

Evolve CUDA kernels using SkyDiscover and KernelBench in 3 steps!

## Installation

```bash
uv sync --extra kernelbench
```

## Basic Usage

```bash
skydiscover-run \
  --kernelbench-level 1 \
  --kernelbench-problem-id 5 \
  --config config.yaml \
  -i 10
```

That's it! SkyDiscover will:
1. ✓ Download the kernel from HuggingFace
2. ✓ Create an evaluator
3. ✓ Evolve the kernel for 10 iterations
4. ✓ Report the best score

## What Happens

- **Level 1**: Single operators (100 kernels) - Quick to evaluate, great for testing
- **Level 2**: Fusion patterns (100 kernels) - Intermediate difficulty
- **Level 3**: Full architectures (50 kernels) - Complex kernels, longer evaluation
- **Level 4**: HuggingFace models (20 kernels) - Advanced optimization

## Configuration File

You still need a config file with LLM settings:

```yaml
# config.yaml
llm:
  models:
    - name: gpt-4
      weight: 1.0
  api_base: https://api.openai.com/v1

max_iterations: 10

search:
  type: beam_search
```

## Examples

```bash
# Quick test with Level 1 (simple kernel)
skydiscover-run --kernelbench-level 1 --kernelbench-problem-id 1 \
  --config config.yaml -i 5

# Level 2 with more iterations
skydiscover-run --kernelbench-level 2 --kernelbench-problem-id 10 \
  --config config.yaml -i 20

# Level 3 with specific model
skydiscover-run --kernelbench-level 3 --kernelbench-problem-id 5 \
  --config config.yaml --model gpt-4 -i 30
```

## Finding Problem IDs

KernelBench problem IDs start at 1, not 0:

- **Level 1**: IDs 1-100 (100 problems)
- **Level 2**: IDs 1-100 (100 problems)
- **Level 3**: IDs 1-50 (50 problems)
- **Level 4**: IDs 1-20 (20 problems)

If you use an invalid ID, you'll see available IDs in the error message.

## Success Criteria

The evaluator checks:
- **Correctness**: Output matches reference (tolerance: 1e-2)
- **Speedup**: Performance vs PyTorch baseline

Results show:
- `correct: true/false` - Did the kernel produce correct output?
- `speedup: 1.5` - How many times faster than PyTorch?

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: datasets` | Run `uv sync --extra kernelbench` |
| `ModuleNotFoundError: torch` | Run `uv sync --extra kernelbench` |
| `Problem ID not found` | Use valid IDs (1-100 for levels 1-2, 1-50 for level 3, 1-20 for level 4) |
| `Config file not found` | Create a config.yaml file |
| `Error: Both --kernelbench-level and --kernelbench-problem-id must be specified` | Provide both arguments together |

## Next Steps

1. **Start small**: Try a Level 1 kernel first
2. **Check results**: Look at the best program metrics
3. **Scale up**: Progress to Level 2/3 if successful
4. **Fine-tune**: Adjust iterations and LLM models based on results

## Advanced

For programmatic access:

```python
from skydiscover.extras.kernelbench import load_kernelbench_kernel

# Load a kernel directly
code, name = load_kernelbench_kernel(level=1, problem_id=5)
print(code)
```

See [KERNELBENCH_INTEGRATION.md](docs/KERNELBENCH_INTEGRATION.md) for complete documentation.
