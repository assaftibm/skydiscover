#!/bin/bash
# Demo: Running SkyDiscover with KernelBench kernels

# Example 1: Level 1, Problem 0 (a simple kernel)
# This loads a kernel from KernelBench and creates an evaluator
# Then runs SkyDiscover to evolve the kernel

echo "KernelBench Integration Examples"
echo "=================================="
echo ""
echo "To run SkyDiscover with a KernelBench kernel:"
echo ""
echo "# Level 1, Problem 0"
echo "skydiscover-run \\
  --kernelbench-level 1 \\
  --kernelbench-problem-id 0 \\
  --config config.yaml \\
  -i 5"
echo ""
echo "# Level 2, Problem 10"
echo "skydiscover-run \\
  --kernelbench-level 2 \\
  --kernelbench-problem-id 10 \\
  --config config.yaml \\
  -i 10"
echo ""
echo "# Level 3, Problem 5"
echo "skydiscover-run \\
  --kernelbench-level 3 \\
  --kernelbench-problem-id 5 \\
  --config config.yaml"
echo ""
echo "Notes:"
echo "------"
echo "1. The --kernelbench-level and --kernelbench-problem-id arguments"
echo "   automatically load a kernel from HuggingFace (ScalingIntelligence/KernelBench)"
echo "2. A temporary evaluator is created that checks correctness and speedup"
echo "3. You still need to provide a config file with LLM settings"
echo "4. Available levels: 1, 2, 3, 4"
echo "5. Problem IDs vary by level (run with invalid ID to see available ones)"
