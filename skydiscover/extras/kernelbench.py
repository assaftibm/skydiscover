"""KernelBench dataset integration for SkyDiscover."""

import os
import tempfile
from typing import Tuple

from datasets import load_dataset


def load_kernelbench_kernel(level: int, problem_id: int) -> Tuple[str, str]:
    """
    Load a kernel from the KernelBench dataset.

    Args:
        level: Difficulty level (1-4)
        problem_id: Problem identifier

    Returns:
        Tuple of (kernel_code, kernel_name)

    Raises:
        ValueError: If kernel not found
        ValueError: If level is invalid
    """
    if not 1 <= level <= 4:
        raise ValueError(f"Level must be 1-4, got {level}")

    # Map level to dataset split
    split_name = f"level_{level}"

    try:
        dataset = load_dataset("ScalingIntelligence/KernelBench", split=split_name)
    except Exception as exc:
        raise ValueError(
            f"Failed to load KernelBench level {level}. "
            f"Ensure you have datasets library installed: pip install datasets"
        ) from exc

    # Find the problem by problem_id
    for sample in dataset:
        if sample["problem_id"] == problem_id:
            return sample["code"], sample["name"]

    raise ValueError(
        f"Problem ID {problem_id} not found in KernelBench level {level}. "
        f"Available problems: {sorted(set(s['problem_id'] for s in dataset))}"
    )


def save_kernelbench_kernel_to_file(level: int, problem_id: int, output_path: str) -> str:
    """
    Load a KernelBench kernel and save it to a file.

    Args:
        level: Difficulty level (1-4)
        problem_id: Problem identifier
        output_path: Path to save the kernel file

    Returns:
        Path to the saved kernel file
    """
    code, name = load_kernelbench_kernel(level, problem_id)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write(code)

    return output_path


def create_kernelbench_evaluator(level: int, problem_id: int) -> str:
    """
    Create an evaluator script for a KernelBench problem.

    This generates an evaluator that will be used by SkyDiscover to evaluate kernels.

    Args:
        level: Difficulty level (1-4)
        problem_id: Problem identifier

    Returns:
        Path to the evaluator script
    """
    # Load the kernel to get its name
    _, kernel_name = load_kernelbench_kernel(level, problem_id)

    evaluator_code = f'''"""KernelBench evaluator for {kernel_name}."""

import json
import sys
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple


def evaluate(generated_code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Evaluate a generated kernel solution.

    Args:
        generated_code: The generated kernel code as a string
        timeout: Timeout in seconds for execution

    Returns:
        Dictionary with evaluation results including:
        - correct: bool - whether output matches reference
        - speedup: float - speedup vs PyTorch baseline
        - error: str - error message if evaluation failed
    """
    try:
        # Execute the generated code to get the Model class
        namespace = {{"nn": nn, "torch": torch}}
        exec(generated_code, namespace)

        if "Model" not in namespace:
            return {{"correct": False, "error": "Generated code must define a Model class"}}

        Model = namespace["Model"]

        # Load reference implementation from KernelBench
        from datasets import load_dataset

        dataset = load_dataset("ScalingIntelligence/KernelBench", split="level_{level}")
        reference_code = None

        for sample in dataset:
            if sample["problem_id"] == {problem_id}:
                reference_code = sample["code"]
                break

        if not reference_code:
            return {{"correct": False, "error": "Could not load reference kernel"}}

        # Execute reference code
        ref_namespace = {{"nn": nn, "torch": torch}}
        exec(reference_code, ref_namespace)
        ReferenceModel = ref_namespace["Model"]

        # Get test inputs
        model = Model()
        ref_model = ReferenceModel()

        # Create test inputs
        if hasattr(model, "get_inputs"):
            inputs = model.get_inputs()
        else:
            # Fallback: try to infer from reference
            if hasattr(ref_model, "get_inputs"):
                inputs = ref_model.get_inputs()
            else:
                return {{"correct": False, "error": "Could not generate test inputs"}}

        # Run forward pass
        with torch.no_grad():
            output = model(*inputs)
            ref_output = ref_model(*inputs)

        # Check correctness with tolerance
        correct = torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)

        # Benchmark performance (simplified)
        # In a real scenario, you'd use proper timing utilities
        speedup = 1.0  # Placeholder

        return {{"correct": correct, "speedup": speedup}}

    except Exception as exc:
        import traceback
        return {{
            "correct": False,
            "error": str(exc),
            "traceback": traceback.format_exc()
        }}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            code = f.read()
    else:
        code = sys.stdin.read()

    result = evaluate(code)
    print(json.dumps(result))
    sys.exit(0 if result.get("correct") else 1)
'''

    # Create a temporary file for the evaluator
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="."
    ) as f:
        f.write(evaluator_code)
        evaluator_path = f.name

    return evaluator_path
