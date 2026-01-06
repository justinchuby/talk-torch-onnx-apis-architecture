# PyTorch ONNX Exporter: New Features and Architecture Tutorial

*January 2026*

![infographics](<src/pt-onnx-infographics.png>)

## Introduction

This tutorial covers the latest features and architecture changes in PyTorch's ONNX exporter. Starting with PyTorch 2.9, there are significant improvements in how models are exported to ONNX format, with a focus on the new Dynamo-based export mechanism.

## `dynamo=True` is Now the Default

The PyTorch ONNX exporter has undergone a major transformation:

- **The New Default**: Starting from PyTorch 2.9, the **`dynamo=True`** option is the **default and recommended** way to export models to ONNX.
- **Core Shift**: It moves away from the older TorchScript-based capture mechanism to a torch.export based modern stack.
- **Deprecation Plan**: While the TorchScript exporter (dynamo=False) is currently usable, it is planned for eventual deprecation in alignment with PyTorch core's handling of TorchScript.

## New Options in `export()`

The `torch.onnx.export()` function now includes several powerful new options:

```python
torch.onnx.export(
    model, args, kwargs=kwargs,
    # New way of expressing dynamic shapes (more examples later)
    dynamic_shapes=({0: "batch", 1: "sequence_len"}),
    # dynamic_axes=...,  # Deprecated
    dynamo=True,  # Default (2.9)
    report=True,  # Creates a markdown report
    verify=True,  # Runs onnx runtime on the example
    optimize=True, # Runs onnxscript graph optimizations
) -> torch.onnx.ONNXProgram
```

## Understanding the Export Process

The export process follows this flow:

1. **torch.export()** captures FX graph
2. **Translate** and build ONNX IR
3. **Graph optimization** with ONNX Script

Entry point is at: https://github.com/pytorch/pytorch/blob/0ad306cac740eaf2ce582e2bdf097cc61d929a40/torch/onnx/_internal/exporter/_core.py#L1282

![diagram](https://raw.githubusercontent.com/justinchuby/diagrams/refs/heads/main/pytorch/torch-export-flow.svg)

## Working with FX Graph and ExportedProgram

Let's start with a practical example. First, we'll create a simple PyTorch model:

```python
import torch
import torch.export

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 3, 1, 10))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        a.add_(y)
        b = a * self.weight
        return torch.nn.functional.scaled_dot_product_attention(b, b, b)

example_args = (torch.randn(2, 3, 10, 10), torch.randn(2, 3, 10, 10))

# Important to set to eval mode before exporting
mod = Mod().eval()
exported_program: "ExportedProgram" = torch.export.export(mod, args=example_args)
print(exported_program)
```

You can also run decompositions on the exported program:

```python
decomposed = exported_program.run_decompositions()
print(decomposed)
```

## Translation to ONNX

Converting the exported program to ONNX is straightforward:

```python
onnx_program = torch.onnx.export(exported_program, report=True, opset_version=23)
print(onnx_program)
```

Save the ONNX model to disk:

```python
onnx_program.save("model.onnx")
```

Visualize the model using onnxvis:

```bash
onnxvis model.onnx
```

## Working with ONNX IR Model

The model in `onnx_program.model` is an `onnx_ir.Model` object with several advantages:

- You can run any ONNX→ONNX transformation on it
- The exporter by default runs ONNX Script pattern replacement and whole graph optimization. These are robust, in-memory graph passes the team has created
- Low memory consumption by sharing tensor data with the PyTorch model

### Exploring the IR Model

```python
# Explore the IR model

model = onnx_program.model
print("Model has", len(model.graph), "nodes")

print("All initializers:")
for init in model.graph.initializers.values():
    print(" ", init)
```

Check if initializer shares memory with the original PyTorch tensor:

```python
print(model.graph.initializers["weight"].const_value.raw is mod.weight)
```

Display initializer value:

```python
model.graph.initializers["weight"].const_value.display()
```

Find all users of an initializer:

```python
print("All users of the initializer:", model.graph.initializers["weight"].uses())
```

## Verifying Model Outputs

Verification is crucial to ensure correctness. You can verify outputs and compare intermediate values:

```python
from torch.onnx.verification import verify_onnx_program
from model_explorer_onnx.torch_utils import save_node_data_from_verification_info

verification_infos = verify_onnx_program(onnx_program, compare_intermediates=True)

# Produce node data for Model Explorer for visualization
save_node_data_from_verification_info(
    verification_infos, onnx_program.model, model_name="model"
)
```

Visualize with verification data:

```bash
onnxvis model.onnx --node_data_paths=model_max_abs_diff.json,model_max_rel_diff.json
```

More information: https://github.com/justinchuby/model-explorer-onnx

## Multiple Ways to Represent Dynamic Shapes

PyTorch offers several flexible approaches to specify dynamic shapes:

### Method 1: Using Strings (Recommended)

```python
from torch.export import Dim, ShapesCollection

# Using strings
simple_dynamic_shapes = (
    {0: "batch", 2: "seq_len"},
    {0: "batch", 2: "seq_len"}
)

# Or by specifying name keys:
# simple_dynamic_shapes = {
#     "x": {0: "batch", 2: "seq_len"},
#     "y": {0: "batch", 2: "seq_len"}
# }

onnx_prog_simple = torch.onnx.export(
    mod,
    args=example_args,
    dynamic_shapes=simple_dynamic_shapes,
    opset_version=23,
)
print(onnx_prog_simple)
```

### Method 2: Using ShapesCollection

```python
shapes_collection = ShapesCollection()
shapes_collection[example_args[0]] = {0: Dim.DYNAMIC, 2: Dim.DYNAMIC}
shapes_collection[example_args[1]] = {0: Dim.DYNAMIC, 2: Dim.DYNAMIC}

onnx_prog_shapes = torch.onnx.export(
    mod,
    args=example_args,
    dynamic_shapes=shapes_collection,
    opset_version=23,
)
print(onnx_prog_shapes)
```

### Method 3: Using AdditionalInputs

```python
ai = torch.export.AdditionalInputs()
example_args_1 = (torch.randn(2, 3, 10, 10), torch.randn(2, 3, 10, 10))
example_args_2 = (torch.randn(4, 3, 2, 10), torch.randn(4, 3, 2, 10))
ai.add(example_args_1)
ai.add(example_args_2)

onnx_prog_shapes = torch.onnx.export(
    mod,
    args=example_args,
    dynamic_shapes=ai,
    opset_version=23,
)
print(onnx_prog_shapes)
```

## Conclusion

The new PyTorch ONNX exporter with `dynamo=True` provides a modern, robust way to export PyTorch models to ONNX format. Key benefits include:

- Improved graph capture using torch.export
- Better optimization passes
- Flexible dynamic shape specification
- Built-in verification capabilities
- Memory-efficient IR representation

For more information, visit the [PyTorch ONNX documentation](https://pytorch.org/docs/stable/onnx.html).
