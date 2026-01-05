---
marp: true
theme: default
paginate: true
---

# PyTorch ONNX Exporter new features and architecture

Jan 2026

---

![infographics](pt-onnx-infographics.png)

---

## `dynamo=True` is the default

- The New Default: Starting from PyTorch 2.9, the **`dynamo=True`** option is the **default and recommended** way to export models to ONNX.
- Core Shift: It moves away from the older TorchScript-based capture mechanism to a torch.export based modern stack.
- Deprecation Plan: While the TorchScript exporter (dynamo=False) is currently usable, it is planned for eventual deprecation in alignment with PyTorch core's handling of TorchScript.

---

## New options in `export()`

```py
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

---

## What happens inside `torch.onnx.export`

torch.export() **captures FX** graph
-> **translate** and build ONNX IR
-> graph **optimization** with ONNX Script

Entry point is at: https://github.com/pytorch/pytorch/blob/0ad306cac740eaf2ce582e2bdf097cc61d929a40/torch/onnx/_internal/exporter/_core.py#L1282

![diagram](https://raw.githubusercontent.com/justinchuby/diagrams/refs/heads/main/pytorch/torch-export-flow.svg)

---

## FX graph and the ExportedProgram

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

---

## Decomposed ExportedProgram

```python
decomposed = exported_program.run_decompositions()
print(decomposed)
```

---

## Translation to ONNX

```python
onnx_program = torch.onnx.export(exported_program, report=True, opset_version=23)
print(onnx_program)
```

```python
onnx_program.save("model.onnx")
```

```bash
onnxvis model.onnx
```

---

## Model in `onnx_program.model` is an onnx_ir.Model

- You can run any ONNX->ONNX transformation on it.
- The exporter by default runs ONNX Script pattern replacement and whole graph optimization. These are robust, in-memory graph passes the team has created
- Low memory consumption by sharing tensor data with the PyTorch model

---

## Explore the IR model

```python
# Explore the IR model

model = onnx_program.model
print("Model has", len(model.graph), "nodes")

print("All initializers:")
for init in model.graph.initializers.values():
    print(" ", init)
```

---

## Shared Memory with PyTorch

```python
print(model.graph.initializers["weight"].const_value.raw is mod.weight)
```

```python
model.graph.initializers["weight"].const_value.display()
```

```python
print("All users of the initializer:", model.graph.initializers["weight"].uses())
```

---

## Verify model outputs

https://github.com/justinchuby/model-explorer-onnx

```python
from torch.onnx.verification import verify_onnx_program
from model_explorer_onnx.torch_utils import save_node_data_from_verification_info

verification_infos = verify_onnx_program(onnx_program, compare_intermediates=True)

# Produce node data for Model Explorer for visualization
save_node_data_from_verification_info(
    verification_infos, onnx_program.model, model_name="model"
)
```

```bash
onnxvis model.onnx --node_data_paths=model_max_abs_diff.json,model_max_rel_diff.json
```

---

## Multiple ways to represent dynamic shapes

```python
# Method 1: Using strings (recommended)
simple_dynamic_shapes = {
    "x": {0: "batch", 2: "seq_len"},
    "y": {0: "batch", 2: "seq_len"}
}
onnx_prog_simple = torch.onnx.export(
    mod, args=example_args,
    dynamic_shapes=simple_dynamic_shapes,
    opset_version=23,
)
```

---

## Method 2: Using ShapesCollection

```python
from torch.export import Dim, ShapesCollection

shapes_collection = ShapesCollection()
shapes_collection[example_args[0]] = {0: Dim.DYNAMIC, 2: Dim.DYNAMIC}
shapes_collection[example_args[1]] = {0: Dim.DYNAMIC, 2: Dim.DYNAMIC}

onnx_prog_shapes = torch.onnx.export(
    mod, args=example_args,
    dynamic_shapes=shapes_collection,
    opset_version=23,
)
```

---

## Method 3: Using AdditionalInputs

```python
ai = torch.export.AdditionalInputs()
example_args_1 = (torch.randn(2, 3, 10, 10), torch.randn(2, 3, 10, 10))
example_args_2 = (torch.randn(4, 3, 2, 10), torch.randn(4, 3, 2, 10))
ai.add(example_args_1)
ai.add(example_args_2)

onnx_prog_shapes = torch.onnx.export(
    mod, args=example_args,
    dynamic_shapes=ai,
    opset_version=23,
)
```
