---
marp: true
theme: default
paginate: true
---

# PyTorch ONNX Exporter new features and architecture

Jan 2026

Justin Chu

---

![infographics](pt-onnx-infographics.png)

---

## 1. The new APIs

- The New Default: Starting from PyTorch 2.9, the **`dynamo=True`** option is the **default and recommended** way to export models to ONNX.
- Core Shift: It moves away from the older TorchScript-based capture mechanism to a torch.export based modern stack.
- Deprecation Plan: While the TorchScript exporter (dynamo=False) is currently usable, it is planned for eventual deprecation in alignment with PyTorch core's handling of TorchScript.

---
