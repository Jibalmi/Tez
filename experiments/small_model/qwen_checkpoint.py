"""Immutable local adapter/optimizer checkpoints for a future registered continuation."""
import json
from pathlib import Path

from structure_probe import digest


def parameter_names(named, optimizer):
    names = [name for name, _ in named]
    if not names or len(names) != len(set(names)) or any("lora_" not in name or not p.requires_grad for name, p in named):
        raise ValueError("Checkpoint only the uniquely named trainable LoRA tensors")
    expected = [id(p) for _, p in named]
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if actual != expected:
        raise ValueError("Optimizer parameter order must match the saved adapter order")
    return names


def save_checkpoint(path, named, optimizer, state, registration_sha256, *, cuda=False):
    import torch
    names = parameter_names(named, optimizer)
    # Keep progress metadata portable and reject NaN, tensors or application objects in it.
    state = json.loads(json.dumps(state, allow_nan=False))
    if len(registration_sha256) != 64:
        raise ValueError("Pin the complete experiment registration")
    tensors = {name: p.detach().cpu().clone() for name, p in named}
    if not all(torch.isfinite(value).all().item() for value in tensors.values()):
        raise ValueError("Do not checkpoint non-finite adapter tensors")
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise ValueError("Preserve immutable checkpoints and incomplete temporary files")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": 1, "registration_sha256": registration_sha256, "parameter_names": names,
               "adapter": tensors, "optimizer": optimizer.state_dict(), "state": state,
               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if cuda else []}
    with temporary.open("xb") as output:
        torch.save(payload, output)
    temporary.replace(path)
    return {"sha256": digest(path), "bytes": path.stat().st_size, "path": str(path),
            "registration_sha256": registration_sha256}


def load_checkpoint(path, expected_sha256, named, optimizer, registration_sha256, *, cuda=False):
    import torch
    names = parameter_names(named, optimizer)
    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError("Checkpoint content changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("format") != 1 or saved.get("registration_sha256") != registration_sha256
            or saved.get("parameter_names") != names or set(saved.get("adapter", {})) != set(names)
            or bool(saved.get("cuda_rng")) != cuda):
        raise ValueError("Checkpoint does not belong to this continuation and parameter layout")
    json.dumps(saved["state"], allow_nan=False)
    for name, parameter in named:
        value = saved["adapter"][name]
        if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all().item():
            raise ValueError("Adapter tensor shape, dtype or finite-value check failed")
    with torch.no_grad():
        for name, parameter in named:
            parameter.copy_(saved["adapter"][name])
    optimizer.load_state_dict(saved["optimizer"])
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(saved["torch_rng"])
    if cuda:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["state"]
