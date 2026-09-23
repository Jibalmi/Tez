"""Keep only the first N transformer blocks of a GGUF (depth pruning for a mid-depth readout).

The output norm and head are kept, so the pruned model's letter logits are the logit lens at layer N and
llama-server `--embeddings --pooling last` returns the normed layer-N state of the last token (what a
mid-depth probe reads). Works for architectures whose per-layer type follows from the layer index
(e.g. qwen35: full attention every `full_attention_interval` layers). Needs `pip install gguf`.
"""
from __future__ import annotations

from pathlib import Path

from .errors import TezError


def truncate_gguf(src: str | Path, n_blocks: int, dst: str | Path) -> dict:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise TezError(f"no such file: {src}")
    if src.resolve() == dst.resolve():
        raise TezError("the output file must differ from the input file")
    try:
        import gguf
    except ImportError as exc:
        raise TezError("tez truncate needs the gguf package: pip install \"tez-decisions[truncate]\"") from exc
    n = int(n_blocks)
    reader = gguf.GGUFReader(str(src), "r")
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    key_blocks = f"{arch}.block_count"
    old = int(reader.fields[key_blocks].contents())
    if not 0 < n < old:
        raise TezError(f"{src.name} has {old} blocks; N must be between 1 and {old - 1}")
    writer = gguf.GGUFWriter(str(dst), arch=arch, endianess=reader.endianess)
    if gguf.Keys.General.ALIGNMENT in reader.fields:
        writer.data_alignment = reader.fields[gguf.Keys.General.ALIGNMENT].contents()
    for fld in reader.fields.values():
        if fld.name == gguf.Keys.General.ARCHITECTURE or fld.name.startswith("GGUF."):
            continue
        val_type = fld.types[0]
        sub_type = fld.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = fld.contents()
        if fld.name == key_blocks:
            val = n
        elif val_type == gguf.GGUFValueType.ARRAY and isinstance(val, list) and len(val) == old and fld.name.startswith(arch + "."):
            val = val[:n]   # per-layer arrays (head counts, layer patterns) follow the kept blocks
        writer.add_key_value(fld.name, val, val_type, sub_type=sub_type)
    keep = [t for t in reader.tensors if not (t.name.startswith("blk.") and int(t.name.split(".")[1]) >= n)]
    for t in keep:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in keep:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    return {"src": str(src), "dst": str(dst), "arch": arch, "blocks": [old, n], "tensors": [len(reader.tensors), len(keep)]}
