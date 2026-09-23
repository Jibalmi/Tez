"""Keep only the first N transformer blocks of a GGUF (depth pruning for early readout).

The output norm and the output head are kept, so the pruned model's letter logits are exactly the
logit lens at layer N, and llama-server `--embeddings --pooling last` returns the normed layer-N
state of the last token (what a mid-depth probe reads). Works for architectures whose per-layer
type is derived from the layer index (e.g. qwen35: full attention every `full_attention_interval`).

  py experiments/gguf_truncate.py tools/models/Qwen3.5-4B-Q8_0.gguf 20 tools/models/Qwen3.5-4B-Q8_0-L20.gguf
"""
from __future__ import annotations

import sys

import gguf


def main(src, n, dst):
    n = int(n)
    reader = gguf.GGUFReader(src, "r")
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    if gguf.Keys.General.ALIGNMENT in reader.fields:
        writer.data_alignment = reader.fields[gguf.Keys.General.ALIGNMENT].contents()
    key_blocks = f"{arch}.block_count"
    old = reader.fields[key_blocks].contents()
    assert n < old, f"{n} >= {old}"
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == key_blocks:
            val = n
        elif val_type == gguf.GGUFValueType.ARRAY and isinstance(val, list) and len(val) == old and field.name.startswith(arch + "."):
            val = val[:n]   # per-layer arrays (head counts, layer patterns) follow the kept blocks
        writer.add_key_value(field.name, val, val_type, sub_type=sub_type)
    keep = [t for t in reader.tensors if not (t.name.startswith("blk.") and int(t.name.split(".")[1]) >= n)]
    for t in keep:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
    for t in keep:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"{src}: {old} -> {n} blocks, {len(reader.tensors)} -> {len(keep)} tensors -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
