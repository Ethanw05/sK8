# PS3 Bone and Skin Data Walkthrough

This note documents the deterministic path we use to pull bone transforms, rotations, and skinning data from a Skate 3 PlayStation 3 arena file.

## 1. Follow the arena header to the dictionary

1. Read the global arena header defined by `ArenaFileHeader` to obtain the counts and offsets that drive the rest of the parse. In particular:【F:sK8/Renderware/Arena/Arena.cs†L30-L52】
   * `NumEntries` (0x20) tells us how many dictionary rows are serialized.
   * `DictStart` (0x30) points to the start of the dictionary table.
   * `Sections` (0x34) points at the section manifest, which in turn leads us to the type list (`ArenaSectionTypes`).
   * `ResourceMainBase` (0x44) gives the base address we must add when resolving `RW_CORE_BASERESOURCE_*` entries.
2. Walk each 0x18-byte dictionary entry (`ArenaDictionaryEntry`). For base resources (`TypeID` between `RW_CORE_BASERESOURCE_0` and `RW_CORE_BASERESOURCE_4`) add `ResourceMainBase` to the stored pointer; for every other type use the pointer as-is to locate the object payload.【F:sK8/Renderware/Arena/Arena.cs†L7-L26】【F:sK8/Renderware/ERwObjectType.cs†L31-L47】

## 2. Identify the Carrier skeleton block

The skeleton (“Carrier”) lives in one of the dictionary objects. We detect it by validating the mini-header at `blockStart + 0x20`:

| Offset | Type  | Description |
| ------:| ----- | ----------- |
| 0x00   | u32   | Offset (relative to block start) to the inverse bind matrix array |
| 0x04   | u32   | Offset to table A (unused for skin extraction) |
| 0x08   | u32   | Offset to the bone name pointer table |
| 0x0C   | u32   | Offset to the bone name string pool |
| 0x10   | u16   | Count for table A |
| 0x12   | u16   | Unknown (padding/flags) |
| 0x14   | u16   | Bone count |
| 0x16   | u16   | Unknown (padding/flags) |
| 0x18   | u32   | Offset to additional data (0x820 block) |
| 0x1C   | u32   | Offset to additional data (0x8A0 block) |

We only accept entries whose offsets resolve inside the block and whose first bone name pointer produces a printable, null-terminated ASCII string. The relative offsets are added to `blockStart` to obtain absolute addresses.

## 3. Recover bone names, transforms, and rotations

1. Use the pointer table (`offTblIdx`) to read `boneCount` relative offsets. Each offset leads to a C-string in the name pool. This yields the stable, global bone order used throughout the file.
2. Read the inverse bind matrices starting at `blockStart + offIBM`. The game serializes 4×4 matrices in row-major order with big-endian IEEE 754 floats, one matrix per bone. Each matrix packs the bone’s rotation (upper 3×3) and translation (last row’s XYZ).
3. Invert every inverse bind matrix to obtain the bind pose matrices. These matrices directly encode the bone-space transforms; extracting rotations simply requires decomposing the bind matrix’s 3×3 submatrix (e.g., to a quaternion) if a different representation is needed.

## 4. Locate the bone palette for the mesh

Mesh draw calls reference bones through a palette stored in the `PEGASUS_RENDEROPTIMESHDATA` object (type 0x00EB0023). Once we find the dictionary row with that type ID, we read a `uint16` array at `blockStart + 0x6C` until we hit either `0xFFFF` or a value ≥ `boneCount`. Each entry is the global bone index that palette slot points to, so palette slot `n` resolves to `boneNames[palette[n]]` if the value is in range.【F:sK8/Renderware/ERwObjectType.cs†L181-L216】

## 5. Map vertex indices and weights back to bones

1. Resolve the vertex descriptor (`VDES`, type 0x000200E9), vertex buffer (`VB`, type 0x000200EA), and optional index buffer (`IB`, type 0x000200EB) that describe the mesh stream. The PS3 vertex descriptor format documented in `Renderware/Arena/VertexDescriptor PS3` provides the per-element metadata we need: each element stores the vertex type, component count, stream, and byte offset within that stream.【F:sK8/Renderware/Arena/VertexDescriptor PS3†L1-L40】
2. Choose the descriptor stream that contains both `BLENDINDICES` and `BLENDWEIGHT` elements. Record its stride and the offsets of those two elements.
3. Decode each vertex by respecting the element’s `vertexType`:

| vertexType | Bytes per component | Notes |
| ---------- | ------------------- | ----- |
| `UB` (0x04) / `UB256` (0x07) | 1 | Unsigned bytes; divide by 255 if normalized |
| `S1` (0x01) / `S32K` (0x05) | 2 | Signed shorts; normalize if required |
| `SF` (0x03) | 2 | IEEE 754 half floats |
| `F` (0x02) | 4 | IEEE 754 32-bit floats |
| `CMP` (0x06) | 4 | Packed 2_10_10_10 data (not used for weights/indices) |

Apply the component size when stepping through both the indices and the weights so that multi-byte encodings (half/float) are handled correctly.
4. Convert the four palette-relative indices for each vertex into global bone indices by looking them up in the palette array. The associated weights are normalized so that their sum is 1 (after discarding near-zero contributions).
5. The resulting `(boneIndex, weight)` pairs per vertex give the final skinning links in terms of the carrier skeleton defined earlier.

## 6. Putting it together

Following the chain—header → dictionary → carrier block → bind matrices → render-opt mesh palette → vertex descriptor—we recover a deterministic bone list with bind transforms, a palette that constrains each draw call to a subset of bones, and per-vertex influences that map back to the global bone indices. This data is sufficient to rebuild the skeleton hierarchy (with the relationships in `Renderware/Skeleton/Hierarchy`), extract rotations, and drive skinning in external tools.

