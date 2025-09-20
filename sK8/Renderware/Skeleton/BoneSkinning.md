# PS3 Bone + Skin Data (Skate 3, PS3) — Deterministic Walkthrough

This document describes a reproducible path to extract bone names, bind transforms (and rotations if desired), and per-vertex skinning from a Skate 3 PS3 arena/PSG.

Validated against multiple PS3 assets. Focus is bones + skin; animation and physics are out of scope.

# 1) Arena header → dictionary

Read the global header (aka ArenaFileHeader) to get counts/offsets, then walk the dictionary:

NumEntries @ 0x20 → number of dictionary rows.

DictStart @ 0x30 → start of the dictionary table.

Sections @ 0x34 → section manifest; from here you can resolve the type list (section types → concrete TypeId).

ResourceMainBase @ 0x44 → add this base when resolving pointers for RW_CORE_BASERESOURCE_* types.

Each dictionary row is 0x18 bytes. For rows whose TypeId falls in the BaseResource range, compute blockStart = ResourceMainBase + Ptr; otherwise, blockStart = Ptr. The object payload lives at blockStart .. blockStart+Size.

# 2) Find the skeleton (“Carrier”) block

The skeleton lives inside one dictionary object. Identify it by a mini-header at blockStart + 0x20 that passes strict bounds checks and yields a printable first bone name.

Carrier mini-header (relative to blockStart):

Off	Type	Meaning
0x00	u32	offIBM → inverse bind matrix array (row-major, BE floats), length = boneCount × 64
0x04	u32	offTblA → Table A (see below)
0x08	u32	offIdx → bone name pointer table (u32 rel-offsets)
0x0C	u32	offNames → bone name pool (C-strings)
0x10	u16	cntA → Table A count
0x12	u16	flags/reserved (observed constant = 1)
0x14	u16	boneCount
0x16	u16	flags/reserved (observed constant = 1)
0x18	u32	off820 → aux param block (~8 floats)
0x1C	u32	off8A0 → aux attribute block (ragged, variable)

Acceptance rules (hard):

blockStart + offIBM + boneCount*64 is in-bounds.

blockStart + offIdx + boneCount*4 is in-bounds.

offNames points inside the block; the first pointer in offIdx resolves to a printable, NUL-terminated ASCII string.

Table A (what it is)

cntA u32s starting at offTblA. Each is a relative offset into the name pool. In “full” carriers it forms a prefix TOC over the bone names (plus a couple sentinels like 0x00000007 and 0x00000000). In minimal carriers it may collapse to the sentinel only.
Not used for skin extraction.

Aux blocks (what they are not)

off820 (“0x820 block”): ~8×f32 global params. Not per-bone. Ignore for skin.

off8A0 (“0x8A0 block”): variable, sparse float sections (lots of zeros). Not a uniform boneCount×stride table. Ignore for skin.

# 3) Recover bone names, bind transforms, and rotations

Bone names
From offIdx, read boneCount u32 relative offsets. Each points into the string pool at offNames. The order of these strings is the global bone index used everywhere else.

Inverse bind matrices (IBMs)
From offIBM, read boneCount matrices. Each is row-major, big-endian IEEE-754, 4×4 (16 floats).

Translation lives in row 3 (elements [3,0..2]).

The upper 3×3 encodes the bone-space orientation.

Bind pose (optional rotations)
Invert each IBM to get the bind matrix. If you need quaternions, decompose the 3×3 submatrix of the bind matrix (be careful with scale/shear; most are rigid).

# 4) Get the bone palette for the mesh

Find the dictionary row with PEGASUS_RENDEROPTIMESHDATA (0x00EB0023). At blockStart + 0x6C, read a u16 array until:

value == 0xFFFF (sentinel), or

value ≥ boneCount (out of range).

Each entry is a global bone index, i.e., palette slot n → boneNames[palette[n]].

# 5) Map vertex indices + weights → global bones

Resolve the typical trio:

VDES: 0x000200E9 (vertex descriptor)

VB: 0x000200EA (vertex buffer)

IB: 0x000200EB (index buffer; optional for morph streams)

Use the PS3 vertex descriptor to pick the stream that contains both BLENDINDICES and BLENDWEIGHT. Record the stream’s stride and the byte offsets of those two elements.

Decode by vertexType (per-component width):

vertexType	Bytes	Notes
UB (0x04) / UB256 (0x07)	1	Unsigned 8-bit; UB is normalized (÷255). UB256 may be unnormalized in some assets — read raw then normalize overall.
S1 (0x01) / S32K (0x05)	2	16-bit; weights commonly behave as UNORM16 (÷65535).
SF (0x03)	2	16-bit IEEE-754 half-float.
F (0x02)	4	32-bit float.
CMP (0x06)	4	Packed 2_10_10_10; not used for indices/weights.

Important: When stepping to components 0..3 of indices/weights, advance by the element’s component size (1/2/4), not by 1 byte unconditionally.

Per vertex:

Read 4 palette-relative indices and 4 weights using the correct vertexType widths.

Normalize weights (zero tiny contributions; re-scale to sum≈1).

Map each index through the palette to a global bone index (or -1 if OOB).

Emit up to four (globalBoneIndex, weight) pairs per vertex.

# 6) Putting it together

Pipeline (deterministic):

Arena header → dictionary

Carrier mini-header → names + IBMs (then invert to bind pose)

Render-opt mesh (0x00EB0023) → palette

VDES/VB(/IB) → indices + weights (respect vertexType sizes)

Map palette-relative → global indices; normalize weights

This yields:

a stable bone list with bind transforms,

a palette scoping each draw to a subset of bones,

and per-vertex influences expressed in global bone indices.

Hierarchy note: parent/child relationships are not in the Carrier block. Use your external hierarchy manifest (e.g., Renderware/Skeleton/Hierarchy) or the physics skeleton to rebuild the tree. Rotations (quats) are optional; matrices already carry the pose.

Implementation tips & pitfalls

Endianness: all Carrier floats are big-endian. Don’t double-swap: assemble the BE u32 bit pattern, then reinterpret as float.

Table A: prefix TOC over the name pool; ignore for skin.

off820 / off8A0: auxiliary; ignore for skin.

UB256: if you see non-normalized bytes, read raw 0–255 then rely on your normalization pass.

Morph streams: some VDES/VB pairs have no INDICES (morph-only). Choose the stream with both INDICES + WEIGHTS for skin extraction.

Sanity checks: reject carriers whose offsets don’t land in-block or whose first name isn’t printable ASCII.

That’s the accurate, battle-tested path.
