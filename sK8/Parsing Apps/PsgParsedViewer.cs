// PSG_Viewer_Skate3_PS3_VertexFlagsCorrect.cs
// Self-contained WinForms PSG viewer for Skate 3 (PS3)
// Multi-submesh rendering + UVs + Morph Target preview (attaches VB-only streams as morphs),
// with a VertexDescriptor parser consistent with our ground truths:
//  - Endianness: BIG-ENDIAN for all ints/floats
//  - ElementType names: XYZ/WEIGHTS/NORMAL/VERTEXCOLOR/SPECULAR/INDICES/TEX0..TEX5/TANGENT/BINORMAL
//  - VertexType mapping: S1/F/SF/UB/S32K/CMP/UB256
//  - typesFlags = OR of (1 << ElementType)
//  - Frequency divider mask: one bit per element; 1=MODULO, 0=DIVIDE
//  - Packed vectors: DEC3N (CMP 2_10_10_10) and Oct16 alternative
//  - SectionTypes block is found robustly by scanning (no magic header offsets)
//
// Target: .NET Framework 4.8 or .NET 6+ (Windows). Build with /unsafe.
// No external dependencies.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace PSG_Viewer
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.Run(new MainForm());
        }
    }

    // ======== Simple math structs ========

    struct Vec2 { public float X, Y; public Vec2(float x, float y) { X = x; Y = y; } }

    struct Vec3
    {
        public float X, Y, Z;
        public Vec3(float x, float y, float z) { X = x; Y = y; Z = z; }
        public static Vec3 operator +(Vec3 a, Vec3 b) => new Vec3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        public static Vec3 operator -(Vec3 a, Vec3 b) => new Vec3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        public static Vec3 operator *(Vec3 a, float s) => new Vec3(a.X * s, a.Y * s, a.Z * s);
        public float Length() => (float)Math.Sqrt(X * X + Y * Y + Z * Z);
        public static float Dot(in Vec3 a, in Vec3 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        public static Vec3 Cross(in Vec3 a, in Vec3 b) => new Vec3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
        public static Vec3 Normalize(in Vec3 v) { float l = v.Length(); return (l > 1e-8f) ? new Vec3(v.X / l, v.Y / l, v.Z / l) : new Vec3(0, 1, 0); }
        public static Vec3 SwapYZ(in Vec3 v) => new Vec3(v.X, v.Z, v.Y);
        public static Vec3 Lerp(in Vec3 a, in Vec3 b, float t) => a + (b - a) * t;
    }

    struct Tri { public int A, B, C; public Tri(int a, int b, int c) { A = a; B = b; C = c; } }

    // ======== Scene data (multi-submesh + morphs) ========

    sealed class MorphTarget
    {
        public string Name;
        public int VbIndex;   // original VB dict index (for logging/ordering)
        public int VdIndex;
        public readonly List<Vec3> Positions = new();
    }

    sealed class SubMeshData
    {
        public string Name;
        public int VertexStride;
        public int VbIndex, IbIndex, VdIndex; // for pairing/logging
        public readonly List<Vec3> Positions = new();
        public readonly List<Vec2> UVs = new(); // TEX0 only (display)
        public readonly Dictionary<int, List<Vec2>> UVSets = new(); // TEXn (n=0..7)
        public readonly List<Vec3> Normals = new();
        public readonly List<Vec3> Tangents = new();
        public readonly List<Vec3> Binormals = new();
        public readonly List<Tri> Triangles = new();
        public readonly List<MorphTarget> Morphs = new();
        public int UvScaleDivisor = 1;
        // simple scalar to visualize presence of weights (we keep max of quartet here)
        public readonly List<float> Weights = new();
        public override string ToString() => Name;
    }

    sealed class SceneData
    {
        public string SourceFile;
        public readonly List<SubMeshData> SubMeshes = new();
        public Action<string> Logger;
        public void Log(string s) => Logger?.Invoke(s);
    }

    // ======== Big-endian readers & helpers ========

    static class BE
    {
        public static ushort U16(byte[] d, int o) => (ushort)((d[o] << 8) | d[o + 1]);
        public static short I16(byte[] d, int o) => unchecked((short)U16(d, o));
        public static uint U32(byte[] d, int o) => ((uint)d[o] << 24) | ((uint)d[o + 1] << 16) | ((uint)d[o + 2] << 8) | d[o + 3];
        public static int I32(byte[] d, int o) => unchecked((int)U32(d, o));
        public static float F32(byte[] d, int o)
        {
            uint v = U32(d, o);
            unsafe { return *(float*)&v; }
        }
        public static float HalfToFloat(ushort h)
        {
            uint sign = (uint)(h >> 15) & 0x1u;
            uint exp = (uint)(h >> 10) & 0x1Fu;
            uint mant = (uint)h & 0x3FFu;
            uint f;
            if (exp == 0)
            {
                if (mant == 0) f = sign << 31;
                else
                {
                    while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
                    mant &= 0x3FFu;
                    exp = (uint)(127 - 15 + 1 + (int)exp);
                    f = (sign << 31) | (exp << 23) | (mant << 13);
                }
            }
            else if (exp == 31)
            {
                f = (sign << 31) | 0x7F800000u | (mant << 13);
            }
            else
            {
                exp = exp + (127 - 15);
                f = (sign << 31) | (exp << 23) | (mant << 13);
            }
            unsafe { return *(float*)&f; }
        }
        public static string Hex(byte[] d, int off, int len)
        {
            len = Math.Max(0, Math.Min(len, d.Length - off));
            var sb = new StringBuilder(len * 3);
            for (int i = 0; i < len; i++) { if (i > 0) sb.Append(' '); sb.Append(d[off + i].ToString("X2")); }
            return sb.ToString();
        }
        public static string Bits(uint v, int width = 32)
        {
            var sb = new StringBuilder(width);
            for (int i = width - 1; i >= 0; i--) sb.Append(((v >> i) & 1) != 0 ? '1' : '0');
            return sb.ToString();
        }
    }

    // ======== PSG parser (Skate 3 PS3) ========

    static class PsgParser
    {
        // Toggle extra logging
        const bool VerboseVDES = true;

        // RW type IDs (subset)
        const uint RW_CORE_SECTIONTYPES = 0x00010005;
        const uint RW_GRAPHICS_VERTEXDESCRIPTOR = 0x000200E9;
        const uint RW_GRAPHICS_VERTEXBUFFER = 0x000200EA;
        const uint RW_GRAPHICS_INDEXBUFFER = 0x000200EB;

        static bool IsBaseResource(uint type) => (type >= 0x00010030 && type <= 0x0001003F); // RW_CORE_BASERESOURCE_0..15

        sealed class DictEntry
        {
            public int Index; public uint Ptr, Size, Align, TypeIndex, TypeId; public int BlockStart, BlockEnd;
        }

        sealed class Arena
        {
            public readonly List<DictEntry> Dict = new();
            public readonly List<uint> Types = new();
            public uint DictStart; public uint ResourceMainSize; public uint NumEntries;
        }

        // PS3 VertexDescriptor element (8 bytes)
        struct VDElem
        {
            public byte VertexType;     // +0 (1=S1, 2=F, 3=SF(half), 6=CMP, 4=UB, 7=UB256, 5=S32K)
            public byte NumComponents;  // +1
            public byte Stream;         // +2
            public byte Offset;         // +3
            public ushort Stride;       // +4..5
            public byte Type;           // +6 (ElementType 0..15)
            public byte Class;          // +7 (1=PERVERTEX, 2=PERINSTANCE)
        }

        // Captured descriptor info
        sealed class VDInfo
        {
            public int Stride;
            public VDElem? PosElem;
            public VDElem? NormalElem;
            public VDElem? TangentElem;
            public VDElem? BinormalElem;
            public readonly Dictionary<int, VDElem> TexElems = new(); // key: TEX index 0..7

            // For logging/analysis
            public int RefCount; public uint TypesFlags; public ushort NumStreams, NumElements, FreqMask, Pad0;
            public List<VDElem> RawElems = new();
        }

        // PS3 enums (subset)
        static readonly string[] ElementTypeName = new string[256];
        static PsgParser()
        {
            for (int i = 0; i < ElementTypeName.Length; i++) ElementTypeName[i] = $"Type{i}";
            ElementTypeName[0] = "ELEMENTTYPE_XYZ";
            ElementTypeName[1] = "ELEMENTTYPE_WEIGHTS";
            ElementTypeName[2] = "ELEMENTTYPE_NORMAL";
            ElementTypeName[3] = "ELEMENTTYPE_VERTEXCOLOR";
            ElementTypeName[4] = "ELEMENTTYPE_SPECULAR";
            // 5,6 unassigned
            ElementTypeName[7] = "ELEMENTTYPE_INDICES";
            ElementTypeName[8] = "ELEMENTTYPE_TEX0";
            ElementTypeName[9] = "ELEMENTTYPE_TEX1";
            ElementTypeName[10] = "ELEMENTTYPE_TEX2";
            ElementTypeName[11] = "ELEMENTTYPE_TEX3";
            ElementTypeName[12] = "ELEMENTTYPE_TEX4";
            ElementTypeName[13] = "ELEMENTTYPE_TEX5";
            ElementTypeName[14] = "ELEMENTTYPE_TANGENT";   // some docs overlap with TEX6; in assets this is tangent/binormal
            ElementTypeName[15] = "ELEMENTTYPE_BINORMAL";  // some docs overlap with TEX7
        }

        static string VertexTypeName(byte vt) => vt switch
        {
            0x01 => "CELL_GCM_VERTEX_S1 (S16)",
            0x02 => "CELL_GCM_VERTEX_F (F32)",
            0x03 => "CELL_GCM_VERTEX_SF (F16)",
            0x04 => "CELL_GCM_VERTEX_UB (U8 norm)",
            0x05 => "CELL_GCM_VERTEX_S32K (S16 non-norm)",
            0x06 => "CELL_GCM_VERTEX_CMP (DEC3N 10:10:10:2)",
            0x07 => "CELL_GCM_VERTEX_UB256 (U8 non-norm)",
            _ => $"0x{vt:X2}"
        };
        static string ElementClassName(byte c) => c switch
        {
            0x01 => "PERVERTEX",
            0x02 => "PERINSTANCE",
            0x00 => "NA",
            _ => $"0x{c:X2}"
        };

        public static SceneData Parse(byte[] file, Action<string> LOG)
        {
            var scene = new SceneData { Logger = LOG, SourceFile = "PSG" };

            if (file.Length < 0x60) throw new InvalidDataException("Too small to be a valid Arena.");

            // Pull key header integers we depend on (these offsets are stable in our writer).
            uint numEntries = BE.U32(file, 0x20);
            uint dictStart = BE.U32(file, 0x30);
            // mainBase is "end of main resource" (ResourceDescriptor[0].Size) — used when dereferencing BaseResource blocks.
            uint mainBase = BE.U32(file, 0x44);

            scene.Log($"[Header] entries={numEntries} dict@0x{dictStart:X} mainResourceBase@0x{mainBase:X}");

            var arena = new Arena { DictStart = dictStart, ResourceMainSize = mainBase, NumEntries = numEntries };

            // ===== SectionTypes (robust) =====
            // Scan entire file for a SectionTypes block: [Type=0x00010005][Num][DictOffset], then follow the inline dictionary.
            // This avoids magic offsets and matches how our serializer writes sections.
            {
                int found = -1;
                for (int p = 0; p <= file.Length - 12; p += 4)
                {
                    if (BE.U32(file, p) != RW_CORE_SECTIONTYPES) continue;
                    found = p; break;
                }
                if (found >= 0)
                {
                    uint num = BE.U32(file, found + 4);
                    uint dictOff = BE.U32(file, found + 8);
                    int tp = found + (int)dictOff;
                    arena.Types.Clear();
                    for (int i = 0; i < num && (tp + i * 4) <= file.Length - 4; i++)
                        arena.Types.Add(BE.U32(file, tp + i * 4));
                    scene.Log($"[Types] count={arena.Types.Count} @0x{found:X} dictOff=+0x{dictOff:X}");
                }
                else
                {
                    scene.Log("[Types] WARNING: SectionTypes not found by scan. Falling back to TypeID field in dictionary.");
                }
            }

            // ===== Dictionary =====
            for (int i = 0, q = (int)dictStart; i < numEntries && q + 0x18 <= file.Length; i++, q += 0x18)
            {
                uint ptr = BE.U32(file, q + 0x00);
                uint size = BE.U32(file, q + 0x08);
                uint align = BE.U32(file, q + 0x0C);
                uint typeIndex = BE.U32(file, q + 0x10);
                uint typeId = BE.U32(file, q + 0x14);

                // If SectionTypes exists, typeIndex maps to concrete TypeID there (ground truth)
                if (arena.Types.Count > 0 && typeIndex < arena.Types.Count) typeId = arena.Types[(int)typeIndex];

                // BaseResource entries are located after "mainBase", non-BaseResource are absolute offsets.
                int blockStart = IsBaseResource(typeId) ? (int)(mainBase + ptr) : (int)ptr;
                int blockEnd = Math.Min(file.Length, Math.Max(blockStart, blockStart + (int)size));

                arena.Dict.Add(new DictEntry
                {
                    Index = i,
                    Ptr = ptr,
                    Size = size,
                    Align = align,
                    TypeIndex = typeIndex,
                    TypeId = typeId,
                    BlockStart = blockStart,
                    BlockEnd = blockEnd
                });
            }

            // Gather objects
            var listVD = arena.Dict.Where(d => d.TypeId == RW_GRAPHICS_VERTEXDESCRIPTOR).OrderBy(d => d.Index).ToList();
            var listVB = arena.Dict.Where(d => d.TypeId == RW_GRAPHICS_VERTEXBUFFER).OrderBy(d => d.Index).ToList();
            var listIB = arena.Dict.Where(d => d.TypeId == RW_GRAPHICS_INDEXBUFFER).OrderBy(d => d.Index).ToList();
            scene.Log($"[Counts] VDES={listVD.Count} VB={listVB.Count} IB={listIB.Count}");

            // Parse all VDES
            var vdInfos = listVD.Select(vd => ParseVDES(file, vd.BlockStart, vd.BlockEnd, scene.Log)).ToList();

            // Associate each VB with a VDES (nearest following, else nearest overall)
            int NextVDForVB(int vbDictIndex)
            {
                int nextVbIndex = listVB.Where(v => v.Index > vbDictIndex).Select(v => v.Index).DefaultIfEmpty(int.MaxValue).Min();
                var vdBetween = listVD.Where(v => v.Index > vbDictIndex && v.Index < nextVbIndex).OrderBy(v => v.Index).FirstOrDefault();
                if (vdBetween != null) return listVD.IndexOf(vdBetween);
                var vdBefore = listVD.Where(v => v.Index < vbDictIndex).OrderBy(v => Math.Abs(v.Index - vbDictIndex)).FirstOrDefault();
                if (vdBefore != null) return listVD.IndexOf(vdBefore);
                return 0;
            }

            // Build submesh triples (VB, VD, optional IB)
            var subTriples = new List<(DictEntry vb, int vdInfoIdx, DictEntry ib)>();
            if (listVB.Count == listIB.Count)
            {
                for (int i = 0; i < listVB.Count; i++) { var vb = listVB[i]; var ib = listIB[i]; subTriples.Add((vb, NextVDForVB(vb.Index), ib)); }
            }
            else
            {
                var used = new HashSet<int>();
                foreach (var ib in listIB)
                {
                    var vb = listVB.OrderBy(v => Math.Abs(v.Index - ib.Index)).First();
                    used.Add(vb.Index);
                    subTriples.Add((vb, NextVDForVB(vb.Index), ib));
                }
                foreach (var vb in listVB) if (!used.Contains(vb.Index)) subTriples.Add((vb, NextVDForVB(vb.Index), null));
            }

            // Build SubMeshData (including VB-only morph candidates)
            int smCounter = 0;
            foreach (var (vb, vdIdx, ib) in subTriples)
            {
                var vd = vdInfos[vdIdx];
                var sm = new SubMeshData
                {
                    Name = $"SM_{smCounter++:00}_VB{vb.Index}_IB{(ib?.Index.ToString() ?? "—")}",
                    VbIndex = vb.Index,
                    IbIndex = ib?.Index ?? -1,
                    VdIndex = listVD[vdIdx].Index
                };
                scene.Log($"[SubMesh] {sm.Name}  VDES#{listVD[vdIdx].Index}  VB#{vb.Index}  IB#{(ib?.Index.ToString() ?? "none")}");

                // VertexBuffer → BaseResource
                if (vb.BlockStart + 12 > file.Length) throw new InvalidDataException("VB header OOB.");
                uint vbBRIndex = BE.U32(file, vb.BlockStart + 0);
                var brVB = arena.Dict.ElementAtOrDefault((int)vbBRIndex);
                if (brVB == null || !IsBaseResource(brVB.TypeId)) throw new InvalidDataException("VB → BaseResource index invalid.");
                int vbStart = brVB.BlockStart; int vbSize = (int)brVB.Size;
                int stride = vd.Stride > 0 ? vd.Stride : vbSize; // fall back defensively
                int vCount = (stride > 0) ? Math.Max(0, vbSize / stride) : 0;
                sm.VertexStride = stride;

                // IndexBuffer (optional)
                int ibStart = -1, ibCount16 = 0;
                if (ib != null)
                {
                    if (ib.BlockStart + 16 > file.Length) throw new InvalidDataException("IB header OOB.");
                    uint ibBRIndex = BE.U32(file, ib.BlockStart + 0);
                    var brIB = arena.Dict.ElementAtOrDefault((int)ibBRIndex);
                    if (brIB == null || !IsBaseResource(brIB.TypeId)) throw new InvalidDataException("IB → BaseResource index invalid.");
                    ibStart = brIB.BlockStart; int ibSize = (int)brIB.Size; ibCount16 = Math.Max(0, ibSize / 2);
                    scene.Log($"  VB BR#{vbBRIndex} @0x{vbStart:X} size=0x{vbSize:X} → vCount≈{vCount}");
                    scene.Log($"  IB BR#{ibBRIndex} @0x{ibStart:X} size=0x{ibSize:X}");
                }
                else scene.Log($"  VB BR#{vbBRIndex} @0x{vbStart:X} size=0x{vbSize:X} → vCount≈{vCount}");

                // ---- Read vertices using ElementType semantics ----
                var posElem = vd.PosElem; var nrmElem = vd.NormalElem; var tanElem = vd.TangentElem; var binElem = vd.BinormalElem;
                vd.TexElems.TryGetValue(0, out var tex0Elem);

                if (posElem == null)
                    scene.Log("  [VDES-Note] No XYZ element declared; applying legacy fallback Float3@+0.");

                for (int i = 0; i < vCount; i++)
                {
                    int vOff = vbStart + i * stride;

                    // Position (XYZ)
                    Vec3 P = new Vec3(0, 0, 0);
                    if (posElem.HasValue)
                    {
                        var e = posElem.Value;
                        P = DecodePosition(file, vOff + e.Offset, e.VertexType, e.NumComponents);
                    }
                    else if (vOff + 12 <= file.Length)
                    {
                        P = new Vec3(BE.F32(file, vOff + 0), BE.F32(file, vOff + 4), BE.F32(file, vOff + 8));
                    }
                    sm.Positions.Add(P);

                    // UVs (TEX0)
                    Vec2 UV = new Vec2(0, 0);
                    if (tex0Elem.VertexType != 0)
                        UV = DecodeUV(file, vOff + tex0Elem.Offset, tex0Elem.VertexType);
                    else if (vOff + 16 <= file.Length) // fallback: two halfs after pos
                        UV = new Vec2(BE.HalfToFloat(BE.U16(file, vOff + 12)), BE.HalfToFloat(BE.U16(file, vOff + 14)));
                    sm.UVs.Add(UV);

                    // BLENDWEIGHTS (ElementType=1) — decode up to 4, renormalize (ground truth)
                    var weightsElem = vd.RawElems.FirstOrDefault(e => e.Type == 1);
                    float wMax = 1f;
                    if (weightsElem.Type == 1)
                    {
                        var w = DecodeWeights4(file, vOff + weightsElem.Offset, weightsElem.VertexType, weightsElem.NumComponents);
                        float sum = Math.Max(1e-6f, w.w0 + w.w1 + w.w2 + w.w3);
                        wMax = Math.Max(Math.Max(w.w0, w.w1), Math.Max(w.w2, w.w3)) / sum;
                    }
                    sm.Weights.Add(wMax);

                    // Normals / Tangents / Binormals
                    sm.Normals.Add(DecodeDir(file, vOff, nrmElem));
                    sm.Tangents.Add(DecodeDir(file, vOff, tanElem));
                    sm.Binormals.Add(DecodeDir(file, vOff, binElem));
                }

                // ---- Read indices (triangles list) ----
                if (ibStart >= 0 && ibCount16 >= 3)
                {
                    int triCount = ibCount16 / 3;
                    for (int i = 0; i < triCount; i++)
                    {
                        int a = BE.U16(file, ibStart + (i * 6) + 0);
                        int b = BE.U16(file, ibStart + (i * 6) + 2);
                        int c = BE.U16(file, ibStart + (i * 6) + 4);
                        if ((uint)a < (uint)vCount && (uint)b < (uint)vCount && (uint)c < (uint)vCount) sm.Triangles.Add(new Tri(a, b, c));
                    }
                }

                // If normals empty or degenerate, recompute from faces
                RecomputeNormalsIfNeeded(sm);
                scene.SubMeshes.Add(sm);
            }

            // Attach VB-only entries as morph targets to compatible base meshes
            AttachMorphTargets(scene);
            return scene;
        }

        // Decode quartet of weights; treat U8 (norm or UB256) as /255f and renormalize later (ground truth)
        static (float w0, float w1, float w2, float w3) DecodeWeights4(byte[] d, int off, byte vt, byte comps)
        {
            float w0 = 0, w1 = 0, w2 = 0, w3 = 0;
            int n = Math.Min((byte)4, Math.Max((byte)1, comps));

            switch (vt)
            {
                case 0x02: // F32
                    if (off + 4 * n <= d.Length)
                    {
                        if (n > 0) w0 = BE.F32(d, off + 0);
                        if (n > 1) w1 = BE.F32(d, off + 4);
                        if (n > 2) w2 = BE.F32(d, off + 8);
                        if (n > 3) w3 = BE.F32(d, off + 12);
                    }
                    break;

                case 0x04: // U8 normalized
                case 0x07: // U8 non-norm (treat like bytes, then normalize quartet)
                    if (off + n <= d.Length)
                    {
                        if (n > 0) w0 = d[off + 0] / 255f;
                        if (n > 1) w1 = d[off + 1] / 255f;
                        if (n > 2) w2 = d[off + 2] / 255f;
                        if (n > 3) w3 = d[off + 3] / 255f;
                    }
                    break;

                default:
                    // Other encodings are rare for weights; leave zeros → will become 1 via fallback in caller
                    break;
            }
            return (w0, w1, w2, w3);
        }

        static Vec3 DecodePosition(byte[] d, int off, byte vt, byte comps)
        {
            switch (vt)
            {
                case 0x02: // F32
                    if (off + 12 <= d.Length) return new Vec3(BE.F32(d, off + 0), BE.F32(d, off + 4), BE.F32(d, off + 8));
                    break;

                // Assets often store XYZ as raw S16 triplets (S1 or S32K). Keep that behavior.
                case 0x01: // S1 (S16 normalized in docs, but assets use raw)
                case 0x05: // S32K (S16 non-norm)
                    if (off + 6 <= d.Length) return new Vec3(BE.I16(d, off + 0), BE.I16(d, off + 2), BE.I16(d, off + 4));
                    break;
            }
            return new Vec3(0, 0, 0);
        }

        static Vec2 DecodeUV(byte[] d, int off, byte vt)
        {
            switch (vt)
            {
                case 0x03: // SF (half)
                    if (off + 4 <= d.Length) return new Vec2(BE.HalfToFloat(BE.U16(d, off + 0)), BE.HalfToFloat(BE.U16(d, off + 2)));
                    break;
                case 0x01: // S1 (S16)
                case 0x05: // S32K (S16 non-norm)
                    if (off + 4 <= d.Length) return new Vec2(BE.I16(d, off + 0), BE.I16(d, off + 2));
                    break;
                case 0x02: // F32
                    if (off + 8 <= d.Length) return new Vec2(BE.F32(d, off + 0), BE.F32(d, off + 4));
                    break;
                case 0x04: // UB (U8 normalized)
                case 0x07: // UB256 (U8 non-norm → show as 0..1 after /255)
                    if (off + 2 <= d.Length) return new Vec2(d[off + 0] / 255f, d[off + 1] / 255f);
                    break;
            }
            return new Vec2(0, 0);
        }

        static Vec3 DecodeDir(byte[] d, int vOff, VDElem? elem)
        {
            if (elem == null) return new Vec3(0, 0, 1);
            var e = elem.Value; int off = vOff + e.Offset;


            if (e.VertexType == 0x06)
            {
                if (off + 4 <= d.Length) return DecodeDec3nPS3(d, off);
                return new Vec3(1, 0, 0);
            }

            // Octahedral in S16x2 (common alt storage)
            if (e.VertexType == 0x01 && e.NumComponents == 2)
            {
                if (off + 4 <= d.Length) return DecodeOct16BE(d, off);
                return new Vec3(1, 0, 0);
            }

            // Fallbacks: float3 or S16x3 normalized
            if (e.VertexType == 0x02 && off + 12 <= d.Length)
            {
                var v = new Vec3(BE.F32(d, off + 0), BE.F32(d, off + 4), BE.F32(d, off + 8));
                return Vec3.Normalize(v);
            }
            if (e.VertexType == 0x01 && e.NumComponents >= 3 && off + 6 <= d.Length)
            {
                float x = Math.Max(-1f, Math.Min(1f, BE.I16(d, off + 0) / 32767f));
                float y = Math.Max(-1f, Math.Min(1f, BE.I16(d, off + 2) / 32767f));
                float z = Math.Max(-1f, Math.Min(1f, BE.I16(d, off + 4) / 32767f));
                return Vec3.Normalize(new Vec3(x, y, z));
            }
            return new Vec3(0, 0, 1);
        }

        // Octahedral from two S16 BE components
        static Vec3 DecodeOct16BE(byte[] d, int off)
        {
            short sx = BE.I16(d, off + 0);
            short sy = BE.I16(d, off + 2);
            float x = Math.Max(-1f, Math.Min(1f, sx / 32767f));
            float y = Math.Max(-1f, Math.Min(1f, sy / 32767f));

            float z = 1f - (Math.Abs(x) + Math.Abs(y));
            if (z < 0f)
            {
                float ox = (1f - Math.Abs(y)) * Math.Sign(x);
                float oy = (1f - Math.Abs(x)) * Math.Sign(y);
                z = -(1f - (Math.Abs(x) + Math.Abs(y)));
                x = ox; y = oy;
            }
            var v = new Vec3(x, y, z);
            return Vec3.Normalize(v);
        }

        // DEC3N / INT_2_10_10_10_REV, PS3 byte order corrected
        // DEC3N (signed, normalized) used for normals/tangents/binormals.
        // Packing per pack_normal_dec3n:
        //   scale [-1,1] by 511, round to [-511,511], 10-bit signed each
        //   Bits  2..11 : X (10-bit)
        //   Bits 12..21 : Y (10-bit)
        //   Bits 22..31 : Z (10-bit)
        //   Bits 0..1 unused.
        // NOTE: Do NOT byteswap here; BE.U32 already returns the correct numeric value.
        // We sign-extend from 10 bits and divide by 511, then re-normalize.
        static Vec3 DecodeDec3nPS3(byte[] d, int off)
        {
            if (off + 4 > d.Length) return new Vec3(1, 0, 0);

            uint packed = BE.U32(d, off);

            // Extract 10-bit signed components at the correct bit positions
            int ix = (int)((packed >> 2) & 0x3FF);
            int iy = (int)((packed >> 12) & 0x3FF);
            int iz = (int)((packed >> 22) & 0x3FF);

            // Two's-complement sign-extend from 10 bits
            ix = (ix << 22) >> 22;
            iy = (iy << 22) >> 22;
            iz = (iz << 22) >> 22;

            const float inv = 1.0f / 511.0f;
            var v = new Vec3(ix * inv, iy * inv, iz * inv);

            // Small deviations can accumulate; keep unit length.
            return Vec3.Normalize(v);
        }

        static void RecomputeNormalsIfNeeded(SubMeshData sm)
        {
            bool needs = false;
            if (sm.Triangles.Count == 0) return;
            if (sm.Normals.Count != sm.Positions.Count) needs = true;
            else
            {
                int checks = Math.Min(16, sm.Normals.Count); int zeros = 0;
                for (int i = 0; i < checks; i++) { var n = sm.Normals[i]; if (Math.Abs(n.X) < 1e-5f && Math.Abs(n.Y) < 1e-5f && Math.Abs(n.Z) < 1e-5f) zeros++; }
                if (zeros > checks / 2) needs = true;
            }
            if (!needs) return;

            var acc = new Vec3[sm.Positions.Count];
            foreach (var t in sm.Triangles)
            {
                var a = sm.Positions[t.A]; var b = sm.Positions[t.B]; var c = sm.Positions[t.C];
                var n = Vec3.Normalize(Vec3.Cross(b - a, c - a));
                acc[t.A] = acc[t.A] + n; acc[t.B] = acc[t.B] + n; acc[t.C] = acc[t.C] + n;
            }
            sm.Normals.Clear();
            for (int i = 0; i < acc.Length; i++) { var n = Vec3.Normalize(acc[i]); if (n.Length() < 1e-6f) n = new Vec3(0, 0, 1); sm.Normals.Add(n); }
        }

        public static List<Vec3> BuildNormals(SubMeshData sm, IReadOnlyList<Vec3> positions)
        {
            var acc = new Vec3[positions.Count];
            foreach (var t in sm.Triangles)
            {
                var a = positions[t.A]; var b = positions[t.B]; var c = positions[t.C];
                var n = Vec3.Normalize(Vec3.Cross(b - a, c - a));
                acc[t.A] = acc[t.A] + n; acc[t.B] = acc[t.B] + n; acc[t.C] = acc[t.C] + n;
            }
            var list = new List<Vec3>(positions.Count);
            for (int i = 0; i < positions.Count; i++) { var n = Vec3.Normalize(acc[i]); if (n.Length() < 1e-6f) n = new Vec3(0, 0, 1); list.Add(n); }
            return list;
        }

        // ---- VDES parsing ----
        static VDInfo ParseVDES(byte[] d, int start, int end, Action<string> LOG)
        {
            if (start + 16 > d.Length) throw new InvalidDataException("VDES header OOB.");
            int refCount = BE.I32(d, start + 0);
            uint flags = BE.U32(d, start + 4);
            ushort nStream = BE.U16(d, start + 8);
            ushort nElem = BE.U16(d, start + 10);
            ushort freq = BE.U16(d, start + 12);
            ushort pad0 = BE.U16(d, start + 14);

            LOG?.Invoke($"[VDES] @0x{start:X} header = refCount={refCount}  typesFlags=0x{flags:X8} (bits {BE.Bits(flags)})  numStreams={nStream}  numElements={nElem}  freqMask=0x{freq:X4}  pad0=0x{pad0:X4}");
            if (VerboseVDES) LOG?.Invoke($"[VDES] header bytes: {BE.Hex(d, start, Math.Min(32, end - start))}");

            var elems = new List<VDElem>(nElem);
            int eoff = start + 16;
            for (int i = 0; i < nElem && eoff + 8 <= end; i++, eoff += 8)
            {
                var e = new VDElem
                {
                    VertexType = d[eoff + 0],
                    NumComponents = d[eoff + 1],
                    Stream = d[eoff + 2],
                    Offset = d[eoff + 3],
                    Stride = BE.U16(d, eoff + 4),
                    Type = d[eoff + 6],
                    Class = d[eoff + 7]
                };
                elems.Add(e);
            }
            if (elems.Count == 0) throw new InvalidDataException("VDES has no elements.");

            if (VerboseVDES)
            {
                LOG?.Invoke($"[VDES] elements ({elems.Count})");
                LOG?.Invoke("        idx | vType                          | comps | stream | off | stride | type(name)                 | class       ");
                for (int i = 0; i < elems.Count; i++)
                {
                    var e = elems[i];
                    string etn = (e.Type < ElementTypeName.Length) ? ElementTypeName[e.Type] : $"Type{e.Type}";
                    LOG?.Invoke($"        {i,3} | {VertexTypeName(e.VertexType),-30} | {e.NumComponents,5} | {e.Stream,6} | {e.Offset,3} | {e.Stride,6} | 0x{e.Type:X2}({etn,-24}) | {ElementClassName(e.Class),-11}");
                }
            }

            var strideModes = elems.Select(e => (int)e.Stride).Where(s => s > 0).Distinct().ToArray();
            if (strideModes.Length > 1) LOG?.Invoke($"[VDES] [warn] Multiple stride values present: {string.Join("/", strideModes)}");
            int resolvedStride = strideModes.Length > 0 ? strideModes.OrderByDescending(x => x).First() : 0;

            // Cross-check flags vs elements
            if (flags != 0)
            {
                uint maskFromElems = 0;
                foreach (var e in elems) maskFromElems |= (1u << (e.Type & 31));
                if (maskFromElems != flags) LOG?.Invoke($"[VDES] [warn] typesFlags mismatch: header=0x{flags:X8}  fromElems=0x{maskFromElems:X8}");
                else LOG?.Invoke($"[VDES] typesFlags matches elements (0x{flags:X8}).");
                if (VerboseVDES)
                {
                    var setBits = Enumerable.Range(0, 32).Where(b => ((flags >> b) & 1) != 0).ToArray();
                    LOG?.Invoke($"[VDES] typesFlags bits set: {string.Join(", ", setBits.Select(b => $"{(b < ElementTypeName.Length ? ElementTypeName[b] : $"Type{b}")}"))}");
                }
            }

            // Frequency divider operation bits (bit per element): 1 = MODULO, 0 = DIVIDE
            if (freq != 0)
            {
                var bits = new List<string>();
                for (int i = 0; i < elems.Count && i < 16; i++) if (((freq >> i) & 1) != 0) bits.Add($"elem{i}");
                LOG?.Invoke($"[VDES] freqMask=0x{freq:X4}  setBits=[{string.Join(", ", bits)}] (1=MODULO,0=DIVIDE)");
            }

            var vd = new VDInfo { Stride = resolvedStride, RefCount = refCount, TypesFlags = flags, NumStreams = nStream, NumElements = nElem, FreqMask = freq, Pad0 = pad0, RawElems = elems };

            // Pick primary elements by ElementType
            foreach (var e in elems)
            {
                switch (e.Type)
                {
                    case 0: if (vd.PosElem == null) vd.PosElem = e; break; // XYZ
                    case 2: if (vd.NormalElem == null) vd.NormalElem = e; break; // NORMAL
                    case 14: if (vd.TangentElem == null) vd.TangentElem = e; break; // TANGENT
                    case 15: if (vd.BinormalElem == null) vd.BinormalElem = e; break; // BINORMAL
                    case 8:
                    case 9:
                    case 10:
                    case 11:
                    case 12:
                    case 13:
                        int ti = e.Type - 8; if (!vd.TexElems.ContainsKey(ti)) vd.TexElems[ti] = e; break; // TEX0..TEX5
                    default:
                        break; // ignore VERTEXCOLOR/SPECULAR/INDICES here
                }
            }

            // Log resolution
            void LogElem(string label, VDElem? ve)
            {
                if (ve == null) { LOG?.Invoke($"[VDES] {label}: (none)"); return; }
                var e = ve.Value; string etn = (e.Type < ElementTypeName.Length) ? ElementTypeName[e.Type] : $"Type{e.Type}";
                LOG?.Invoke($"[VDES] {label}: off={e.Offset} vt={VertexTypeName(e.VertexType)} comps={e.NumComponents} type={etn} class={ElementClassName(e.Class)}");
            }
            LogElem("XYZ", vd.PosElem);
            LogElem("NRM", vd.NormalElem);
            LogElem("TAN", vd.TangentElem);
            LogElem("BIN", vd.BinormalElem);
            if (vd.TexElems.TryGetValue(0, out var te0)) LOG?.Invoke($"[VDES] TEX0: off={te0.Offset} vt={VertexTypeName(te0.VertexType)} comps={te0.NumComponents}");
            else LOG?.Invoke("[VDES] TEX0: (none)");

            // ---- Correct element size check (per PS3 vertex types) ----
            foreach (var e in elems)
            {
                int expectBytes = ElemByteSize(e.VertexType, e.NumComponents);
                if (vd.Stride > 0 && e.Offset + expectBytes > vd.Stride)
                    LOG?.Invoke($"[VDES] [warn] elem offset+size ({e.Offset}+{expectBytes}) exceeds stride {vd.Stride} (type={e.Type})");
            }

            return vd;
        }

        // PS3 vertex-type aware element byte size
        static int ElemByteSize(byte vertexType, byte numComponents)
        {
            switch (vertexType)
            {
                case 0x01: // CELL_GCM_VERTEX_S1 (S16 normalized in docs; assets often use raw)
                case 0x05: // CELL_GCM_VERTEX_S32K (S16 non-norm)
                    return 2 * Math.Max(1, (int)numComponents);
                case 0x02: // F32
                    return 4 * Math.Max(1, (int)numComponents);
                case 0x03: // F16 (half)
                    return 2 * Math.Max(1, (int)numComponents);
                case 0x04: // U8 normalized
                case 0x07: // U8 non-normalized
                    return 1 * Math.Max(1, (int)numComponents);
                case 0x06: //CELL_GCM_VERTEX_CMP (DEC3N 10:10:10:2),
                    return 4;
                default:
                    return 2 * Math.Max(1, (int)numComponents);
            }
        }

        // Pair VB-only submeshes to the nearest compatible base (same vCount & stride), attach as morph targets.
        static void AttachMorphTargets(SceneData scene)
        {
            var bases = scene.SubMeshes.Where(s => s.Triangles.Count > 0).OrderBy(s => s.VbIndex).ToList();
            var orphans = scene.SubMeshes.Where(s => s.Triangles.Count == 0 && s.Positions.Count > 0).OrderBy(s => s.VbIndex).ToList();
            if (orphans.Count == 0 || bases.Count == 0) return;

            var toRemove = new HashSet<SubMeshData>();

            foreach (var m in orphans)
            {
                // Closest-in-file base
                var baseSm = bases
                    .OrderBy(b => Math.Abs(b.VbIndex - m.VbIndex))
                    .ThenBy(b => Math.Abs(b.VdIndex - m.VdIndex))
                    .FirstOrDefault();

                if (baseSm == null) continue;

                if (baseSm.Positions.Count == m.Positions.Count)
                {
                    var mt = new MorphTarget { Name = $"{m.Name} (VB{m.VbIndex}→Base VB{baseSm.VbIndex})", VbIndex = m.VbIndex, VdIndex = m.VdIndex };
                    mt.Positions.AddRange(m.Positions);
                    baseSm.Morphs.Add(mt);
                    scene.Log($"[MorphPairing] Direct attach {m.Name} → {baseSm.Name} (V={m.Positions.Count})");
                    toRemove.Add(m);
                    continue;
                }

                // Counts differ: try quantized position mapping (orphan is often the welded set)
                var expanded = TryMapAndExpand(baseSm, m, out int hits, out float cov);
                if (expanded != null)
                {
                    var mt = new MorphTarget { Name = $"{m.Name} (mapped→Base VB{baseSm.VbIndex})", VbIndex = m.VbIndex, VdIndex = m.VdIndex };
                    mt.Positions.AddRange(expanded);
                    baseSm.Morphs.Add(mt);
                    scene.Log($"[MorphPairing] Mapped {m.Name} → {baseSm.Name}  hits={hits}/{baseSm.Positions.Count} ({cov:P0})");
                    toRemove.Add(m);
                }
                else
                {
                    scene.Log($"[MorphPairing] No base match for orphan {m.Name} (V={m.Positions.Count}, stride={m.VertexStride}) — leaving as drawable.");
                }
            }

            if (toRemove.Count > 0)
            {
                scene.SubMeshes.RemoveAll(s => toRemove.Contains(s));
                scene.Log($"[MorphPairing] Removed {toRemove.Count} VB-only streams from draw list (now {scene.SubMeshes.Count} drawable submeshes).");
            }
        }

        // Quantize positions, build orphan→base map, then expand orphan positions to base size
        static List<Vec3> TryMapAndExpand(SubMeshData baseSm, SubMeshData orphan, out int hits, out float coverage)
        {
            int[] qs = { 64, 128, 256, 512, 1024, 2048, 4096 };
            foreach (int q in qs)
            {
                var map = BuildQuantizedMap(baseSm.Positions, orphan.Positions, q, out hits);
                coverage = (float)hits / Math.Max(1, baseSm.Positions.Count);
                if (coverage >= 0.90f)
                {
                    var expanded = new List<Vec3>(baseSm.Positions.Count);
                    for (int i = 0; i < baseSm.Positions.Count; i++)
                    {
                        int oi = map[i];
                        expanded.Add(oi >= 0 ? orphan.Positions[oi] : baseSm.Positions[i]);
                    }
                    return expanded;
                }
            }
            hits = 0; coverage = 0f; return null;
        }

        static List<int> BuildQuantizedMap(IReadOnlyList<Vec3> basePos, IReadOnlyList<Vec3> orphanPos, int q, out int hits)
        {
            var dict = new Dictionary<(int, int, int), int>(orphanPos.Count);
            for (int i = 0; i < orphanPos.Count; i++)
            {
                var k = Key(orphanPos[i], q);
                if (!dict.ContainsKey(k)) dict[k] = i; // keep first index for that welded position
            }

            hits = 0;
            var map = new List<int>(basePos.Count);
            for (int j = 0; j < basePos.Count; j++)
            {
                var k = Key(basePos[j], q);
                if (dict.TryGetValue(k, out int oi)) { map.Add(oi); hits++; }
                else map.Add(-1);
            }
            return map;
        }

        static (int, int, int) Key(in Vec3 p, int q)
        {
            int x = (int)Math.Round(p.X * q);
            int y = (int)Math.Round(p.Y * q);
            int z = (int)Math.Round(p.Z * q);
            return (x, y, z);
        }
    }

    // ======== Preview renderer (morph-aware) ========

    static class Preview
    {
        public static Bitmap Render3D(
            SceneData scene, IReadOnlyList<SubMeshData> drawSubs,
            int width, int height,
            float yawDeg, float pitchDeg,
            float zoom, float panPxX, float panPxY,
            bool showFaces, bool showNormals, bool showTangents, bool showBinormals,
            bool swapYZ,
            bool fast,
            SubMeshData morphSm = null, IReadOnlyList<Vec3> morphPositions = null, float morphWeight = 0f)
        {
            var bmp = new Bitmap(Math.Max(1, width), Math.Max(1, height));
            if (scene == null || drawSubs == null || drawSubs.Count == 0) return bmp;

            using var g = Graphics.FromImage(bmp);
            g.Clear(Color.White);
            g.SmoothingMode = fast ? SmoothingMode.None : SmoothingMode.AntiAlias;
            g.PixelOffsetMode = PixelOffsetMode.Half;

            float yaw = yawDeg * (float)Math.PI / 180f;
            float pitch = pitchDeg * (float)Math.PI / 180f;
            float cy = (float)Math.Cos(yaw), sy = (float)Math.Sin(yaw);
            float cp = (float)Math.Cos(pitch), sp = (float)Math.Sin(pitch);

            Vec3 Rotate(Vec3 v)
            {
                if (swapYZ) v = Vec3.SwapYZ(v);
                float x1 = v.X * cy + v.Z * sy;
                float y1 = v.Y;
                float z1 = -v.X * sy + v.Z * cy;
                float x2 = x1;
                float y2 = y1 * cp - z1 * sp;
                float z2 = y1 * sp + z1 * cp;
                return new Vec3(x2, y2, z2);
            }
            Vec3 RotateN(Vec3 n) { var v = Rotate(n); return Vec3.Normalize(v); }

            var worldPos = new List<Vec3>();
            var worldNor = new List<Vec3>();
            var worldTan = new List<Vec3>();
            var worldBin = new List<Vec3>();

            Vec3 min = new Vec3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vec3 max = new Vec3(float.MinValue, float.MinValue, float.MinValue);

            foreach (var sm in drawSubs)
            {
                bool applyMorph = (morphSm != null && morphPositions != null && ReferenceEquals(sm, morphSm) && morphPositions.Count == sm.Positions.Count && morphWeight > 1e-6f);
                List<Vec3> tempNormals = null;
                if (applyMorph && sm.Triangles.Count > 0)
                {
                    var morphed = new Vec3[sm.Positions.Count];
                    for (int i = 0; i < sm.Positions.Count; i++) morphed[i] = Vec3.Lerp(sm.Positions[i], morphPositions[i], morphWeight);
                    tempNormals = PsgParser.BuildNormals(sm, morphed);
                }

                for (int i = 0; i < sm.Positions.Count; i++)
                {
                    var srcP = sm.Positions[i]; if (applyMorph) srcP = Vec3.Lerp(srcP, morphPositions[i], morphWeight);
                    var wp = Rotate(srcP);

                    Vec3 nSrc = (i < sm.Normals.Count) ? sm.Normals[i] : new Vec3(0, 0, 1);
                    if (applyMorph && tempNormals != null) nSrc = tempNormals[i];
                    var wn = RotateN(nSrc);

                    var wt = (i < sm.Tangents.Count) ? RotateN(sm.Tangents[i]) : new Vec3(1, 0, 0);
                    var wb = (i < sm.Binormals.Count) ? RotateN(sm.Binormals[i]) : new Vec3(0, 1, 0);

                    worldPos.Add(wp); worldNor.Add(wn); worldTan.Add(wt); worldBin.Add(wb);

                    min.X = Math.Min(min.X, wp.X); min.Y = Math.Min(min.Y, wp.Y); min.Z = Math.Min(min.Z, wp.Z);
                    max.X = Math.Max(max.X, wp.X); max.Y = Math.Max(max.Y, wp.Y); max.Z = Math.Max(max.Z, wp.Z);
                }
            }

            float sxw = Math.Max(1e-6f, max.X - min.X);
            float syw = Math.Max(1e-6f, max.Y - min.Y);
            float baseScale = 0.9f * Math.Min(width / sxw, height / syw);
            float scale = baseScale * Math.Max(0.05f, zoom);
            float cxm = (min.X + max.X) * 0.5f; float cym = (min.Y + max.Y) * 0.5f;
            float ox = width * 0.5f + panPxX; float oy = height * 0.5f + panPxY;
            PointF Map(Vec3 P) => new PointF(ox + (P.X - cxm) * scale, oy - (P.Y - cym) * scale);

            var screen = worldPos.Select(Map).ToArray();

            if (showFaces)
            {
                var lightDir = Vec3.Normalize(new Vec3(0.5f, 0.9f, 0.4f));
                var draw = new List<(PointF p0, PointF p1, PointF p2, float z, float lambert)>(4096);
                int smBase = 0;
                foreach (var sm in drawSubs)
                {
                    foreach (var t in sm.Triangles)
                    {
                        int a = smBase + t.A, b = smBase + t.B, c = smBase + t.C;
                        if ((uint)a >= (uint)screen.Length || (uint)b >= (uint)screen.Length || (uint)c >= (uint)screen.Length) continue;
                        float za = worldPos[a].Z, zb = worldPos[b].Z, zc = worldPos[c].Z;
                        float la = Math.Max(0, Vec3.Dot(worldNor[a], lightDir));
                        float lb = Math.Max(0, Vec3.Dot(worldNor[b], lightDir));
                        float lc = Math.Max(0, Vec3.Dot(worldNor[c], lightDir));
                        float lam = (la + lb + lc) / 3f;
                        draw.Add((screen[a], screen[b], screen[c], (za + zb + zc) / 3f, lam));
                    }
                    smBase += sm.Positions.Count;
                }
                draw.Sort((u, v) => u.z.CompareTo(v.z));
                foreach (var d in draw)
                {
                    float ambient = 0.25f, diffuse = 0.85f * d.lambert; float shade = Math.Max(0, Math.Min(1, ambient + diffuse));
                    int r = (int)(shade * 204), gcol = (int)(shade * 210), bcol = (int)(shade * 220);
                    using var brush = new SolidBrush(Color.FromArgb(r, gcol, bcol));
                    g.FillPolygon(brush, new[] { d.p0, d.p1, d.p2 });
                }
            }
            else
            {
                using var vb = new SolidBrush(Color.Black);
                foreach (var p in screen) g.FillEllipse(vb, p.X - 1, p.Y - 1, 2, 2);
            }

            float bboxDiag = (max - min).Length(); float L = 0.03f * bboxDiag;
            using var pN = new Pen(Color.Blue, 1f);
            using var pT = new Pen(Color.Red, 1f);
            using var pB = new Pen(Color.Green, 1f);
            int sampleEvery = Math.Max(1, screen.Length / 2000);
            for (int i = 0; i < screen.Length; i += sampleEvery)
            {
                var p0 = screen[i];
                if (showNormals) g.DrawLine(pN, p0, Map(worldPos[i] + worldNor[i] * L));
                if (showTangents) g.DrawLine(pT, p0, Map(worldPos[i] + worldTan[i] * L));
                if (showBinormals) g.DrawLine(pB, p0, Map(worldPos[i] + worldBin[i] * L));
            }

            if (!fast)
            {
                using var font = new Font("Segoe UI", 8f);
                using var sb = new SolidBrush(Color.FromArgb(70, 0, 0, 0));
                int vTotal = drawSubs.Sum(s => s.Positions.Count);
                int tTotal = drawSubs.Sum(s => s.Triangles.Count);
                string hud = $"Submeshes:{drawSubs.Count}  V:{vTotal}  T:{tTotal}  |  yaw:{yawDeg:0.#}  pitch:{pitchDeg:0.#}  zoom:{zoom:0.###}";
                if (morphSm != null && morphPositions != null && morphWeight > 1e-6f) hud += $"  |  Morph: {morphSm.Name} w={morphWeight:0.00}";
                g.DrawString(hud, font, sb, 6, 6);
            }
            return bmp;
        }

        public static Bitmap RenderUV(SubMeshData sm, bool normalizeUVs, bool flipV, int width, int height)
        {
            var bmp = new Bitmap(Math.Max(32, width), Math.Max(32, height));
            if (sm == null || sm.UVs.Count == 0) return bmp;

            float div = normalizeUVs ? Math.Max(1, sm.UvScaleDivisor) : 1f;

            float minU = float.MaxValue, minV = float.MaxValue; float maxU = float.MinValue, maxV = float.MinValue;
            for (int i = 0; i < sm.UVs.Count; i++)
            {
                float u = sm.UVs[i].X / div; float v = sm.UVs[i].Y / div; if (flipV) v = 1f - v;
                minU = Math.Min(minU, u); maxU = Math.Max(maxU, u); minV = Math.Min(minV, v); maxV = Math.Max(maxV, v);
            }
            float du = maxU - minU; if (du <= 1e-6f) { minU -= 0.5f; maxU += 0.5f; du = maxU - minU; }
            float dv = maxV - minV; if (dv <= 1e-6f) { minV -= 0.5f; maxV += 0.5f; dv = maxV - minV; }

            float scale = 0.90f * Math.Min(width / du, height / dv);
            float cx = (minU + maxU) * 0.5f; float cy = (minV + maxV) * 0.5f;

            PointF Map(Vec2 uv)
            {
                float u = uv.X / div; float v = uv.Y / div; if (flipV) v = 1f - v;
                return new PointF(width * 0.5f + (u - cx) * scale, height * 0.5f - (v - cy) * scale);
            }

            using var g = Graphics.FromImage(bmp);
            g.SmoothingMode = SmoothingMode.AntiAlias; g.Clear(Color.White);
            using var framePen = new Pen(Color.FromArgb(200, 0, 0, 0), 1f); g.DrawRectangle(framePen, 0, 0, width - 1, height - 1);

            if (sm.Triangles.Count > 0 && sm.UVs.Count == sm.Positions.Count)
            {
                using var fill = new SolidBrush(Color.FromArgb(24, 0, 0, 0));
                using var triPen = new Pen(Color.FromArgb(140, 0, 0, 0), 1f);
                foreach (var t in sm.Triangles)
                {
                    var p0 = Map(sm.UVs[t.A]); var p1 = Map(sm.UVs[t.B]); var p2 = Map(sm.UVs[t.C]);
                    g.FillPolygon(fill, new[] { p0, p1, p2 }); g.DrawPolygon(triPen, new[] { p0, p1, p2 });
                }
            }
            else
            {
                using var b = new SolidBrush(Color.Black);
                foreach (var uv in sm.UVs) { var p = Map(uv); g.FillEllipse(b, p.X - 1, p.Y - 1, 2, 2); }
            }

            using var font = new Font("Segoe UI", 8f);
            using var sb = new SolidBrush(Color.FromArgb(60, 0, 0, 0));
            string hud = $"UV:{sm.UVs.Count}  Scale:1/{(normalizeUVs ? sm.UvScaleDivisor : 1)}";
            g.DrawString(hud, font, sb, 6, 6);
            return bmp;
        }
    }

    // ======== Main Form (UI) ========

    public sealed class MainForm : Form
    {
        // Toolbar & status
        private ToolStrip tool;
        private ToolStripButton tbtnOpen, tbtnResetCam, tbtnFit, tbtnCopyLog;
        private ToolStripLabel lblPick;
        private ToolStripComboBox cmbSubmesh;
        private StatusStrip status; private ToolStripStatusLabel lblCounts, lblCamera, lblSpacer;
        private ToolStripButton tbtnExportObj;

        // Splitters
        private SplitContainer splitMain; private SplitContainer splitLeftRight; private SplitContainer splitPreviews;

        // Options
        private Panel leftPanel; private GroupBox grpOptions;
        private CheckBox chkShowFaces, chkShowNormals, chkShowTangents, chkShowBinormals;
        private CheckBox chkSwapYZ, chkNormalizeUVs, chkFlipV;

        // Morphs UI
        private GroupBox grpMorph; private Label lblMorphTarget, lblMorphWeight; private ComboBox cmbMorphTargets; private TrackBar trkMorphWeight;

        // Viewers
        private Panel viewHost3D, viewHostUV; private Label lbl3D, lblUV; private PictureBox pic3D, picUV;

        // Log
        private TextBox txtLog;

        // Data
        private byte[] currentBytes; private string currentFilePath; private SceneData scene;

        // Camera
        private bool isRotating = false, isPanning = false; private Point lastMouse; private float yawDeg = 35f, pitchDeg = 20f, zoom = 1.0f, panPxX = 0f, panPxY = 0f;

        // Throttle
        private long lastDragTick; private const int DragFrameMs = 16;

        public MainForm()
        {
            Text = "PSG Viewer (Skate 3 • PS3) — Ground-Truth Vertex Flags + Morphs";
            StartPosition = FormStartPosition.CenterScreen; MinimumSize = new Size(1200, 760); Size = new Size(1500, 980);
            BuildUi(); WireEvents(); Shown += (s, e) => InitSplittersSafe(); ResizeEnd += (s, e) => InitSplittersSafe();
        }

        private void BuildUi()
        {
            // Toolbar (create the ToolStrip first)
            tool = new ToolStrip { GripStyle = ToolStripGripStyle.Hidden, Stretch = true };

            tbtnOpen = new ToolStripButton("Open PSG") { DisplayStyle = ToolStripItemDisplayStyle.Text };
            tbtnResetCam = new ToolStripButton("Reset Camera") { DisplayStyle = ToolStripItemDisplayStyle.Text };
            tbtnFit = new ToolStripButton("Fit View") { DisplayStyle = ToolStripItemDisplayStyle.Text };
            tbtnCopyLog = new ToolStripButton("Copy Log") { DisplayStyle = ToolStripItemDisplayStyle.Text, ToolTipText = "Copy full log to clipboard" };
            lblPick = new ToolStripLabel("  Submesh: ");
            cmbSubmesh = new ToolStripComboBox { DropDownStyle = ComboBoxStyle.DropDownList, AutoSize = false, Width = 320 };

            // Export OBJ button as a FIELD (so WireEvents can hook it later)
            tbtnExportObj = new ToolStripButton("Export OBJ") { DisplayStyle = ToolStripItemDisplayStyle.Text };

            tool.Items.AddRange(new ToolStripItem[]
            {
        tbtnOpen, new ToolStripSeparator(),
        tbtnResetCam, tbtnFit, new ToolStripSeparator(),
        lblPick, cmbSubmesh, new ToolStripSeparator(),
        tbtnCopyLog, new ToolStripSeparator(),
        tbtnExportObj
            });

            Controls.Add(tool);
            tool.Dock = DockStyle.Top;

            // Status
            status = new StatusStrip();
            lblCounts = new ToolStripStatusLabel("Sub:0  V:0  T:0");
            lblSpacer = new ToolStripStatusLabel { Spring = true };
            lblCamera = new ToolStripStatusLabel("Yaw:0  Pitch:0  Zoom:1.0");
            status.Items.AddRange(new ToolStripItem[] { lblCounts, lblSpacer, lblCamera });
            Controls.Add(status);
            status.Dock = DockStyle.Bottom;

            // Main split
            splitMain = new SplitContainer { Dock = DockStyle.Fill, Orientation = Orientation.Horizontal, SplitterWidth = 6 };
            Controls.Add(splitMain);

            // Left-right split
            splitLeftRight = new SplitContainer { Dock = DockStyle.Fill, Orientation = Orientation.Vertical, FixedPanel = FixedPanel.Panel1, SplitterWidth = 6 };
            splitMain.Panel1.Controls.Add(splitLeftRight);

            // Options column
            leftPanel = new Panel { Dock = DockStyle.Fill, AutoScroll = true, Padding = new Padding(18, 30, 14, 16) };
            splitLeftRight.Panel1.Controls.Add(leftPanel);

            grpOptions = new GroupBox { Text = "Draw Options", Dock = DockStyle.Top, Height = 220, Padding = new Padding(10, 10, 10, 8) };
            leftPanel.Controls.Add(grpOptions);
            int y = 18, x = 18, w = 260;
            chkShowFaces = new CheckBox { Left = x, Top = y, Width = w, Text = "Show Faces (Lambert)", Checked = true }; y += 24;
            chkShowNormals = new CheckBox { Left = x, Top = y, Width = w, Text = "Show Normals (blue)", Checked = true }; y += 24;
            chkShowTangents = new CheckBox { Left = x, Top = y, Width = w, Text = "Show Tangents (red)", Checked = true }; y += 24;
            chkShowBinormals = new CheckBox { Left = x, Top = y, Width = w, Text = "Show Binormals (green)", Checked = true }; y += 24;
            chkSwapYZ = new CheckBox { Left = x, Top = y, Width = w, Text = "Swap Y/Z (Z-up → Y-up)" }; y += 24;
            chkNormalizeUVs = new CheckBox { Left = x, Top = y, Width = w, Text = "Normalize UVs (auto)", Checked = true }; y += 24;
            chkFlipV = new CheckBox { Left = x, Top = y, Width = w, Text = "Flip V (v = 1 - v)" }; y += 24;
            grpOptions.Controls.AddRange(new Control[] { chkShowFaces, chkShowNormals, chkShowTangents, chkShowBinormals, chkSwapYZ, chkNormalizeUVs, chkFlipV });

            // Morph Preview group
            grpMorph = new GroupBox { Text = "Morph Preview (for selected base submesh)", Dock = DockStyle.Top, Height = 130, Padding = new Padding(10, 10, 10, 10) };
            leftPanel.Controls.Add(grpMorph);
            lblMorphTarget = new Label { Left = x, Top = 24, Width = 60, Text = "Target:" };
            cmbMorphTargets = new ComboBox { Left = x + 62, Top = 20, Width = 260, DropDownStyle = ComboBoxStyle.DropDownList };
            lblMorphWeight = new Label { Left = x, Top = 58, Width = 120, Text = "Weight: 0%" };
            trkMorphWeight = new TrackBar { Left = x, Top = 80, Width = 320, Minimum = 0, Maximum = 100, TickFrequency = 10, Value = 0 };
            grpMorph.Controls.AddRange(new Control[] { lblMorphTarget, cmbMorphTargets, lblMorphWeight, trkMorphWeight });

            // Previews split
            splitPreviews = new SplitContainer { Dock = DockStyle.Fill, Orientation = Orientation.Vertical, SplitterWidth = 6 };
            splitLeftRight.Panel2.Controls.Add(splitPreviews);

            // 3D host
            viewHost3D = new Panel { Dock = DockStyle.Fill, Padding = new Padding(8) };
            lbl3D = new Label { Dock = DockStyle.Top, Height = 22, Text = "3D (LMB rotate, RMB pan, Wheel zoom, Double-click reset)", Font = new Font("Segoe UI", 9, FontStyle.Bold) };
            pic3D = new PictureBox { Dock = DockStyle.None, BorderStyle = BorderStyle.FixedSingle, BackColor = Color.White, TabStop = true };
            viewHost3D.Controls.Add(pic3D);
            viewHost3D.Controls.Add(lbl3D);
            splitPreviews.Panel1.Controls.Add(viewHost3D);

            // UV host
            viewHostUV = new Panel { Dock = DockStyle.Fill, Padding = new Padding(8) };
            lblUV = new Label { Dock = DockStyle.Top, Height = 22, Text = "UV Viewer (showing TEX0 of selected submesh)", Font = new Font("Segoe UI", 9, FontStyle.Bold) };
            picUV = new PictureBox { Dock = DockStyle.None, BorderStyle = BorderStyle.FixedSingle, BackColor = Color.White };
            viewHostUV.Controls.Add(picUV);
            viewHostUV.Controls.Add(lblUV);
            splitPreviews.Panel2.Controls.Add(viewHostUV);

            // Log
            txtLog = new TextBox { Dock = DockStyle.Fill, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Both, WordWrap = false, Font = new Font("Consolas", 9f) };
            splitMain.Panel2.Controls.Add(txtLog);

            // Initial square layout
            LayoutViewerSquare(viewHost3D, lbl3D, pic3D);
            LayoutViewerSquare(viewHostUV, lblUV, picUV);
            cmbSubmesh.Items.Add("(All submeshes)");
            cmbSubmesh.SelectedIndex = 0;
            EnableMorphUI(false);
        }


        private void WireEvents()
        {
            tbtnOpen.Click += (s, e) => OpenPsg();
            tbtnResetCam.Click += (s, e) => { ResetCamera(); RenderPreviews(); UpdateStatus(); };
            tbtnFit.Click += (s, e) => { FitViewToContent(); RenderPreviews(); UpdateStatus(); };
            tbtnCopyLog.Click += (s, e) => { try { Clipboard.SetText(txtLog.Text); } catch { } };
            cmbSubmesh.SelectedIndexChanged += (s, e) => { RefreshMorphUiForSelection(); RenderPreviews(); UpdateStatus(); };

            chkShowFaces.CheckedChanged += (s, e) => RenderPreviews();
            chkShowNormals.CheckedChanged += (s, e) => RenderPreviews();
            chkShowTangents.CheckedChanged += (s, e) => RenderPreviews();
            chkShowBinormals.CheckedChanged += (s, e) => RenderPreviews();
            chkSwapYZ.CheckedChanged += (s, e) => RenderPreviews();
            chkNormalizeUVs.CheckedChanged += (s, e) => RenderPreviews();
            chkFlipV.CheckedChanged += (s, e) => RenderPreviews();

            cmbMorphTargets.SelectedIndexChanged += (s, e) => RenderPreviews();
            trkMorphWeight.ValueChanged += (s, e) => { lblMorphWeight.Text = $"Weight: {trkMorphWeight.Value}%"; RenderPreviews(); };

            splitMain.SplitterMoved += (s, e) => { LayoutViewerSquare(viewHost3D, lbl3D, pic3D); LayoutViewerSquare(viewHostUV, lblUV, picUV); };
            splitLeftRight.SplitterMoved += (s, e) => { LayoutViewerSquare(viewHost3D, lbl3D, pic3D); LayoutViewerSquare(viewHostUV, lblUV, picUV); };
            splitPreviews.SplitterMoved += (s, e) => { LayoutViewerSquare(viewHost3D, lbl3D, pic3D); LayoutViewerSquare(viewHostUV, lblUV, picUV); };
            viewHost3D.Resize += (s, e) => LayoutViewerSquare(viewHost3D, lbl3D, pic3D);
            viewHostUV.Resize += (s, e) => LayoutViewerSquare(viewHostUV, lblUV, picUV);

            pic3D.Resize += (s, e) => { RenderPreviews(); UpdateStatus(); };
            picUV.Resize += (s, e) => { RenderPreviews(); UpdateStatus(); };

            // 3D interactions
            pic3D.MouseEnter += (s, e) => pic3D.Focus();
            pic3D.MouseDown += Pic3D_MouseDown;
            pic3D.MouseMove += Pic3D_MouseMove;
            pic3D.MouseUp += Pic3D_MouseUp;
            pic3D.MouseLeave += (s, e) => { isRotating = false; isPanning = false; };
            pic3D.MouseWheel += Pic3D_MouseWheel;
            pic3D.DoubleClick += (s, e) => { ResetCamera(); RenderPreviews(); UpdateStatus(); };

            // Export OBJ (hook the FIELD created in BuildUi)
            if (tbtnExportObj != null)
            {
                tbtnExportObj.Click += (s, e) =>
                {
                    if (scene == null || scene.SubMeshes == null || scene.SubMeshes.Count == 0)
                    {
                        Log("Nothing to export.");
                        return;
                    }

                    using var sfd = new SaveFileDialog
                    {
                        Title = "Export OBJ",
                        Filter = "OBJ (*.obj)|*.obj",
                        FileName = ((currentFilePath != null ? Path.GetFileNameWithoutExtension(currentFilePath) : "mesh") + ".obj")
                    };

                    if (sfd.ShowDialog(this) != DialogResult.OK) return;

                    try
                    {
                        ObjExporter.ExportSceneToObj(scene, sfd.FileName, bakeYupToZup: true, flipV: false, includeNormals: true);
                        Log($"Exported OBJ: {sfd.FileName}");
                    }
                    catch (Exception ex)
                    {
                        AppendException("Export OBJ", ex);
                    }
                };
            }
        }


        private void OpenPsg()
        {
            using var ofd = new OpenFileDialog { Title = "Open PSG", Filter = "PSG files (*.psg)|*.psg|All files (*.*)|*.*" };
            if (ofd.ShowDialog(this) != DialogResult.OK) return;
            try
            {
                currentBytes = File.ReadAllBytes(ofd.FileName); currentFilePath = ofd.FileName; txtLog.Clear(); Log($"Loaded PSG: {ofd.FileName} ({currentBytes.Length:N0} bytes)");
                scene = PsgParser.Parse(currentBytes, Log);

                cmbSubmesh.Items.Clear(); cmbSubmesh.Items.Add("(All submeshes)"); foreach (var sm in scene.SubMeshes) cmbSubmesh.Items.Add(sm.Name); cmbSubmesh.SelectedIndex = 0;

                Log(""); Log("=== SUBMESHES ===");
                for (int i = 0; i < scene.SubMeshes.Count; i++)
                {
                    var sm = scene.SubMeshes[i];
                    Log($"[{i:00}] {sm.Name} : V={sm.Positions.Count}  T={sm.Triangles.Count}  stride={sm.VertexStride}  morphs={sm.Morphs.Count}");
                }

                ResetCamera(); RefreshMorphUiForSelection(); RenderPreviews(); UpdateStatus();
            }
            catch (Exception ex) { AppendException("OpenPsg", ex); }
        }

        private IReadOnlyList<SubMeshData> SelectedSubmeshes()
        {
            if (scene == null || scene.SubMeshes.Count == 0) return Array.Empty<SubMeshData>();
            if (cmbSubmesh.SelectedIndex <= 0) return scene.SubMeshes;
            int idx = cmbSubmesh.SelectedIndex - 1; if (idx < 0 || idx >= scene.SubMeshes.Count) return scene.SubMeshes; return new[] { scene.SubMeshes[idx] };
        }

        private void RenderPreviews()
        {
            var subs = SelectedSubmeshes();
            if (subs == null || subs.Count == 0) { pic3D.Image?.Dispose(); pic3D.Image = null; picUV.Image?.Dispose(); picUV.Image = null; return; }

            bool fast = isRotating || isPanning;
            SubMeshData morphSm = null; IReadOnlyList<Vec3> morphPos = null; float morphW = 0f;
            if (subs.Count == 1 && cmbMorphTargets.Enabled && cmbMorphTargets.SelectedIndex >= 0 && trkMorphWeight.Value > 0)
            {
                var smSel = subs[0]; if (smSel.Morphs.Count > 0) { var mt = smSel.Morphs[cmbMorphTargets.SelectedIndex]; morphSm = smSel; morphPos = mt.Positions; morphW = trkMorphWeight.Value / 100f; }
            }

            var bmp3D = Preview.Render3D(scene, subs, Math.Max(1, pic3D.Width), Math.Max(1, pic3D.Height), yawDeg, pitchDeg, zoom, panPxX, panPxY,
                chkShowFaces.Checked, chkShowNormals.Checked, chkShowTangents.Checked, chkShowBinormals.Checked, chkSwapYZ.Checked, fast, morphSm, morphPos, morphW);

            var uvSm = subs[0]; var bmpUV = Preview.RenderUV(uvSm, chkNormalizeUVs.Checked, chkFlipV.Checked, Math.Max(1, picUV.Width), Math.Max(1, picUV.Height));
            pic3D.Image?.Dispose(); picUV.Image?.Dispose(); pic3D.Image = bmp3D; picUV.Image = bmpUV;
        }

        private void RefreshMorphUiForSelection()
        {
            var subs = SelectedSubmeshes();
            if (subs.Count != 1) { EnableMorphUI(false); return; }
            var sm = subs[0]; if (sm.Morphs.Count == 0) { EnableMorphUI(false); return; }

            cmbMorphTargets.Items.Clear(); foreach (var m in sm.Morphs) cmbMorphTargets.Items.Add(m.Name); if (cmbMorphTargets.Items.Count > 0) cmbMorphTargets.SelectedIndex = 0;
            trkMorphWeight.Value = 0; lblMorphWeight.Text = "Weight: 0%"; EnableMorphUI(true);
        }

        private void EnableMorphUI(bool enable)
        {
            grpMorph.Enabled = enable; cmbMorphTargets.Enabled = enable; trkMorphWeight.Enabled = enable; lblMorphTarget.Enabled = enable; lblMorphWeight.Enabled = enable;
        }

        private void UpdateStatus()
        {
            int sub = scene?.SubMeshes.Count ?? 0; int v = SelectedSubmeshes().Sum(s => s.Positions.Count); int t = SelectedSubmeshes().Sum(s => s.Triangles.Count);
            lblCounts.Text = $"Sub:{sub}  V:{v}  T:{t}"; lblCamera.Text = $"Yaw:{yawDeg:0.#}  Pitch:{pitchDeg:0.#}  Zoom:{zoom:0.###}";
        }

        private void ResetCamera() { yawDeg = 35f; pitchDeg = 20f; zoom = 1.0f; panPxX = 0; panPxY = 0; }
        private void FitViewToContent() { panPxX = panPxY = 0; zoom = 1.0f; }

        private void LayoutViewerSquare(Panel host, Label label, PictureBox viewer)
        {
            int availW = Math.Max(0, host.ClientSize.Width - host.Padding.Left - host.Padding.Right);
            int availH = Math.Max(0, host.ClientSize.Height - host.Padding.Top - host.Padding.Bottom - label.Height);
            if (availW <= 0 || availH <= 0) return; int side = Math.Max(64, Math.Min(availW, availH));
            int left = host.Padding.Left + (availW - side) / 2; int top = host.Padding.Top + label.Height + (availH - side) / 2; viewer.Bounds = new Rectangle(left, top, side, side);
        }

        private void Pic3D_MouseDown(object sender, MouseEventArgs e) { pic3D.Focus(); lastMouse = e.Location; if (e.Button == MouseButtons.Left) isRotating = true; if (e.Button == MouseButtons.Right) isPanning = true; }
        private void Pic3D_MouseMove(object sender, MouseEventArgs e)
        {
            if (!isRotating && !isPanning) return; long now = Environment.TickCount64; if (now - lastDragTick < DragFrameMs) return; lastDragTick = now;
            int dx = e.X - lastMouse.X; int dy = e.Y - lastMouse.Y; lastMouse = e.Location;
            if (isRotating) { yawDeg += dx * 0.5f; pitchDeg -= dy * 0.5f; if (pitchDeg < -89f) pitchDeg = -89f; if (pitchDeg > 89f) pitchDeg = 89f; }
            if (isPanning) { panPxX += dx; panPxY += dy; }
            RenderPreviews(); UpdateStatus();
        }
        private void Pic3D_MouseUp(object sender, MouseEventArgs e) { if (e.Button == MouseButtons.Left) isRotating = false; if (e.Button == MouseButtons.Right) isPanning = false; }
        private void Pic3D_MouseWheel(object sender, MouseEventArgs e) { float factor = (float)Math.Pow(1.1, e.Delta / 120.0); zoom *= factor; if (zoom < 0.1f) zoom = 0.1f; if (zoom > 10f) zoom = 10f; RenderPreviews(); UpdateStatus(); }

        private void InitSplittersSafe()
        {
            splitMain.Panel1MinSize = 250; splitMain.Panel2MinSize = 120; var topDesired = (int)Math.Round(splitMain.Height * 0.7);
            splitMain.SplitterDistance = Clamp(topDesired, splitMain.Panel1MinSize, splitMain.Height - splitMain.SplitterWidth - splitMain.Panel2MinSize);

            splitLeftRight.Panel1MinSize = 340; splitLeftRight.Panel2MinSize = 420; int leftDesired = 460;
            splitLeftRight.SplitterDistance = Clamp(leftDesired, splitLeftRight.Panel1MinSize, splitLeftRight.Width - splitLeftRight.SplitterWidth - splitLeftRight.Panel2MinSize);

            splitPreviews.Panel1MinSize = 220; splitPreviews.Panel2MinSize = 220; int desired = splitPreviews.Width / 2;
            splitPreviews.SplitterDistance = Clamp(desired, splitPreviews.Panel1MinSize, Math.Max(splitPreviews.Panel1MinSize, splitPreviews.Width - splitPreviews.SplitterWidth - splitPreviews.Panel2MinSize));

            LayoutViewerSquare(viewHost3D, lbl3D, pic3D); LayoutViewerSquare(viewHostUV, lblUV, picUV);
        }

        private static int Clamp(int v, int min, int max) => v < min ? min : (v > max ? max : v);
        private void Log(string line) => txtLog.AppendText(line + Environment.NewLine);
        private void AppendException(string where, Exception ex)
        {
            Log($"[ERROR] {where}: {ex.GetType().Name}: {ex.Message}"); if (ex.StackTrace != null) Log(ex.StackTrace);
            for (var ie = ex.InnerException; ie != null; ie = ie.InnerException) { Log($"  Inner: {ie.GetType().Name}: {ie.Message}"); if (ie.StackTrace != null) Log(ie.StackTrace); }
        }
    }
}
