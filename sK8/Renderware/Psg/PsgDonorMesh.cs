using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace sK8.Renderware.Psg
{
    /// <summary>
    /// Represents the base mesh data extracted from a donor PSG file.
    /// </summary>
    public sealed class PsgDonorMesh
    {
        private PsgDonorMesh(
            byte[] fileBytes,
            IReadOnlyList<string> boneNames,
            IReadOnlyDictionary<string, int> normalizedBoneIndex,
            IReadOnlyList<ushort> bonePalette,
            PsgVertexDescriptor descriptor,
            int vertexCount,
            int vertexStride,
            int vertexBufferOffset,
            int vertexBufferSize)
        {
            FileBytes = fileBytes;
            BoneNames = boneNames;
            NormalizedBoneNameToIndex = normalizedBoneIndex;
            BonePalette = bonePalette;
            Descriptor = descriptor;
            VertexCount = vertexCount;
            VertexStride = vertexStride;
            VertexBufferOffset = vertexBufferOffset;
            VertexBufferSize = vertexBufferSize;
        }

        public byte[] FileBytes { get; }

        public IReadOnlyList<string> BoneNames { get; }

        public IReadOnlyDictionary<string, int> NormalizedBoneNameToIndex { get; }

        public IReadOnlyList<ushort> BonePalette { get; }

        public PsgVertexDescriptor Descriptor { get; }

        public int VertexCount { get; }

        public int VertexStride { get; }

        public int VertexBufferOffset { get; }

        public int VertexBufferSize { get; }

        public static PsgDonorMesh Load(ReadOnlySpan<byte> data, Action<string>? log = null)
        {
            log ??= _ => { };
            if (data.Length < 0x60)
            {
                throw new InvalidOperationException("File is too small to be a valid PSG arena.");
            }

            var bytes = data.ToArray();
            var arena = ParseArena(bytes);
            log($"[Arena] entries={arena.Dict.Count} types={arena.Types.Count}");

            var carrier = FindCarrier(bytes, arena) ?? throw new InvalidOperationException("Could not locate skeleton carrier in donor PSG.");
            log($"[Carrier] dict#{carrier.Index} range=0x{carrier.BlockStart:X}-0x{carrier.BlockEnd:X}");

            var boneNames = ReadBoneNames(bytes, carrier, out var boneMap);
            log($"[Bones] count={boneNames.Count}");

            var paletteEntry = arena.Dict.FirstOrDefault(e => e.TypeId == 0x00EB0023)
                ?? throw new InvalidOperationException("Bone palette entry (type 0x00EB0023) not found in donor PSG.");
            var palette = ReadBonePalette(bytes, paletteEntry, boneNames.Count);
            log($"[Palette] entries={palette.Count} dict#{paletteEntry.Index}");

            var submeshes = ResolveSubMeshes(bytes, arena);
            if (submeshes.Count == 0)
            {
                throw new InvalidOperationException("No vertex descriptors/vertex buffers could be paired in donor PSG.");
            }

            var baseSub = ChooseBaseSubMesh(submeshes);
            log($"[SubMesh] chosen VDES#{baseSub.Vdes.Index} VB#{baseSub.Vb.Index} stride={baseSub.Descriptor.Stride} verts={baseSub.VertexCount}");

            if (baseSub.Descriptor.IndicesElement == null || baseSub.Descriptor.WeightsElement == null)
            {
                throw new InvalidOperationException("Base submesh does not expose indices and weights in the active stream.");
            }

            return new PsgDonorMesh(
                bytes,
                boneNames,
                boneMap,
                palette,
                baseSub.Descriptor,
                baseSub.VertexCount,
                baseSub.Descriptor.Stride,
                baseSub.VertexBufferStart,
                baseSub.VertexBufferSize);
        }

        public static string NormalizeBoneName(string? name)
        {
            return PsgNameUtilities.Normalize(name);
        }

        public int GetPaletteSlotForGlobalBone(int globalIndex)
        {
            for (int i = 0; i < BonePalette.Count; i++)
            {
                if (BonePalette[i] == globalIndex)
                {
                    return i;
                }
            }
            return -1;
        }

        private static IReadOnlyList<string> ReadBoneNames(byte[] data, DictEntry carrier, out Dictionary<string, int> normalizedMap)
        {
            int header = carrier.BlockStart + 0x20;
            uint offIdx = BE.U32(data, header + 0x08);
            uint offNames = BE.U32(data, header + 0x0C);
            ushort boneCount = BE.U16(data, header + 0x14);

            int idxBase = carrier.BlockStart + (int)offIdx;
            int nameBase = carrier.BlockStart + (int)offNames;

            var names = new List<string>(boneCount);
            normalizedMap = new Dictionary<string, int>(boneCount);

            for (int i = 0; i < boneCount; i++)
            {
                uint rel = BE.U32(data, idxBase + i * 4);
                int strPos = carrier.BlockStart + (int)rel;
                string name = ReadCString(data, strPos, carrier.BlockEnd);
                names.Add(name);

                string key = PsgNameUtilities.Normalize(name);
                if (!normalizedMap.ContainsKey(key))
                {
                    normalizedMap[key] = i;
                }
            }

            return names;
        }

        private static IReadOnlyList<ushort> ReadBonePalette(byte[] data, DictEntry paletteEntry, int boneCount)
        {
            var palette = new List<ushort>();
            int pos = paletteEntry.BlockStart + 0x6C;
            while (pos + 1 < paletteEntry.BlockEnd)
            {
                ushort value = BE.U16(data, pos);
                if (value == 0xFFFF || value >= boneCount)
                {
                    break;
                }
                palette.Add(value);
                pos += 2;
            }
            if (palette.Count == 0)
            {
                throw new InvalidOperationException("Bone palette is empty in donor PSG.");
            }
            return palette;
        }

        private static List<ResolvedSubMesh> ResolveSubMeshes(byte[] data, Arena arena)
        {
            var vdesList = arena.Dict.Where(x => x.TypeId == 0x000200E9).OrderBy(x => x.Index).ToList();
            var vbList = arena.Dict.Where(x => x.TypeId == 0x000200EA).OrderBy(x => x.Index).ToList();
            var ibList = arena.Dict.Where(x => x.TypeId == 0x000200EB).OrderBy(x => x.Index).ToList();

            var descriptors = new List<(DictEntry Vdes, PsgVertexDescriptor Descriptor)>();
            foreach (var vdes in vdesList)
            {
                var vd = ParseVertexDescriptor(data, vdes);
                ChooseActiveStream(vd);
                descriptors.Add((vdes, new PsgVertexDescriptor(vd)));
            }

            var resolved = new List<ResolvedSubMesh>();
            var usedVbs = new HashSet<int>();

            for (int i = 0; i < descriptors.Count; i++)
            {
                var (vdesEntry, descriptor) = descriptors[i];

                DictEntry? chosenVb = null;
                int vertexCount = 0;

                foreach (var vb in vbList)
                {
                    if (usedVbs.Contains(vb.Index)) continue;

                    if (vb.BlockStart + 12 > data.Length) continue;
                    uint brIndex = BE.U32(data, vb.BlockStart + 0);
                    var br = arena.Dict.ElementAtOrDefault((int)brIndex);
                    if (br == null || !arena.IsBaseResource(br.TypeId)) continue;

                    int size = (int)br.Size;
                    int stride = descriptor.Stride > 0 ? descriptor.Stride : descriptor.Elements.FirstOrDefault()?.Stride ?? 0;
                    if (stride <= 0 || size <= 0) continue;
                    if (size % stride != 0) continue;

                    chosenVb = vb;
                    vertexCount = size / stride;
                    break;
                }

                if (chosenVb == null) continue;
                usedVbs.Add(chosenVb.Index);

                DictEntry? nextVdes = (i + 1 < descriptors.Count) ? descriptors[i + 1].Vdes : null;
                DictEntry? chosenIb = ibList.FirstOrDefault(ib => ib.Index > vdesEntry.Index && (nextVdes == null || ib.Index < nextVdes.Index));

                uint vbBrIndex = BE.U32(data, chosenVb.BlockStart + 0);
                var vbBr = arena.Dict.ElementAtOrDefault((int)vbBrIndex);
                if (vbBr == null || !arena.IsBaseResource(vbBr.TypeId)) continue;

                resolved.Add(new ResolvedSubMesh
                {
                    Vdes = vdesEntry,
                    Vb = chosenVb,
                    Ib = chosenIb,
                    Descriptor = descriptor,
                    VertexCount = vertexCount,
                    VertexBufferStart = vbBr.BlockStart,
                    VertexBufferSize = (int)vbBr.Size
                });
            }

            return resolved;
        }

        private static ResolvedSubMesh ChooseBaseSubMesh(List<ResolvedSubMesh> submeshes)
        {
            var candidates = submeshes
                .Where(s => s.Ib != null && s.Descriptor.IndicesElement != null && s.Descriptor.WeightsElement != null)
                .ToList();

            if (candidates.Count == 0)
            {
                candidates = submeshes
                    .Where(s => s.Descriptor.IndicesElement != null && s.Descriptor.WeightsElement != null)
                    .ToList();
            }

            if (candidates.Count == 0)
            {
                throw new InvalidOperationException("Unable to identify a base submesh containing indices and weights.");
            }

            return candidates
                .OrderByDescending(s => s.Descriptor.HasTangent ? 1 : 0)
                .ThenByDescending(s => s.Descriptor.HasBinormal ? 1 : 0)
                .ThenByDescending(s => s.Descriptor.HasTexCoord)
                .ThenByDescending(s => s.Descriptor.Stride)
                .First();
        }

        private static VertexDescriptorInfo ParseVertexDescriptor(byte[] data, DictEntry entry)
        {
            if (entry.BlockStart + 16 > data.Length)
            {
                throw new InvalidOperationException("Vertex descriptor header is out of range.");
            }

            ushort elementCount = BE.U16(data, entry.BlockStart + 10);
            var elements = new List<VertexElementInfo>(elementCount);
            int offset = entry.BlockStart + 16;

            for (int i = 0; i < elementCount && offset + 8 <= entry.BlockEnd; i++, offset += 8)
            {
                elements.Add(new VertexElementInfo
                {
                    VertexType = data[offset + 0],
                    ComponentCount = data[offset + 1],
                    Stream = data[offset + 2],
                    Offset = data[offset + 3],
                    Stride = BE.U16(data, offset + 4),
                    ElementType = data[offset + 6],
                    Class = data[offset + 7]
                });
            }

            return new VertexDescriptorInfo(elements);
        }

        private static void ChooseActiveStream(VertexDescriptorInfo descriptor)
        {
            var groups = descriptor.Elements.GroupBy(e => e.Stream).ToList();

            (bool hasIdx, bool hasWgt, bool hasXyz, bool hasNrm, bool hasTan, bool hasBin, bool hasTex, int stride, VertexElementInfo? idx, VertexElementInfo? wgt) Analyze(IEnumerable<VertexElementInfo> elems)
            {
                bool idx = false, wgt = false, xyz = false, nrm = false, tan = false, bin = false, tex = false;
                int strideMax = 0;
                VertexElementInfo? idxEl = null, wgtEl = null;

                foreach (var e in elems)
                {
                    strideMax = Math.Max(strideMax, e.Stride);
                    switch ((PsgVertexElementType)e.ElementType)
                    {
                        case PsgVertexElementType.Indices:
                            idx = true; idxEl = e; break;
                        case PsgVertexElementType.Weights:
                            wgt = true; wgtEl = e; break;
                        case PsgVertexElementType.Xyz:
                            xyz = true; break;
                        case PsgVertexElementType.Normal:
                            nrm = true; break;
                        case PsgVertexElementType.Tangent:
                            tan = true; break;
                        case PsgVertexElementType.Binormal:
                            bin = true; break;
                        case PsgVertexElementType.Tex0:
                        case PsgVertexElementType.Tex1:
                        case PsgVertexElementType.Tex2:
                        case PsgVertexElementType.Tex3:
                        case PsgVertexElementType.Tex4:
                        case PsgVertexElementType.Tex5:
                            tex = true; break;
                    }
                }

                return (idx, wgt, xyz, nrm, tan, bin, tex, strideMax, idxEl, wgtEl);
            }

            var primary = groups
                .Select(g => (g.Key, info: Analyze(g)))
                .Where(x => x.info.hasIdx && x.info.hasWgt)
                .OrderByDescending(x => x.info.stride)
                .FirstOrDefault();

            if (primary.info.stride > 0)
            {
                descriptor.ActiveStream = primary.Key;
                descriptor.Stride = primary.info.stride;
                descriptor.IndicesElement = primary.info.idx;
                descriptor.WeightsElement = primary.info.wgt;
                descriptor.HasXyz = primary.info.hasXyz;
                descriptor.HasNormal = primary.info.hasNrm;
                descriptor.HasTangent = primary.info.hasTan;
                descriptor.HasBinormal = primary.info.hasBin;
                descriptor.HasTex = primary.info.hasTex;
                return;
            }

            var fallback = groups
                .Select(g => (g.Key, info: Analyze(g)))
                .OrderByDescending(x => x.info.stride)
                .First();

            descriptor.ActiveStream = fallback.Key;
            descriptor.Stride = fallback.info.stride;
            descriptor.IndicesElement = fallback.info.idx;
            descriptor.WeightsElement = fallback.info.wgt;
            descriptor.HasXyz = fallback.info.hasXyz;
            descriptor.HasNormal = fallback.info.hasNrm;
            descriptor.HasTangent = fallback.info.hasTan;
            descriptor.HasBinormal = fallback.info.hasBin;
            descriptor.HasTex = fallback.info.hasTex;
        }

        private static Arena ParseArena(byte[] data)
        {
            uint numEntries = BE.U32(data, 0x20);
            uint dictStart = BE.U32(data, 0x30);
            uint mainBase = BE.U32(data, 0x44);
            uint sections = BE.U32(data, 0x34);

            var arena = new Arena { DictStart = dictStart, ResourceMainBase = mainBase };

            if (sections != 0)
            {
                const uint SectionTypesId = 0x00010005;
                for (int p = (int)sections; p <= data.Length - 12; p += 4)
                {
                    if (BE.U32(data, p) != SectionTypesId) continue;
                    uint count = BE.U32(data, p + 4);
                    uint dictOffset = BE.U32(data, p + 8);
                    int baseOffset = p + (int)dictOffset;
                    for (int i = 0; i < count && baseOffset + i * 4 <= data.Length - 4; i++)
                    {
                        arena.Types.Add(BE.U32(data, baseOffset + i * 4));
                    }
                    break;
                }
            }

            for (int i = 0, q = (int)dictStart; i < numEntries && q + 0x18 <= data.Length; i++, q += 0x18)
            {
                uint ptr = BE.U32(data, q + 0x00);
                uint size = BE.U32(data, q + 0x08);
                uint align = BE.U32(data, q + 0x0C);
                uint typeIndex = BE.U32(data, q + 0x10);
                uint typeId = BE.U32(data, q + 0x14);

                if (arena.Types.Count > 0 && typeIndex < arena.Types.Count)
                {
                    typeId = arena.Types[(int)typeIndex];
                }

                int blockStart = arena.IsBaseResource(typeId) ? (int)(mainBase + ptr) : (int)ptr;
                int blockEnd = Math.Min(data.Length, Math.Max(blockStart, blockStart + (int)size));

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

            return arena;
        }

        private static DictEntry? FindCarrier(byte[] data, Arena arena)
        {
            foreach (var entry in arena.Dict)
            {
                int header = entry.BlockStart + 0x20;
                if (header + 0x24 > data.Length) continue;

                uint offIbm = BE.U32(data, header + 0x00);
                uint offIndex = BE.U32(data, header + 0x08);
                uint offNames = BE.U32(data, header + 0x0C);
                ushort boneCount = BE.U16(data, header + 0x14);

                if (boneCount == 0 || boneCount > 512) continue;

                int ibmAbs = entry.BlockStart + (int)offIbm;
                int idxAbs = entry.BlockStart + (int)offIndex;
                int nameAbs = entry.BlockStart + (int)offNames;

                if (!entry.Contains(ibmAbs, boneCount * 64)) continue;
                if (!entry.Contains(idxAbs, boneCount * 4)) continue;
                if (!entry.Contains(nameAbs, 1)) continue;

                int rel0 = (int)BE.U32(data, idxAbs);
                int str0 = entry.BlockStart + rel0;
                if (!entry.Contains(str0, 1)) continue;
                if (!LooksLikeCString(data, str0, entry.BlockEnd)) continue;

                return entry;
            }

            return null;
        }

        private static string ReadCString(byte[] data, int start, int limit)
        {
            if (start < 0 || start >= data.Length) return string.Empty;
            var sb = new StringBuilder();
            int max = Math.Min(limit, data.Length);
            for (int i = start; i < max; i++)
            {
                byte b = data[i];
                if (b == 0) break;
                sb.Append((char)b);
            }
            return sb.ToString();
        }

        private static bool LooksLikeCString(byte[] data, int offset, int limit)
        {
            int max = Math.Min(limit, offset + 64);
            for (int i = offset; i < max; i++)
            {
                byte b = data[i];
                if (b == 0) return true;
                if (b < 0x20 || b > 0x7E) return false;
            }
            return false;
        }

        private sealed class Arena
        {
            public uint DictStart { get; set; }
            public uint ResourceMainBase { get; set; }
            public List<uint> Types { get; } = new();
            public List<DictEntry> Dict { get; } = new();
            public bool IsBaseResource(uint typeId) => typeId >= 0x00010030 && typeId <= 0x0001003F;
        }

        private sealed class DictEntry
        {
            public int Index { get; set; }
            public uint Ptr { get; set; }
            public uint Size { get; set; }
            public uint Align { get; set; }
            public uint TypeIndex { get; set; }
            public uint TypeId { get; set; }
            public int BlockStart { get; set; }
            public int BlockEnd { get; set; }

            public bool Contains(int start, int length) => start >= BlockStart && start + length <= BlockEnd;
        }

        private sealed class ResolvedSubMesh
        {
            public DictEntry Vdes { get; set; } = default!;
            public DictEntry Vb { get; set; } = default!;
            public DictEntry? Ib { get; set; }
            public PsgVertexDescriptor Descriptor { get; set; } = default!;
            public int VertexCount { get; set; }
            public int VertexBufferStart { get; set; }
            public int VertexBufferSize { get; set; }
        }

        private static class BE
        {
            public static ushort U16(byte[] d, int o) => (ushort)((d[o] << 8) | d[o + 1]);
            public static uint U32(byte[] d, int o) => ((uint)d[o] << 24) | ((uint)d[o + 1] << 16) | ((uint)d[o + 2] << 8) | d[o + 3];
        }
    }

    public enum PsgVertexComponentType : byte
    {
        S1 = 0x01,
        Float32 = 0x02,
        Float16 = 0x03,
        UByteNormalized = 0x04,
        S32K = 0x05,
        PackedCmp = 0x06,
        UByte = 0x07
    }

    public enum PsgVertexElementType : byte
    {
        Xyz = 0x00,
        Weights = 0x01,
        Normal = 0x02,
        VertexColor = 0x03,
        Specular = 0x04,
        Indices = 0x07,
        Tex0 = 0x08,
        Tex1 = 0x09,
        Tex2 = 0x0A,
        Tex3 = 0x0B,
        Tex4 = 0x0C,
        Tex5 = 0x0D,
        Tangent = 0x0E,
        Binormal = 0x0F
    }

    internal sealed class VertexElementInfo
    {
        public byte VertexType { get; set; }
        public byte ComponentCount { get; set; }
        public byte Stream { get; set; }
        public byte Offset { get; set; }
        public ushort Stride { get; set; }
        public byte ElementType { get; set; }
        public byte Class { get; set; }
    }

    internal sealed class VertexDescriptorInfo
    {
        public VertexDescriptorInfo(List<VertexElementInfo> elements)
        {
            Elements = elements;
        }

        public List<VertexElementInfo> Elements { get; }
        public int ActiveStream { get; set; }
        public int Stride { get; set; }
        public VertexElementInfo? IndicesElement { get; set; }
        public VertexElementInfo? WeightsElement { get; set; }
        public bool HasXyz { get; set; }
        public bool HasNormal { get; set; }
        public bool HasTangent { get; set; }
        public bool HasBinormal { get; set; }
        public bool HasTex { get; set; }
    }

    public sealed class PsgVertexDescriptor
    {
        private readonly List<PsgVertexElement> _elements;

        internal PsgVertexDescriptor(VertexDescriptorInfo info)
        {
            _elements = info.Elements.Select(e => new PsgVertexElement(
                (PsgVertexComponentType)e.VertexType,
                e.ComponentCount,
                e.Stream,
                e.Offset,
                e.Stride,
                (PsgVertexElementType)e.ElementType,
                e.Class)).ToList();

            ActiveStream = info.ActiveStream;
            Stride = info.Stride;
            IndicesElement = info.IndicesElement != null ? _elements.FirstOrDefault(e => e.Offset == info.IndicesElement.Offset && e.Stream == info.IndicesElement.Stream) : null;
            WeightsElement = info.WeightsElement != null ? _elements.FirstOrDefault(e => e.Offset == info.WeightsElement.Offset && e.Stream == info.WeightsElement.Stream) : null;
            HasNormal = info.HasNormal;
            HasTangent = info.HasTangent;
            HasBinormal = info.HasBinormal;
            HasTexCoord = info.HasTex;
        }

        public IReadOnlyList<PsgVertexElement> Elements => _elements;

        public int ActiveStream { get; }

        public int Stride { get; }

        public bool HasNormal { get; }

        public bool HasTangent { get; }

        public bool HasBinormal { get; }

        public bool HasTexCoord { get; }

        public PsgVertexElement? IndicesElement { get; }

        public PsgVertexElement? WeightsElement { get; }

        public PsgVertexElement? FindElement(PsgVertexElementType type)
        {
            return _elements.FirstOrDefault(e => e.ElementType == type && e.Stream == ActiveStream);
        }
    }

    public sealed class PsgVertexElement
    {
        internal PsgVertexElement(PsgVertexComponentType vertexType, byte componentCount, byte stream, byte offset, ushort stride, PsgVertexElementType elementType, byte elementClass)
        {
            ComponentType = vertexType;
            ComponentCount = componentCount;
            Stream = stream;
            Offset = offset;
            Stride = stride;
            ElementType = elementType;
            ElementClass = elementClass;
        }

        public PsgVertexComponentType ComponentType { get; }
        public byte ComponentCount { get; }
        public byte Stream { get; }
        public byte Offset { get; }
        public ushort Stride { get; }
        public PsgVertexElementType ElementType { get; }
        public byte ElementClass { get; }
    }

    internal static class PsgNameUtilities
    {
        public static string Normalize(string? name)
        {
            if (string.IsNullOrWhiteSpace(name)) return string.Empty;
            var sb = new StringBuilder(name.Length);
            foreach (char c in name)
            {
                if (char.IsWhiteSpace(c) || c == '_' || c == '-' )
                {
                    continue;
                }
                sb.Append(char.ToUpperInvariant(c));
            }
            return sb.ToString();
        }
    }
}
