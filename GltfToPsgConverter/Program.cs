using SharpGLTF.Schema2;
using sK8.Renderware.Psg;
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace GltfToPsgConverter
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }

    public sealed class MainForm : Form
    {
        private readonly TextBox _txtDonor;
        private readonly TextBox _txtGltf;
        private readonly TextBox _txtOutput;
        private readonly TextBox _txtLog;
        private readonly Button _btnConvert;

        public MainForm()
        {
            Text = "GLTF → PSG Converter";
            StartPosition = FormStartPosition.CenterScreen;
            MinimumSize = new System.Drawing.Size(900, 600);

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 4,
                RowCount = 6,
                Padding = new Padding(10)
            };
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 120));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 120));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 120));

            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            layout.Controls.Add(new Label { Text = "Donor PSG:", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 0);
            _txtDonor = new TextBox { Anchor = AnchorStyles.Left | AnchorStyles.Right };
            layout.Controls.Add(_txtDonor, 1, 0);
            layout.SetColumnSpan(_txtDonor, 2);
            var btnDonor = new Button { Text = "Browse...", Anchor = AnchorStyles.Left };
            btnDonor.Click += (s, e) => BrowseFile(_txtDonor, "Select donor PSG", "PSG files (*.psg)|*.psg|All files (*.*)|*.*");
            layout.Controls.Add(btnDonor, 3, 0);

            layout.Controls.Add(new Label { Text = "GLTF/GLB:", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 1);
            _txtGltf = new TextBox { Anchor = AnchorStyles.Left | AnchorStyles.Right };
            layout.Controls.Add(_txtGltf, 1, 1);
            layout.SetColumnSpan(_txtGltf, 2);
            var btnGltf = new Button { Text = "Browse...", Anchor = AnchorStyles.Left };
            btnGltf.Click += (s, e) => BrowseFile(_txtGltf, "Select GLTF", "GLTF/GLB files (*.gltf;*.glb)|*.gltf;*.glb|All files (*.*)|*.*");
            layout.Controls.Add(btnGltf, 3, 1);

            layout.Controls.Add(new Label { Text = "Output PSG:", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 2);
            _txtOutput = new TextBox { Anchor = AnchorStyles.Left | AnchorStyles.Right };
            layout.Controls.Add(_txtOutput, 1, 2);
            layout.SetColumnSpan(_txtOutput, 2);
            var btnOutput = new Button { Text = "Browse...", Anchor = AnchorStyles.Left };
            btnOutput.Click += (s, e) => BrowseSaveFile(_txtOutput, "Save converted PSG", "PSG files (*.psg)|*.psg|All files (*.*)|*.*");
            layout.Controls.Add(btnOutput, 3, 2);

            _btnConvert = new Button { Text = "Convert", Anchor = AnchorStyles.Left, Width = 120 };
            _btnConvert.Click += async (s, e) => await ConvertAsync();
            layout.Controls.Add(_btnConvert, 0, 3);

            var lblHint = new Label
            {
                Text = "The donor PSG provides bone and vertex layout information. The GLTF must contain a mesh bound to a compatible skin.",
                AutoSize = true,
                MaximumSize = new System.Drawing.Size(600, 0),
                Anchor = AnchorStyles.Left
            };
            layout.Controls.Add(lblHint, 1, 3);
            layout.SetColumnSpan(lblHint, 3);

            layout.Controls.Add(new Label { Text = "Log:", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 4);
            _txtLog = new TextBox
            {
                Multiline = true,
                ScrollBars = ScrollBars.Both,
                WordWrap = false,
                Font = new System.Drawing.Font("Consolas", 9f),
                Dock = DockStyle.Fill
            };
            layout.Controls.Add(_txtLog, 0, 5);
            layout.SetColumnSpan(_txtLog, 4);

            Controls.Add(layout);
        }

        private void BrowseFile(TextBox target, string title, string filter)
        {
            using var dialog = new OpenFileDialog { Title = title, Filter = filter };
            if (dialog.ShowDialog(this) == DialogResult.OK)
            {
                target.Text = dialog.FileName;
                if (target == _txtDonor && string.IsNullOrWhiteSpace(_txtOutput.Text))
                {
                    _txtOutput.Text = Path.Combine(Path.GetDirectoryName(dialog.FileName) ?? string.Empty, Path.GetFileNameWithoutExtension(dialog.FileName) + "_converted.psg");
                }
            }
        }

        private void BrowseSaveFile(TextBox target, string title, string filter)
        {
            using var dialog = new SaveFileDialog { Title = title, Filter = filter };
            if (!string.IsNullOrWhiteSpace(target.Text))
            {
                dialog.FileName = target.Text;
            }
            if (dialog.ShowDialog(this) == DialogResult.OK)
            {
                target.Text = dialog.FileName;
            }
        }

        private async Task ConvertAsync()
        {
            var donorPath = _txtDonor.Text.Trim();
            var gltfPath = _txtGltf.Text.Trim();
            var outputPath = _txtOutput.Text.Trim();

            if (!File.Exists(donorPath))
            {
                MessageBox.Show(this, "Select a donor PSG first.", "Missing PSG", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            if (!File.Exists(gltfPath))
            {
                MessageBox.Show(this, "Select a GLTF/GLB file.", "Missing GLTF", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            if (string.IsNullOrWhiteSpace(outputPath))
            {
                MessageBox.Show(this, "Choose an output path for the converted PSG.", "Missing output", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            try
            {
                _btnConvert.Enabled = false;
                UseWaitCursor = true;
                _txtLog.Clear();
                Log($"Donor PSG: {donorPath}");
                Log($"GLTF: {gltfPath}");
                Log($"Output: {outputPath}");

                await Task.Run(() => ConvertInternal(donorPath, gltfPath, outputPath));

                Log("Conversion complete.");
                MessageBox.Show(this, "Conversion complete.", "Done", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                Log($"[ERROR] {ex}");
                MessageBox.Show(this, $"Conversion failed:\n{ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                UseWaitCursor = false;
                _btnConvert.Enabled = true;
            }
        }

        private void ConvertInternal(string donorPath, string gltfPath, string outputPath)
        {
            Log("Loading donor PSG...");
            var donorBytes = File.ReadAllBytes(donorPath);
            var donorMesh = PsgDonorMesh.Load(donorBytes, Log);

            Log("Loading GLTF...");
            var model = ModelRoot.Load(gltfPath);
            var meshData = LoadGltfMesh(model);

            if (meshData.Positions.Length != donorMesh.VertexCount)
            {
                throw new InvalidOperationException($"Vertex count mismatch. Donor PSG has {donorMesh.VertexCount} verts, GLTF has {meshData.Positions.Length} verts.");
            }

            var jointPaletteMap = BuildJointPaletteMap(donorMesh, meshData.Skin);
            if (jointPaletteMap.Count == 0)
            {
                throw new InvalidOperationException("No overlapping bones were found between the donor PSG palette and the GLTF skin.");
            }

            var outputBytes = (byte[])donorMesh.FileBytes.Clone();
            var vertexData = new Span<byte>(outputBytes, donorMesh.VertexBufferOffset, donorMesh.VertexBufferSize);

            const float zOffset = 0.8f; // Undo import offset from Blender script
            var rotation = Matrix4x4.CreateRotationX(-MathF.PI / 2f);

            var missingNormals = meshData.Normals.Length == 0;
            var missingTangents = meshData.Tangents.Length == 0;
            var missingUv = meshData.Tex0.Length == 0;

            if (missingNormals) Log("[WARN] GLTF does not contain normals. Existing PSG normals will be preserved.");
            if (missingTangents) Log("[WARN] GLTF does not contain tangents. Existing PSG tangents/binormals will be preserved.");
            if (missingUv) Log("[WARN] GLTF does not contain TEXCOORD_0. Existing PSG UVs will be preserved.");

            _ = donorMesh.Descriptor.IndicesElement ?? throw new InvalidOperationException("Donor PSG descriptor has no indices element in active stream.");
            _ = donorMesh.Descriptor.WeightsElement ?? throw new InvalidOperationException("Donor PSG descriptor has no weights element in active stream.");

            var missingJointNames = new HashSet<string>();

            for (int i = 0; i < donorMesh.VertexCount; i++)
            {
                Vector3 pos = meshData.Positions[i];
                pos.Z -= zOffset;
                pos = Vector3.Transform(pos, rotation);

                Vector3 normal = Vector3.Zero;
                if (!missingNormals)
                {
                    normal = Vector3.TransformNormal(meshData.Normals[i], rotation);
                    if (normal.LengthSquared() > 0) normal = Vector3.Normalize(normal);
                }

                Vector4 tangent = Vector4.Zero;
                Vector3 binormal = Vector3.Zero;
                if (!missingTangents && meshData.Tangents.Length > i && !missingNormals)
                {
                    tangent = meshData.Tangents[i];
                    var tangentVec = new Vector3(tangent.X, tangent.Y, tangent.Z);
                    tangentVec = Vector3.TransformNormal(tangentVec, rotation);
                    if (tangentVec.LengthSquared() > 0) tangentVec = Vector3.Normalize(tangentVec);
                    tangent = new Vector4(tangentVec, tangent.W);

                    var nrm = normal.LengthSquared() > 0 ? normal : Vector3.Zero;
                    if (nrm != Vector3.Zero && tangentVec != Vector3.Zero)
                    {
                        binormal = Vector3.Normalize(Vector3.Cross(nrm, tangentVec) * tangent.W);
                    }
                }

                Vector2 uv = Vector2.Zero;
                if (!missingUv)
                {
                    uv = meshData.Tex0[i];
                }

                Span<int> paletteIndices = stackalloc int[4];
                Span<float> paletteWeights = stackalloc float[4];
                PopulateInfluences(i, meshData, jointPaletteMap, paletteIndices, paletteWeights, missingJointNames);

                WriteVertex(vertexData, donorMesh.Descriptor, donorMesh.VertexStride, i, pos, normal, tangent, binormal, uv, paletteIndices, paletteWeights);
            }

            if (missingJointNames.Count > 0)
            {
                foreach (var name in missingJointNames.OrderBy(n => n))
                {
                    Log($"[WARN] Bone '{name}' from GLTF skin is not present in donor palette; its weights were discarded.");
                }
            }

            Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? string.Empty);
            File.WriteAllBytes(outputPath, outputBytes);
        }

        private Dictionary<int, int> BuildJointPaletteMap(PsgDonorMesh donor, Skin skin)
        {
            var map = new Dictionary<int, int>();
            for (int i = 0; i < skin.Joints.Count; i++)
            {
                var node = skin.Joints[i];
                string normalized = PsgDonorMesh.NormalizeBoneName(node?.Name);
                if (string.IsNullOrEmpty(normalized))
                {
                    continue;
                }
                if (!donor.NormalizedBoneNameToIndex.TryGetValue(normalized, out int globalIndex))
                {
                    continue;
                }
                int paletteSlot = donor.GetPaletteSlotForGlobalBone(globalIndex);
                if (paletteSlot >= 0)
                {
                    map[i] = paletteSlot;
                }
            }
            return map;
        }

        private void PopulateInfluences(int vertexIndex, GltfMeshData meshData, Dictionary<int, int> jointPaletteMap, Span<int> indices, Span<float> weights, HashSet<string> missing)
        {
            indices[0] = indices[1] = indices[2] = indices[3] = 0;
            weights[0] = weights[1] = weights[2] = weights[3] = 0;

            if (meshData.Joints.Length == 0 || meshData.Weights.Length == 0)
            {
                weights[0] = 1f;
                return;
            }

            var jointsVec = meshData.Joints[vertexIndex];
            var weightsVec = meshData.Weights[vertexIndex];

            Span<int> jointIndices = stackalloc int[4]
            {
                (int)MathF.Round(jointsVec.X),
                (int)MathF.Round(jointsVec.Y),
                (int)MathF.Round(jointsVec.Z),
                (int)MathF.Round(jointsVec.W)
            };

            Span<float> weightValues = stackalloc float[4] { weightsVec.X, weightsVec.Y, weightsVec.Z, weightsVec.W };

            int count = 0;
            for (int c = 0; c < 4; c++)
            {
                float w = weightValues[c];
                if (w <= 0f) continue;
                int joint = jointIndices[c];
                if (!jointPaletteMap.TryGetValue(joint, out int palette))
                {
                    string name = meshData.GetJointName(joint);
                    if (!string.IsNullOrEmpty(name)) missing.Add(name);
                    continue;
                }
                indices[count] = palette;
                weights[count] = w;
                count++;
                if (count == 4) break;
            }

            if (count == 0)
            {
                weights[0] = 1f;
                indices[0] = 0;
                return;
            }

            float sum = 0f;
            for (int i = 0; i < count; i++) sum += weights[i];
            if (sum > 1e-6f)
            {
                float inv = 1f / sum;
                for (int i = 0; i < count; i++) weights[i] *= inv;
            }
            else
            {
                weights[0] = 1f;
                for (int i = 1; i < 4; i++) weights[i] = 0f;
            }

            for (int i = count; i < 4; i++)
            {
                indices[i] = indices[0];
                weights[i] = 0f;
            }
        }

        private void WriteVertex(Span<byte> vertexData, PsgVertexDescriptor descriptor, int stride, int vertexIndex, Vector3 position, Vector3 normal, Vector4 tangent, Vector3 binormal, Vector2 uv, Span<int> paletteIndices, Span<float> paletteWeights)
        {
            var vertexSpan = vertexData.Slice(vertexIndex * stride, stride);
            Span<float> valueBuffer = stackalloc float[4];

            foreach (var element in descriptor.Elements.Where(e => e.Stream == descriptor.ActiveStream))
            {
                int componentSize = GetComponentSize(element.ComponentType);
                int requiredLength = componentSize * element.ComponentCount;
                if (element.Offset + requiredLength > vertexSpan.Length)
                {
                    continue;
                }

                var destination = vertexSpan.Slice(element.Offset, requiredLength);
                int valueCount = FillElementValues(element, position, normal, tangent, binormal, uv, paletteIndices, paletteWeights, valueBuffer);
                if (valueCount <= 0)
                {
                    continue;
                }

                WriteElement(destination, element, valueBuffer[..valueCount]);
            }
        }

        private static int FillElementValues(PsgVertexElement element, Vector3 position, Vector3 normal, Vector4 tangent, Vector3 binormal, Vector2 uv, Span<int> paletteIndices, Span<float> paletteWeights, Span<float> buffer)
        {
            buffer.Clear();
            switch (element.ElementType)
            {
                case PsgVertexElementType.Xyz:
                    buffer[0] = position.X;
                    buffer[1] = position.Y;
                    buffer[2] = position.Z;
                    return 3;
                case PsgVertexElementType.Normal:
                    if (normal == Vector3.Zero) return 0;
                    buffer[0] = normal.X;
                    buffer[1] = normal.Y;
                    buffer[2] = normal.Z;
                    return 3;
                case PsgVertexElementType.Tangent:
                    if (tangent == Vector4.Zero) return 0;
                    buffer[0] = tangent.X;
                    buffer[1] = tangent.Y;
                    buffer[2] = tangent.Z;
                    if (element.ComponentCount >= 4)
                    {
                        buffer[3] = tangent.W;
                        return 4;
                    }
                    return Math.Min(3, element.ComponentCount);
                case PsgVertexElementType.Binormal:
                    if (binormal == Vector3.Zero) return 0;
                    buffer[0] = binormal.X;
                    buffer[1] = binormal.Y;
                    buffer[2] = binormal.Z;
                    return 3;
                case PsgVertexElementType.Tex0:
                    buffer[0] = uv.X;
                    buffer[1] = uv.Y;
                    return Math.Min(2, element.ComponentCount);
                case PsgVertexElementType.Weights:
                    for (int i = 0; i < Math.Min(4, element.ComponentCount); i++) buffer[i] = paletteWeights[i];
                    return Math.Min(4, element.ComponentCount);
                case PsgVertexElementType.Indices:
                    for (int i = 0; i < Math.Min(4, element.ComponentCount); i++) buffer[i] = paletteIndices[i];
                    return Math.Min(4, element.ComponentCount);
                default:
                    return 0;
            }
        }

        private static void WriteElement(Span<byte> destination, PsgVertexElement element, ReadOnlySpan<float> values)
        {
            int componentSize = GetComponentSize(element.ComponentType);
            int count = Math.Min(values.Length, element.ComponentCount);
            switch (element.ComponentType)
            {
                case PsgVertexComponentType.Float32:
                    for (int i = 0; i < element.ComponentCount; i++)
                    {
                        float value = i < count ? values[i] : 0f;
                        BinaryPrimitives.WriteSingleBigEndian(destination.Slice(i * componentSize, componentSize), value);
                    }
                    break;
                case PsgVertexComponentType.Float16:
                    for (int i = 0; i < element.ComponentCount; i++)
                    {
                        float value = i < count ? values[i] : 0f;
                        Half half = (Half)value;
                        ref ushort rawRef = ref Unsafe.As<Half, ushort>(ref half);
                        BinaryPrimitives.WriteUInt16BigEndian(destination.Slice(i * componentSize, componentSize), rawRef);
                    }
                    break;
                case PsgVertexComponentType.S1:
                    for (int i = 0; i < element.ComponentCount; i++)
                    {
                        float value = i < count ? values[i] : 0f;
                        short raw = (short)MathF.Round(Math.Clamp(value, -1f, 1f) * 32767f);
                        BinaryPrimitives.WriteInt16BigEndian(destination.Slice(i * componentSize, componentSize), raw);
                    }
                    break;
                case PsgVertexComponentType.S32K:
                    float scale = element.ElementType == PsgVertexElementType.Xyz ? 16384f : 1f;
                    for (int i = 0; i < element.ComponentCount; i++)
                    {
                        float value = i < count ? values[i] : 0f;
                        short raw = (short)Math.Clamp(MathF.Round(value * scale), short.MinValue, short.MaxValue);
                        BinaryPrimitives.WriteInt16BigEndian(destination.Slice(i * componentSize, componentSize), raw);
                    }
                    break;
                case PsgVertexComponentType.UByteNormalized:
                    for (int i = 0; i < element.ComponentCount; i++)
                    {
                        float value = i < count ? values[i] : 0f;
                        bool normalize = element.ElementType != PsgVertexElementType.Indices;
                        byte raw = (byte)Math.Clamp(MathF.Round(normalize ? value * 255f : value), 0f, 255f);
                        destination[i] = raw;
                    }
                    break;
                case PsgVertexComponentType.UByte:
                    for (int i = 0; i < element.ComponentCount; i++)
                    {
                        float value = i < count ? values[i] : 0f;
                        byte raw = (byte)Math.Clamp(MathF.Round(value), 0f, 255f);
                        destination[i] = raw;
                    }
                    break;
                case PsgVertexComponentType.PackedCmp:
                    // Packed 10:10:10:2 vectors are not adjusted; retain original bytes.
                    break;
            }
        }

        private static int GetComponentSize(PsgVertexComponentType type)
        {
            return type switch
            {
                PsgVertexComponentType.Float32 => 4,
                PsgVertexComponentType.Float16 => 2,
                PsgVertexComponentType.S1 => 2,
                PsgVertexComponentType.S32K => 2,
                PsgVertexComponentType.UByte => 1,
                PsgVertexComponentType.UByteNormalized => 1,
                PsgVertexComponentType.PackedCmp => 4,
                _ => 1
            };
        }

        private GltfMeshData LoadGltfMesh(ModelRoot model)
        {
            var mesh = model.LogicalMeshes.FirstOrDefault(m => m.Primitives.Count > 0)
                ?? throw new InvalidOperationException("GLTF file does not contain a mesh with primitives.");
            var primitive = mesh.Primitives[0];

            Vector3[] positions = primitive.GetVertexAccessor("POSITION")?.AsVector3Array()
                ?? throw new InvalidOperationException("GLTF mesh is missing POSITION data.");
            Vector3[] normals = primitive.GetVertexAccessor("NORMAL")?.AsVector3Array() ?? Array.Empty<Vector3>();
            Vector4[] tangents = primitive.GetVertexAccessor("TANGENT")?.AsVector4Array() ?? Array.Empty<Vector4>();
            Vector2[] tex0 = primitive.GetVertexAccessor("TEXCOORD_0")?.AsVector2Array() ?? Array.Empty<Vector2>();
            Vector4[] joints = primitive.GetVertexAccessor("JOINTS_0")?.AsVector4Array() ?? Array.Empty<Vector4>();
            Vector4[] weights = primitive.GetVertexAccessor("WEIGHTS_0")?.AsVector4Array() ?? Array.Empty<Vector4>();

            var node = model.LogicalNodes.FirstOrDefault(n => n.Mesh == mesh && n.Skin != null)
                ?? model.LogicalNodes.FirstOrDefault(n => n.Skin != null);
            var skin = node?.Skin ?? model.LogicalSkins.FirstOrDefault()
                ?? throw new InvalidOperationException("GLTF file does not contain a skin with joints.");

            return new GltfMeshData(positions, normals, tangents, tex0, joints, weights, skin);
        }

        private void Log(string message)
        {
            if (InvokeRequired)
            {
                BeginInvoke(new Action<string>(Log), message);
                return;
            }
            _txtLog.AppendText(message + Environment.NewLine);
        }

        private sealed class GltfMeshData
        {
            public GltfMeshData(Vector3[] positions, Vector3[] normals, Vector4[] tangents, Vector2[] tex0, Vector4[] joints, Vector4[] weights, Skin skin)
            {
                Positions = positions;
                Normals = normals;
                Tangents = tangents;
                Tex0 = tex0;
                Joints = joints;
                Weights = weights;
                Skin = skin;
            }

            public Vector3[] Positions { get; }
            public Vector3[] Normals { get; }
            public Vector4[] Tangents { get; }
            public Vector2[] Tex0 { get; }
            public Vector4[] Joints { get; }
            public Vector4[] Weights { get; }
            public Skin Skin { get; }

            public string GetJointName(int jointIndex)
            {
                if (jointIndex < 0 || jointIndex >= Skin.Joints.Count) return string.Empty;
                return Skin.Joints[jointIndex]?.Name ?? string.Empty;
            }
        }
    }
}
