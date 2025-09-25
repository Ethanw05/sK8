"""All-in-one Blender add-on that stitches together the existing PSG utilities.

The goal is to expose a linear workflow:

1. Import a donor PSG (armature + mesh) directly into Blender using the
   low-level parser from ``MeshNBones``.
2. Allow the artist to select a donor texture and keep track of its size.
3. Combine the edited meshes + materials into a single atlas using the baking
   helpers from ``Atlas_Bake_addon``.
4. Export the customised mesh back to PSG format using the conversion logic
   from the ``Gltf2PSG`` tool (via Blender's glTF exporter).
5. Resize & save the baked atlas so it matches the donor texture dimensions.

All heavy lifting still happens in the original scripts; this add-on simply
coordinates them behind a small UI.  The user only needs to set the donor
files, press the buttons in order, and handle skinning in Blender before the
export step.
"""

from __future__ import annotations

bl_info = {
    "name": "PSG Pipeline Toolkit",
    "author": "OpenAI ChatGPT",
    "version": (1, 0, 0),
    "blender": (3, 3, 0),
    "location": "3D Viewport > N-Panel > PSG Pipeline",
    "description": "Imports donor PSGs, combines meshes/UVs, bakes atlas textures,"
                   " and exports modified PSG + texture in one place.",
    "category": "Import-Export",
}

import os
import tempfile
import struct
import importlib.machinery
import importlib.util
from dataclasses import dataclass

import bpy
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

# Third-party dependencies used by the original conversion scripts.
import numpy
import pygltflib


# -----------------------------------------------------------------------------
# Utility loaders for the legacy scripts (they live next to this add-on).
# -----------------------------------------------------------------------------

_ADDON_DIR = os.path.dirname(__file__)


def _load_module(name: str, relative_path: str):
    """Load a Python source file (possibly without .py extension) as a module."""

    module_path = os.path.join(_ADDON_DIR, relative_path)
    loader = importlib.machinery.SourceFileLoader(name, module_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Unable to load helper module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_MESHNBONES = None
_ATLAS = None


def _meshnbones():
    global _MESHNBONES
    if _MESHNBONES is None:
        _MESHNBONES = _load_module("psg_meshnbones", "MeshNBones")
    return _MESHNBONES


def _atlas():
    global _ATLAS
    if _ATLAS is None:
        _ATLAS = _load_module("psg_atlas", "Atlas_Bake_addon.py")
    return _ATLAS


# -----------------------------------------------------------------------------
# Data model for conversion (ported from the Qt tool but Blender-friendly).
# -----------------------------------------------------------------------------


class PsgTemplateParser:
    """Minimal wrapper around the donor PSG file used as export template."""

    RW_GRAPHICS_VERTEXDESCRIPTOR = 0x000200E9
    RW_GRAPHICS_VERTEXBUFFER = 0x000200EA
    RW_GRAPHICS_INDEXBUFFER = 0x000200EB
    PEGASUS_OPTIMESHDATA = 0x00EB0023

    def __init__(self, file_path: str):
        with open(file_path, "rb") as f:
            self.data = f.read()

        self.vdes_offset = -1
        self.vertex_offset = -1
        self.face_offset = -1
        self.vbuff_dict_ptr = -1
        self.ibuff_dict_ptr = -1
        self.main_baseresource_size = 0x44
        self.graphics_baseresource_size = 0x6C
        self.vertex_buffer_size_offset = -1
        self.index_count_offset = -1
        self.optimesh_index_offset = -1
        self.bone_names: list[str] = []
        self.bone_palette: list[int] = []

        self._parse_dictionary_and_skeleton()
        self.layout = self._parse_vdes()

    # --- helpers -----------------------------------------------------------
    def _u16_be(self, offset: int) -> int:
        return int.from_bytes(self.data[offset:offset + 2], "big")

    def _u32_be(self, offset: int) -> int:
        return int.from_bytes(self.data[offset:offset + 4], "big")

    def _is_base_resource(self, type_id: int) -> bool:
        return 0x00010030 <= type_id <= 0x0001003F

    # --- parsing -----------------------------------------------------------
    def _parse_dictionary_and_skeleton(self):
        num_entries = self._u32_be(0x20)
        dict_start = self._u32_be(0x30)
        main_base = self._u32_be(0x44)

        dict_entries = []
        for i in range(num_entries):
            entry_offset = dict_start + (i * 0x18)
            entry = {
                "ptr": self._u32_be(entry_offset + 0x00),
                "size": self._u32_be(entry_offset + 0x08),
                "type_id": self._u32_be(entry_offset + 0x14),
                "offset": entry_offset,
            }
            dict_entries.append(entry)

        carrier_entry = self._find_carrier(dict_entries, main_base)
        if carrier_entry:
            self._parse_carrier(carrier_entry, main_base)
        palette_entry = next((e for e in dict_entries if e["type_id"] == self.PEGASUS_OPTIMESHDATA), None)
        if palette_entry:
            self._parse_bone_palette(palette_entry, main_base)

        for entry in dict_entries:
            type_id = entry["type_id"]
            ptr = entry["ptr"]
            block_start = (main_base + ptr) if self._is_base_resource(type_id) else ptr

            if type_id == self.RW_GRAPHICS_VERTEXDESCRIPTOR and self.vdes_offset == -1:
                self.vdes_offset = block_start
            elif type_id == self.RW_GRAPHICS_VERTEXBUFFER and self.vertex_offset == -1:
                br_index = self._u32_be(block_start)
                br_entry = dict_entries[br_index]
                br_ptr = br_entry["ptr"]
                br_type_id = br_entry["type_id"]
                self.vertex_offset = (main_base + br_ptr) if self._is_base_resource(br_type_id) else br_ptr
                self.vertex_buffer_size_offset = block_start + 8
                self.vbuff_dict_ptr = br_entry["offset"]
            elif type_id == self.RW_GRAPHICS_INDEXBUFFER and self.face_offset == -1:
                br_index = self._u32_be(block_start)
                br_entry = dict_entries[br_index]
                br_ptr = br_entry["ptr"]
                br_type_id = br_entry["type_id"]
                self.face_offset = (main_base + br_ptr) if self._is_base_resource(br_type_id) else br_ptr
                self.index_count_offset = block_start + 8
                self.ibuff_dict_ptr = br_entry["offset"]
            elif type_id == self.PEGASUS_OPTIMESHDATA and self.optimesh_index_offset == -1:
                self.optimesh_index_offset = block_start + 0x64

        if self.vdes_offset == -1 or self.vertex_offset == -1 or self.face_offset == -1:
            raise ValueError("Template is missing vertex/index buffers or descriptor.")

    def _find_carrier(self, dict_entries, main_base):
        for entry in dict_entries:
            block_start = (main_base + entry["ptr"]) if self._is_base_resource(entry["type_id"]) else entry["ptr"]
            block_end = block_start + entry["size"]
            header_offset = block_start + 0x20
            if header_offset + 0x24 > len(self.data):
                continue
            bone_count = self._u16_be(header_offset + 0x14)
            if not (0 < bone_count <= 512):
                continue
            off_ibm = self._u32_be(header_offset + 0x00)
            off_tbl_idx = self._u32_be(header_offset + 0x08)
            ibm_abs = block_start + off_ibm
            idx_abs = block_start + off_tbl_idx
            if (ibm_abs + bone_count * 64 <= block_end) and (idx_abs + bone_count * 4 <= block_end):
                return entry
        return None

    def _parse_carrier(self, carrier_entry, main_base):
        block_start = (main_base + carrier_entry["ptr"]) if self._is_base_resource(carrier_entry["type_id"]) else carrier_entry["ptr"]
        header_offset = block_start + 0x20
        bone_count = self._u16_be(header_offset + 0x14)
        off_tbl_idx = self._u32_be(header_offset + 0x08)
        idx_abs = block_start + off_tbl_idx
        self.bone_names = []
        for i in range(bone_count):
            rel = self._u32_be(idx_abs + 4 * i)
            name_offset = block_start + rel
            end_offset = self.data.find(b"\x00", name_offset)
            name = self.data[name_offset:end_offset].decode("ascii", errors="ignore")
            self.bone_names.append(name)

    def _parse_bone_palette(self, palette_entry, main_base):
        block_start = (main_base + palette_entry["ptr"]) if self._is_base_resource(palette_entry["type_id"]) else palette_entry["ptr"]
        palette_offset = block_start + 0x6C
        self.bone_palette = []
        p = palette_offset
        while p + 1 < len(self.data):
            idx = self._u16_be(p)
            if idx == 0xFFFF or idx >= len(self.bone_names):
                break
            self.bone_palette.append(idx)
            p += 2

    def _parse_vdes(self):
        num_elements = self._u16_be(self.vdes_offset + 10)
        elements_offset = self.vdes_offset + 16
        parsed_elements = []
        strides = set()

        for i in range(num_elements):
            elem_offset = elements_offset + (i * 8)
            elem_data = self.data[elem_offset:elem_offset + 8]
            parsed_elements.append(VertexElement(
                vertex_type=elem_data[0],
                num_components=elem_data[1],
                stream=elem_data[2],
                offset=elem_data[3],
                stride=int.from_bytes(elem_data[4:6], "big"),
                elem_type=elem_data[6],
                class_id=elem_data[7],
            ))
            if parsed_elements[-1].stride > 0:
                strides.add(parsed_elements[-1].stride)

        if not strides:
            raise ValueError("Template vertex descriptor has no stride information.")

        return VertexLayout(stride=max(strides), elements=parsed_elements)


@dataclass
class VertexElement:
    vertex_type: int
    num_components: int
    stream: int
    offset: int
    stride: int
    elem_type: int
    class_id: int


@dataclass
class VertexLayout:
    stride: int
    elements: list[VertexElement]


class GltfToPsgExporter:
    """Logic ported from the PyQt tool but decoupled from any UI framework."""

    def __init__(self, template_path: str, vertex_scale: float = 256.0):
        self.template_path = template_path
        self.template = PsgTemplateParser(template_path)
        self.vertex_scale = vertex_scale

    # --- helper math ------------------------------------------------------
    @staticmethod
    def normalize_bone_name(name: str | None) -> str | None:
        if name is None:
            return None
        return ''.join(ch for ch in name if ch.isalnum()).lower()

    @staticmethod
    def pack_normal_dec3n(vec) -> bytes:
        nx, ny, nz = vec
        if nx == 0 and ny == 0 and nz == 0:
            return struct.pack('>I', 0)
        clamp = lambda c: max(-1.0, min(1.0, float(c)))
        ix = int(round(clamp(nx) * 511.0)) & 0x3FF
        iy = int(round(clamp(ny) * 511.0)) & 0x3FF
        iz = int(round(clamp(nz) * 511.0)) & 0x3FF
        packed_val = (iz << 22) | (iy << 12) | (ix << 2)
        return struct.pack('>I', packed_val)

    # --- conversion steps -------------------------------------------------
    def parse_gltf_to_data(self, gltf_path: str, bin_path: str | None):
        gltf = pygltflib.GLTF2.load(gltf_path)

        blob = None
        if gltf_path.lower().endswith('.glb'):
            blob = gltf.binary_blob()
        elif bin_path and os.path.exists(bin_path):
            with open(bin_path, 'rb') as f:
                blob = f.read()
        if blob is None:
            raise ValueError("Could not load binary data for glTF export.")

        if not gltf.meshes:
            raise ValueError("No meshes found in exported glTF file.")

        primitive = gltf.meshes[0].primitives[0]

        def get_accessor_data(accessor_id):
            accessor = gltf.accessors[accessor_id]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
            dtype_map = {
                5120: numpy.int8,
                5121: numpy.uint8,
                5122: numpy.int16,
                5123: numpy.uint16,
                5125: numpy.uint32,
                5126: numpy.float32,
            }
            dtype = dtype_map[accessor.componentType]
            num_components = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[accessor.type]
            data = numpy.frombuffer(blob, dtype=dtype, count=accessor.count * num_components, offset=offset)
            return data.reshape(accessor.count, num_components) if num_components > 1 else data

        raw_vertices = get_accessor_data(primitive.attributes.POSITION)
        raw_normals = get_accessor_data(primitive.attributes.NORMAL)
        raw_uvs = get_accessor_data(primitive.attributes.TEXCOORD_0) if primitive.attributes.TEXCOORD_0 is not None else numpy.zeros((len(raw_vertices), 2), dtype=numpy.float32)
        indices = get_accessor_data(primitive.indices)
        faces_indices = indices.reshape(-1, 3)

        raw_joints = None
        raw_weights = None
        glb_bone_map = None

        if primitive.attributes.JOINTS_0 is not None and primitive.attributes.WEIGHTS_0 is not None:
            raw_joints = get_accessor_data(primitive.attributes.JOINTS_0)
            raw_weights = get_accessor_data(primitive.attributes.WEIGHTS_0)

            weights_accessor = gltf.accessors[primitive.attributes.WEIGHTS_0]
            if weights_accessor.componentType == 5121:  # UNSIGNED_BYTE
                raw_weights = raw_weights.astype(numpy.float32) / 255.0
            elif weights_accessor.componentType == 5123:  # UNSIGNED_SHORT
                raw_weights = raw_weights.astype(numpy.float32) / 65535.0

            skin_index = None
            for node in gltf.nodes:
                if node.mesh == 0 and node.skin is not None:
                    skin_index = node.skin
                    break
            if skin_index is None:
                raise ValueError("Skinned data found, but no node in glTF references the skin.")

            if gltf.skins and len(gltf.skins) > skin_index:
                skin = gltf.skins[skin_index]
                glb_bone_map = {i: gltf.nodes[joint_index].name for i, joint_index in enumerate(skin.joints)}
            else:
                raise ValueError("glTF skin definition missing joints for exported mesh.")

        tangent_acc = numpy.zeros_like(raw_vertices)
        for i0, i1, i2 in faces_indices:
            p0, p1, p2 = raw_vertices[[i0, i1, i2]]
            uv0, uv1, uv2 = raw_uvs[[i0, i1, i2]]
            edge1 = p1 - p0
            edge2 = p2 - p0
            delta_uv1 = uv1 - uv0
            delta_uv2 = uv2 - uv0
            f = delta_uv1[0] * delta_uv2[1] - delta_uv2[0] * delta_uv1[1]
            if abs(f) > 1e-6:
                r = 1.0 / f
                tangent = (edge1 * delta_uv2[1] - edge2 * delta_uv1[1]) * r
                tangent_acc[[i0, i1, i2]] += tangent

        t_ortho = tangent_acc - raw_normals * numpy.sum(tangent_acc * raw_normals, axis=1, keepdims=True)
        final_raw_tangents = t_ortho / (numpy.linalg.norm(t_ortho, axis=1, keepdims=True) + 1e-9)
        final_raw_binormals = numpy.cross(raw_normals, final_raw_tangents)

        final_data = {
            "vertices": [],
            "uvs": [],
            "normals": [],
            "tangents": [],
            "binormals": [],
            "joints": [],
            "weights": [],
        }
        is_skinned = raw_joints is not None

        for v_idx in indices:
            final_data["vertices"].append(raw_vertices[v_idx])
            final_data["normals"].append(raw_normals[v_idx])
            final_data["uvs"].append(raw_uvs[v_idx])
            final_data["tangents"].append(final_raw_tangents[v_idx])
            final_data["binormals"].append(final_raw_binormals[v_idx])
            if is_skinned:
                final_data["joints"].append(raw_joints[v_idx])
                final_data["weights"].append(raw_weights[v_idx])

        final_faces = numpy.arange(len(indices)).reshape(-1, 3).tolist()
        joints_out = final_data["joints"] if is_skinned else None
        weights_out = final_data["weights"] if is_skinned else None

        return (
            final_data["vertices"],
            final_data["uvs"],
            final_data["normals"],
            final_data["tangents"],
            final_data["binormals"],
            final_faces,
            joints_out,
            weights_out,
            glb_bone_map,
        )

    def remap_skin_to_donor_palette(self, gltf_joints, gltf_weights, glb_bone_map):
        donor_name_to_global_idx = {self.normalize_bone_name(name): i for i, name in enumerate(self.template.bone_names)}
        global_idx_to_palette_idx = {global_idx: palette_idx for palette_idx, global_idx in enumerate(self.template.bone_palette)}
        gltf_to_palette_map = {}
        for gltf_idx, gltf_name in (glb_bone_map or {}).items():
            norm_name = self.normalize_bone_name(gltf_name)
            global_idx = donor_name_to_global_idx.get(norm_name)
            if global_idx is None:
                continue
            palette_idx = global_idx_to_palette_idx.get(global_idx)
            if palette_idx is not None:
                gltf_to_palette_map[gltf_idx] = palette_idx

        final_palette_indices = []
        final_weights = []
        for indices, weights in zip(gltf_joints, gltf_weights):
            weight_by_palette_idx = {}
            for i in range(4):
                w = float(weights[i])
                if w <= 1e-6:
                    continue
                gltf_joint_idx = int(indices[i])
                palette_idx = gltf_to_palette_map.get(gltf_joint_idx)
                if palette_idx is not None:
                    weight_by_palette_idx[palette_idx] = weight_by_palette_idx.get(palette_idx, 0.0) + w

            sorted_pairs = sorted(weight_by_palette_idx.items(), key=lambda x: x[1], reverse=True)[:4]
            palette_indices_per_vertex = [0] * 4
            weights_per_vertex = [0.0] * 4
            for i, (pal_idx, w_val) in enumerate(sorted_pairs):
                palette_indices_per_vertex[i] = int(pal_idx)
                weights_per_vertex[i] = float(w_val)

            total_weight = sum(weights_per_vertex)
            if total_weight > 1e-6:
                inv = 1.0 / total_weight
                weights_per_vertex = [w * inv for w in weights_per_vertex]
            else:
                palette_indices_per_vertex = [0, 0, 0, 0]
                weights_per_vertex = [1.0, 0.0, 0.0, 0.0]

            final_palette_indices.append(palette_indices_per_vertex)
            final_weights.append(weights_per_vertex)

        return numpy.array(final_palette_indices, dtype=numpy.uint8), numpy.array(final_weights, dtype=numpy.float32)

    def make_vertex_bin_dynamic(self, vertices, uvs, normals, tangents, binormals, joints, weights):
        elem_map = {
            'XYZ': 0,
            'WEIGHTS': 1,
            'NORMAL': 2,
            'VERTEXCOLOR': 3,
            'SPECULAR': 4,
            'BONEINDICES': 7,
            'TEX0': 8,
            'TEX1': 9,
            'TEX2': 10,
            'TEX3': 11,
            'TEX4': 12,
            'TEX5': 13,
            'TANGENT': 14,
            'BINORMAL': 15,
        }

        output = bytearray()
        is_skinned = joints is not None and weights is not None

        for i in range(len(vertices)):
            vertex_bytes = bytearray(self.template.layout.stride)
            for elem in self.template.layout.elements:
                packed = b''
                if elem.elem_type == elem_map['XYZ']:
                    x_s, y_s, z_s = [max(-32768, min(32767, int(c * self.vertex_scale))) for c in vertices[i]]
                    if elem.vertex_type in (0x01, 0x05):
                        packed = struct.pack('>hhh', x_s, y_s, z_s)
                    elif elem.vertex_type == 0x02:
                        packed = struct.pack('>fff', *vertices[i])
                elif elem.elem_type == elem_map['NORMAL']:
                    if elem.vertex_type == 0x06:
                        packed = self.pack_normal_dec3n(normals[i])
                elif elem.elem_type == elem_map['TANGENT']:
                    if elem.vertex_type == 0x06:
                        packed = self.pack_normal_dec3n(tangents[i])
                elif elem.elem_type == elem_map['BINORMAL']:
                    if elem.vertex_type == 0x06:
                        packed = self.pack_normal_dec3n(binormals[i])
                elif elem.elem_type == elem_map['TEX0']:
                    u, v = uvs[i]
                    if elem.vertex_type == 0x03:
                        packed = numpy.array([u, v], dtype='>f2').tobytes()
                    elif elem.vertex_type in (0x01, 0x05):
                        u_s, v_s = [max(-32768, min(32767, int(round(c * 32767.0)))) for c in (u, v)]
                        packed = struct.pack('>hh', u_s, v_s)
                elif elem.elem_type == elem_map['WEIGHTS']:
                    weights_val = weights[i] if is_skinned else [1.0, 0.0, 0.0, 0.0]
                    if elem.vertex_type == 0x02:
                        packed = struct.pack('>ffff', *weights_val)
                    elif elem.vertex_type in (0x04, 0x07):
                        w_u8 = [int(round(max(0.0, min(1.0, float(c))) * 255.0)) for c in weights_val]
                        packed = struct.pack('>BBBB', *w_u8)
                elif elem.elem_type == elem_map['BONEINDICES']:
                    joints_val = joints[i] if is_skinned else [0, 0, 0, 0]
                    if elem.vertex_type in (0x04, 0x07):
                        j_u8 = [int(max(0, min(255, int(v)))) for v in joints_val]
                        packed = struct.pack('>BBBB', *j_u8)
                elif elem.elem_type == elem_map['VERTEXCOLOR']:
                    if elem.vertex_type in (0x04, 0x07):
                        packed = struct.pack('>BBBB', 255, 255, 255, 255)
                elif elem.elem_type == elem_map['SPECULAR']:
                    if elem.vertex_type in (0x04, 0x07):
                        packed = struct.pack('>BBBB', 0, 0, 0, 255)

                if packed:
                    vertex_bytes[elem.offset:elem.offset + len(packed)] = packed
            output.extend(vertex_bytes)
        return output

    @staticmethod
    def make_face_bin(faces):
        output = bytearray()
        for face in faces:
            for idx in face:
                output.extend(struct.pack('>H', int(idx)))
        return output

    def export(self, gltf_path: str, bin_path: str | None, output_path: str):
        (vertices, uvs, normals, tangents, binormals,
         faces, joints, weights, glb_bone_map) = self.parse_gltf_to_data(gltf_path, bin_path)

        remapped_joints = remapped_weights = None
        if joints is not None and weights is not None:
            if not self.template.bone_names or not self.template.bone_palette:
                raise ValueError("Donor template does not contain skeleton data; cannot export skinned mesh.")
            remapped_joints, remapped_weights = self.remap_skin_to_donor_palette(joints, weights, glb_bone_map)

        vertex_data = self.make_vertex_bin_dynamic(vertices, uvs, normals, tangents, binormals, remapped_joints, remapped_weights)
        face_data = self.make_face_bin(faces)

        with open(self.template_path, 'rb') as f:
            psg_data = bytearray(f.read())

        v_offset = self.template.vertex_offset
        original_file_end = int.from_bytes(psg_data[self.template.main_baseresource_size:self.template.main_baseresource_size + 4], 'big')
        psg_data = psg_data[:original_file_end]

        psg_data[self.template.graphics_baseresource_size:self.template.graphics_baseresource_size + 4] = int.to_bytes(len(vertex_data) + len(face_data), 4, 'big')
        psg_data[self.template.vertex_buffer_size_offset:self.template.vertex_buffer_size_offset + 4] = int.to_bytes(len(vertex_data), 4, 'big')
        psg_data[self.template.index_count_offset:self.template.index_count_offset + 4] = int.to_bytes(len(faces) * 3, 4, 'big')
        if self.template.optimesh_index_offset > 0:
            psg_data[self.template.optimesh_index_offset:self.template.optimesh_index_offset + 4] = int.to_bytes(len(faces) * 3, 4, 'big')

        psg_data.extend(b'\x00' * (len(vertex_data) + len(face_data)))
        psg_data[v_offset:v_offset + len(vertex_data)] = vertex_data
        psg_data[self.template.vbuff_dict_ptr + 8:self.template.vbuff_dict_ptr + 12] = int.to_bytes(len(vertex_data), 4, 'big')

        new_f_offset = v_offset + len(vertex_data)
        psg_data[self.template.ibuff_dict_ptr:self.template.ibuff_dict_ptr + 4] = int.to_bytes(len(vertex_data), 4, 'big')
        psg_data[self.template.ibuff_dict_ptr + 8:self.template.ibuff_dict_ptr + 12] = int.to_bytes(len(face_data), 4, 'big')
        psg_data[new_f_offset:new_f_offset + len(face_data)] = face_data

        with open(output_path, 'wb') as f:
            f.write(psg_data)


# -----------------------------------------------------------------------------
# Blender property storage
# -----------------------------------------------------------------------------


class PSGPipelineProperties(PropertyGroup):
    donor_psg: StringProperty(name="Donor PSG", subtype='FILE_PATH')
    donor_texture: StringProperty(name="Donor Texture", subtype='FILE_PATH')
    donor_texture_width: IntProperty(name="Donor Width", default=0)
    donor_texture_height: IntProperty(name="Donor Height", default=0)
    atlas_image_name: StringProperty(name="Atlas Image Name", default="Combined_Albedo")
    atlas_material_name: StringProperty(name="Atlas Material", default="Combined_Material")
    atlas_size: EnumProperty(
        name="Atlas Size",
        items=[('1024', '1024', ''), ('2048', '2048', ''), ('4096', '4096', ''), ('8192', '8192', '')],
        default='4096'
    )
    pack_margin: FloatProperty(name="Pack Margin", default=0.02, min=0.0, max=0.2)
    vertex_scale: FloatProperty(name="Vertex Scale", default=256.0, min=0.01, max=2048.0)
    restrict_images: BoolProperty(name="Limit Bake Sources", default=False)
    allowed_images: StringProperty(name="Allowed Image Names", default="Image_0,Image_3,Image_9")
    apply_transforms: BoolProperty(name="Apply Transform Before Join", default=True)


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------


class PSGPIPELINE_OT_import_donor(Operator):
    bl_idname = "psg_pipeline.import_donor"
    bl_label = "Import Donor PSG"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.psg_pipeline
        if not props.donor_psg:
            self.report({'ERROR'}, "Set a donor PSG file first.")
            return {'CANCELLED'}
        try:
            importer = _meshnbones()
            importer.run_import(props.donor_psg, mesh_name="DonorMesh", armature_name="DonorArmature", context=context)
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import PSG: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, "Donor PSG imported. Use Blender's weighting tools before exporting.")
        return {'FINISHED'}


class PSGPIPELINE_OT_load_texture(Operator):
    bl_idname = "psg_pipeline.load_texture"
    bl_label = "Load Donor Texture"

    def execute(self, context):
        props = context.scene.psg_pipeline
        if not props.donor_texture:
            self.report({'ERROR'}, "Set a donor texture path first.")
            return {'CANCELLED'}
        path = bpy.path.abspath(props.donor_texture)
        if not os.path.exists(path):
            self.report({'ERROR'}, "Donor texture file does not exist.")
            return {'CANCELLED'}
        try:
            img = bpy.data.images.load(path, check_existing=True)
            props.donor_texture_width = img.size[0]
            props.donor_texture_height = img.size[1]
        except RuntimeError as exc:
            self.report({'ERROR'}, f"Could not load texture: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Loaded texture ({img.size[0]}x{img.size[1]}).")
        return {'FINISHED'}


class PSGPIPELINE_OT_combine_meshes(Operator):
    bl_idname = "psg_pipeline.combine"
    bl_label = "Combine Meshes / Bake"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.psg_pipeline
        atlas_module = _atlas()
        prev_engine = context.scene.render.engine
        try:
            prev_engine = atlas_module._ensure_cycles_and_remember()
            meshes = atlas_module._selected_meshes_or_fail()
            if props.apply_transforms:
                for obj in meshes:
                    context.view_layer.objects.active = obj
                    obj.select_set(True)
                    try:
                        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                    except Exception:
                        pass

            atlas_module._preprocess_objects_set_uvsrc(meshes, "UV_SRC", True, props.pack_margin)
            active = atlas_module._join_meshes(meshes)
            context.view_layer.objects.active = active
            active.select_set(True)

            atlas_module._duplicate_uv_to_atlas(active, "UV_SRC", "UV_Atlas")
            atlas_module._pack_islands(active, False, props.pack_margin)

            atlas_size_int = int(props.atlas_size)
            img = atlas_module._get_or_make_image(props.atlas_image_name, atlas_size_int)
            atlas_module._add_bake_target_nodes(active, "UV_Atlas", img)

            allowed = set()
            if props.restrict_images and props.allowed_images.strip():
                allowed = {n.strip() for n in props.allowed_images.split(',') if n.strip()}

            removed_links_per_mat = []
            rewires = []
            for slot in active.material_slots:
                mat = slot.material
                if not mat:
                    continue
                removed = atlas_module._mute_disallowed_image_nodes(mat, allowed)
                removed_links_per_mat.append((mat, removed))
                st, out = atlas_module._rewire_basecolor_to_emission(mat)
                rewires.append((mat, st, out))

            px_margin = max(2, int(atlas_size_int * props.pack_margin * 0.5))
            atlas_module._bake_emit_once(atlas_size_int, px_margin)

            for mat, st, out in rewires:
                atlas_module._restore_from_rewire(mat, st, out)
            for mat, removed in removed_links_per_mat:
                atlas_module._restore_links(mat, removed)
            atlas_module._cleanup_temp_nodes(active)

            atlas_module._assign_final_material(active, img, props.atlas_material_name)

            atlas_module._save_image_if_needed(img, os.path.dirname(bpy.path.abspath(props.donor_texture or "")), props.atlas_image_name)

        except Exception as exc:
            context.scene.render.engine = prev_engine
            self.report({'ERROR'}, f"Atlas bake failed: {exc}")
            return {'CANCELLED'}
        context.scene.render.engine = prev_engine
        self.report({'INFO'}, "Combined meshes and baked atlas texture.")
        return {'FINISHED'}


class PSGPIPELINE_OT_export_psg(Operator):
    bl_idname = "psg_pipeline.export_psg"
    bl_label = "Export Modified PSG"

    def execute(self, context):
        props = context.scene.psg_pipeline
        donor_path = bpy.path.abspath(props.donor_psg)
        if not donor_path:
            self.report({'ERROR'}, "Set a donor PSG file first.")
            return {'CANCELLED'}
        if not os.path.exists(donor_path):
            self.report({'ERROR'}, "Donor PSG path is invalid.")
            return {'CANCELLED'}

        output_dir = os.path.dirname(donor_path)
        donor_name = os.path.splitext(os.path.basename(donor_path))[0]
        output_path = os.path.join(output_dir, f"{donor_name}_modified.psg")

        with tempfile.TemporaryDirectory() as tmp_dir:
            gltf_path = os.path.join(tmp_dir, "scene.gltf")
            bin_path = os.path.join(tmp_dir, "scene.bin")
            try:
                bpy.ops.export_scene.gltf(
                    filepath=gltf_path,
                    export_format='GLTF_SEPARATE',
                    use_selection=True,
                    export_skins=True,
                    export_animations=False,
                    export_materials='EXPORT',
                    use_visible=True,
                    use_active_collection=False,
                    export_apply=False,
                )
            except Exception as exc:
                self.report({'ERROR'}, f"glTF export failed: {exc}")
                return {'CANCELLED'}

            if not os.path.exists(bin_path):
                bin_candidates = [f for f in os.listdir(tmp_dir) if f.lower().endswith('.bin')]
                bin_path = os.path.join(tmp_dir, bin_candidates[0]) if bin_candidates else None

            exporter = GltfToPsgExporter(donor_path, vertex_scale=props.vertex_scale)
            try:
                exporter.export(gltf_path, bin_path, output_path)
            except Exception as exc:
                self.report({'ERROR'}, f"PSG export failed: {exc}")
                return {'CANCELLED'}

        self.report({'INFO'}, f"Exported PSG to {output_path}")
        return {'FINISHED'}


class PSGPIPELINE_OT_export_texture(Operator):
    bl_idname = "psg_pipeline.export_texture"
    bl_label = "Export Baked Texture"

    def execute(self, context):
        props = context.scene.psg_pipeline
        donor_tex_path = bpy.path.abspath(props.donor_texture)
        if not donor_tex_path:
            self.report({'ERROR'}, "Set a donor texture file first.")
            return {'CANCELLED'}
        if not os.path.exists(donor_tex_path):
            self.report({'ERROR'}, "Donor texture path is invalid.")
            return {'CANCELLED'}

        atlas_image = bpy.data.images.get(props.atlas_image_name)
        if not atlas_image:
            self.report({'ERROR'}, "Atlas image not found. Run the combine step first.")
            return {'CANCELLED'}

        try:
            donor_img = bpy.data.images.load(donor_tex_path, check_existing=True)
            width, height = donor_img.size
        except RuntimeError as exc:
            self.report({'ERROR'}, f"Unable to load donor texture: {exc}")
            return {'CANCELLED'}

        atlas_image.scale(width, height)

        output_dir = os.path.dirname(donor_tex_path)
        donor_tex_name, donor_tex_ext = os.path.splitext(os.path.basename(donor_tex_path))
        output_path = os.path.join(output_dir, f"{donor_tex_name}_modified{donor_tex_ext or '.png'}")

        atlas_image.filepath_raw = output_path
        atlas_image.file_format = 'PNG' if donor_tex_ext.lower() != '.dds' else 'DDS'
        try:
            atlas_image.save()
        except RuntimeError as exc:
            self.report({'ERROR'}, f"Failed to save atlas image: {exc}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Saved resized atlas to {output_path}")
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------


class PSGPIPELINE_PT_panel(Panel):
    bl_idname = "PSGPIPELINE_PT_panel"
    bl_label = "PSG Pipeline"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PSG Pipeline'

    def draw(self, context):
        layout = self.layout
        props = context.scene.psg_pipeline

        layout.label(text="1. Donor Assets")
        layout.prop(props, "donor_psg")
        layout.operator(PSGPIPELINE_OT_import_donor.bl_idname, icon='ARMATURE_DATA')
        layout.separator()

        layout.prop(props, "donor_texture")
        layout.operator(PSGPIPELINE_OT_load_texture.bl_idname, icon='IMAGE_DATA')
        layout.separator()

        layout.label(text="2. Combine / Bake")
        layout.prop(props, "atlas_image_name")
        layout.prop(props, "atlas_material_name")
        layout.prop(props, "atlas_size")
        layout.prop(props, "pack_margin")
        layout.prop(props, "apply_transforms")
        layout.prop(props, "restrict_images")
        if props.restrict_images:
            layout.prop(props, "allowed_images")
        layout.operator(PSGPIPELINE_OT_combine_meshes.bl_idname, icon='MESH_DATA')

        layout.separator()
        layout.label(text="3. Export")
        layout.prop(props, "vertex_scale")
        row = layout.row()
        row.operator(PSGPIPELINE_OT_export_psg.bl_idname, icon='EXPORT')
        row.operator(PSGPIPELINE_OT_export_texture.bl_idname, icon='IMAGE')


# -----------------------------------------------------------------------------
# Registration helpers
# -----------------------------------------------------------------------------


classes = (
    PSGPipelineProperties,
    PSGPIPELINE_OT_import_donor,
    PSGPIPELINE_OT_load_texture,
    PSGPIPELINE_OT_combine_meshes,
    PSGPIPELINE_OT_export_psg,
    PSGPIPELINE_OT_export_texture,
    PSGPIPELINE_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.psg_pipeline = bpy.props.PointerProperty(type=PSGPipelineProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.psg_pipeline


if __name__ == "__main__":
    register()
