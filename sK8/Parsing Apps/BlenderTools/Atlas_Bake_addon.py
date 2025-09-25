bl_info = {
    "name": "Atlas Baker – Combine & Bake Base Color",
    "author": "Ethan + ChatGPT",
    "version": (1, 0, 0),
    "blender": (3, 3, 0),
    "location": "3D Viewport > N-Panel > Atlas Baker",
    "description": "Join selected meshes, build a UV atlas, and bake Base Color into one texture + material.",
    "category": "UV",
}

import bpy
import os
from bpy.props import (
    BoolProperty, IntProperty, FloatProperty, StringProperty, EnumProperty,
)

# ----------------------- Core helpers (ported from your working script) -----------------------

def _ensure_cycles_and_remember():
    """Baking requires Cycles. Switch temporarily and return previous engine to restore later."""
    prev = bpy.context.scene.render.engine
    if prev != 'CYCLES':
        bpy.context.scene.render.engine = 'CYCLES'  # Cycles bake requirement (manual)
    return prev

def _selected_meshes_or_fail():
    meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not meshes:
        raise RuntimeError("Select the mesh object(s) you want to combine.")
    return meshes

def _ensure_source_uv_on_object(obj, uv_src_name, use_smart_uv_if_missing, pack_margin):
    me = obj.data
    if len(me.uv_layers) == 0:
        if not use_smart_uv_if_missing:
            raise RuntimeError(f"Object '{obj.name}' has no UVs; enable 'Smart UV if missing'.")
        # create & unwrap so it won’t stack at (0,0)
        me.uv_layers.new(name=uv_src_name)
        me.uv_layers.active = me.uv_layers[uv_src_name]
        bpy.context.view_layer.objects.active = obj
        prev = obj.mode
        try:
            if prev != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=pack_margin,
                                     correct_aspect=True, scale_to_bounds=True, area_weight=0.0)
        finally:
            if prev != 'EDIT':
                bpy.ops.object.mode_set(mode=prev)
    else:
        # normalize name so join merges UVs by name
        active = me.uv_layers.active
        active.name = uv_src_name
        me.uv_layers.active = me.uv_layers[uv_src_name]

def _preprocess_objects_set_uvsrc(objs, uv_src_name, use_smart_uv_if_missing, pack_margin):
    for o in objs:
        _ensure_source_uv_on_object(o, uv_src_name, use_smart_uv_if_missing, pack_margin)

def _join_meshes(objs):
    if len(objs) == 1:
        return objs[0]
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active

def _duplicate_uv_to_atlas(obj, uv_src_name, atlas_uv_name):
    me = obj.data
    if uv_src_name not in me.uv_layers:
        raise RuntimeError(f"Expected UV layer '{uv_src_name}' after join.")
    if atlas_uv_name in me.uv_layers:
        me.uv_layers.remove(me.uv_layers[atlas_uv_name])
    me.uv_layers.new(name=atlas_uv_name)
    src = me.uv_layers[uv_src_name]
    dst = me.uv_layers[atlas_uv_name]
    for i, l in enumerate(dst.data):
        l.uv = src.data[i].uv
    me.uv_layers.active = dst

def _pack_islands(obj, keep_texel_density, pack_margin):
    prev = obj.mode
    try:
        if prev != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_all(action='SELECT')
        try:
            bpy.ops.uv.pack_islands(rotate=True, margin=pack_margin, scale=not keep_texel_density)
        except TypeError:
            bpy.ops.uv.pack_islands(rotate=True, margin=pack_margin)
    finally:
        if prev != 'EDIT':
            bpy.ops.object.mode_set(mode=prev)

def _get_or_make_image(name, size):
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(name, width=size, height=size, alpha=False, float_buffer=False)
    try:
        img.colorspace_settings.name = 'sRGB'
    except Exception:
        pass
    return img

def _add_bake_target_nodes(obj, atlas_uv_name, img):
    """Add UV Map -> Image Texture (selected+active) so the bake writes here (Cycles rule)."""
    for slot in obj.material_slots:
        mat = slot.material
        if not mat:
            continue
        if not mat.use_nodes:
            mat.use_nodes = True
        nt = mat.node_tree
        for n in nt.nodes:
            n.select = False
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = atlas_uv_name
        uvn.location = (-800, 0)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.name = tex.label = "BAKE_TARGET_IMAGE_NODE"
        tex.image = img
        tex.location = (-600, 0)
        nt.links.new(uvn.outputs['UV'], tex.inputs.get('Vector'))
        tex.select = True
        nt.nodes.active = tex

class _LinkStore:
    def __init__(self, node, socket):
        self.node = node
        self.socket = socket
        self.links = []

def _mute_disallowed_image_nodes(mat, allowed_names_set):
    """Temporarily disconnect image textures not in allowed_names_set (except our bake target)."""
    removed = []
    if not mat or not mat.use_nodes or not allowed_names_set:
        return removed
    nt = mat.node_tree
    for n in nt.nodes:
        if n.type == 'TEX_IMAGE' and n.image and n.name != "BAKE_TARGET_IMAGE_NODE":
            if n.image.name not in allowed_names_set:
                for out in n.outputs:
                    if not out.is_linked:
                        continue
                    st = _LinkStore(n, out)
                    for l in list(out.links):
                        st.links.append((l.to_node, l.to_socket))
                        nt.links.remove(l)
                    if st.links:
                        removed.append(st)
    return removed

def _restore_links(mat, removed):
    if not mat or not mat.use_nodes:
        return
    nt = mat.node_tree
    for st in removed:
        for to_node, to_socket in st.links:
            try:
                nt.links.new(st.socket, to_socket)
            except Exception:
                pass

class _RewireState:
    def __init__(self):
        self.surface_links = []
        self.em_nodes = []

def _ensure_bsdf(nt):
    node = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if node is None:
        node = nt.nodes.new("ShaderNodeBsdfPrincipled")
    return node

def _rewire_basecolor_to_emission(mat):
    """BaseColor → Emission → Output for a pure albedo bake (no lighting/spec)."""
    if not mat or not mat.use_nodes:
        return None, None
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None) or nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = _ensure_bsdf(nt)
    st = _RewireState()
    st.surface_links = list(out.inputs['Surface'].links)
    em = nt.nodes.new("ShaderNodeEmission")
    em.location = (bsdf.location.x + 240, bsdf.location.y - 200)
    st.em_nodes.append(em)
    if bsdf.inputs['Base Color'].links:
        src = bsdf.inputs['Base Color'].links[0].from_socket
        nt.links.new(src, em.inputs['Color'])
    else:
        em.inputs['Color'].default_value = bsdf.inputs['Base Color'].default_value
    for l in list(out.inputs['Surface'].links):
        nt.links.remove(l)
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    return st, out

def _restore_from_rewire(mat, st, out):
    if not mat or not mat.use_nodes or not st or not out:
        return
    nt = mat.node_tree
    for l in list(out.inputs['Surface'].links):
        nt.links.remove(l)
    for l in st.surface_links:
        try:
            nt.links.new(l.from_socket, out.inputs['Surface'])
        except Exception:
            pass
    for em in st.em_nodes:
        try:
            nt.nodes.remove(em)
        except Exception:
            pass

def _bake_emit_once(atlas_size, margin_px):
    s = bpy.context.scene
    s.render.bake.use_clear = True          # clear once before baking the whole object
    s.render.bake.margin = margin_px
    s.render.bake.use_selected_to_active = False
    bpy.ops.object.bake(type='EMIT')        # Bake target = ACTIVE Image Texture node per material (manual)

def _cleanup_temp_nodes(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.name == "BAKE_TARGET_IMAGE_NODE" or n.type == 'UV_MAP':
                nt.nodes.remove(n)

def _assign_final_material(obj, img, mat_name):
    """Minimal glTF-friendly graph: Image → Principled(Base Color) → Output."""
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None) or nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None) or nt.nodes.new("ShaderNodeBsdfPrincipled")
    for n in list(nt.nodes):
        if n not in (out, bsdf):
            nt.nodes.remove(n)
    out.location = (400, 300)
    bsdf.location = (100, 300)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.label = "Combined_Albedo"
    tex.location = (-200, 300)
    try:
        tex.colorspace_settings.name = 'sRGB'
    except Exception:
        pass
    nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    obj.data.materials.clear()
    obj.data.materials.append(mat)

def _save_image_if_needed(img, folder, name):
    if not folder:
        return None
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    return path

# ----------------------- Operator -----------------------

class ATLASBAKER_OT_combine_and_bake(bpy.types.Operator):
    """Join selected meshes, build atlas UVs, bake Base Color to a single texture, and assign one material."""
    bl_idname = "atlas_baker.combine_and_bake"
    bl_label = "Combine & Bake (Base Color)"
    bl_options = {'REGISTER', 'UNDO'}

    atlas_size: EnumProperty(
        name="Atlas Size",
        items=[('1024','1024',''), ('2048','2048',''), ('4096','4096',''), ('8192','8192','')],
        default='4096'
    )
    pack_margin: FloatProperty(name="Pack Margin", default=0.02, min=0.0, max=0.2, description="UV padding")
    keep_texel_density: BoolProperty(name="Keep Texel Density (no scale)", default=False)
    use_smart_uv_if_missing: BoolProperty(name="Smart UV if Missing", default=True)
    uv_src_name: StringProperty(name="Source UV Name", default="UV_SRC")
    atlas_uv_name: StringProperty(name="Atlas UV Name", default="UVMap_Atlas")
    bake_image_name: StringProperty(name="Bake Image Name", default="Combined_Albedo")
    final_material_name: StringProperty(name="Final Material", default="Combined_Material")
    allowed_images_csv: StringProperty(
        name="Allowed Image Names (CSV)",
        description="Limit base-color sources to these image names (comma-separated). Leave blank to allow all.",
        default="Image_0,Image_3,Image_9"
    )
    save_folder: StringProperty(name="Save PNG To Folder", subtype='DIR_PATH', default="")
    apply_transforms: BoolProperty(name="Apply Rot/Scale before Join", default=True)

    def execute(self, context):
        try:
            prev_engine = _ensure_cycles_and_remember()

            meshes = _selected_meshes_or_fail()
            if self.apply_transforms:
                # apply R/S so exporters & bakes are consistent
                for o in meshes:
                    bpy.context.view_layer.objects.active = o
                    o.select_set(True)
                    try:
                        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                    except Exception:
                        pass

            # Normalize UV names per-object so they merge correctly
            _preprocess_objects_set_uvsrc(
                meshes, self.uv_src_name, self.use_smart_uv_if_missing, self.pack_margin
            )

            # Join
            active = _join_meshes(meshes)
            bpy.context.view_layer.objects.active = active
            active.select_set(True)

            # Duplicate UV_SRC -> atlas and pack (spreads islands, no overlaps)
            _duplicate_uv_to_atlas(active, self.uv_src_name, self.atlas_uv_name)
            _pack_islands(active, self.keep_texel_density, self.pack_margin)

            # Bake Base Color → EMIT
            atlas_size_int = int(self.atlas_size)
            img = _get_or_make_image(self.bake_image_name, atlas_size_int)
            _add_bake_target_nodes(active, self.atlas_uv_name, img)

            # Prepare per-material rewires & optional filtering of image nodes
            allowed = {n.strip() for n in self.allowed_images_csv.split(",")} if self.allowed_images_csv.strip() else set()
            removed_links_per_mat = []
            rewires = []
            for slot in active.material_slots:
                mat = slot.material
                if not mat:
                    continue
                removed = _mute_disallowed_image_nodes(mat, allowed)
                removed_links_per_mat.append((mat, removed))
                st, out = _rewire_basecolor_to_emission(mat)
                rewires.append((mat, st, out))

            px_margin = max(2, int(atlas_size_int * self.pack_margin * 0.5))
            _bake_emit_once(atlas_size_int, px_margin)

            # Restore and clean
            for mat, st, out in rewires:
                _restore_from_rewire(mat, st, out)
            for mat, removed in removed_links_per_mat:
                _restore_links(mat, removed)
            _cleanup_temp_nodes(active)

            # Assign single material with baked atlas
            _assign_final_material(active, img, self.final_material_name)

            # Save PNG if requested
            saved_path = _save_image_if_needed(img, self.save_folder, self.bake_image_name)
            if saved_path:
                self.report({'INFO'}, f"Saved atlas to: {saved_path}")

            # Restore previous render engine
            bpy.context.scene.render.engine = prev_engine

            self.report({'INFO'}, f"Done. Mesh '{active.name}' combined. One UV & one material assigned.")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Atlas bake failed: {e}")
            return {'CANCELLED'}

# ----------------------- UI Panel -----------------------

class ATLASBAKER_PT_panel(bpy.types.Panel):
    bl_label = "Atlas Baker"
    bl_idname = "ATLASBAKER_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Atlas Baker"

    def draw(self, ctx):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Combine & Bake Base Color")

        props = col.operator(ATLASBAKER_OT_combine_and_bake.bl_idname, text="Combine Selected → One Atlas", icon='IMAGE_DATA')

        # Lightweight “inline” options
        box = layout.box()
        box.label(text="Settings")
        row = box.row(align=True)
        row.prop(ctx.scene, "atlas_baker_atlas_size", text="Atlas Size")
        row.prop(ctx.scene, "atlas_baker_margin", text="Margin")
        row = box.row(align=True)
        row.prop(ctx.scene, "atlas_baker_keep_td", text="Keep Texel")
        row.prop(ctx.scene, "atlas_baker_smart_uv", text="Smart UV if Missing")

        # Text inputs
        box.prop(ctx.scene, "atlas_baker_uv_src", text="Source UV")
        box.prop(ctx.scene, "atlas_baker_uv_atlas", text="Atlas UV")
        box.prop(ctx.scene, "atlas_baker_img_name", text="Image Name")
        box.prop(ctx.scene, "atlas_baker_mat_name", text="Material Name")
        box.prop(ctx.scene, "atlas_baker_allowed_csv", text="Allowed Images (CSV)")
        box.prop(ctx.scene, "atlas_baker_save_dir", text="Save PNG Folder")
        box.prop(ctx.scene, "atlas_baker_apply_xforms", text="Apply Rot/Scale")

        # Pass scene props into operator call
        props.atlas_size              = ctx.scene.atlas_baker_atlas_size
        props.pack_margin            = ctx.scene.atlas_baker_margin
        props.keep_texel_density     = ctx.scene.atlas_baker_keep_td
        props.use_smart_uv_if_missing= ctx.scene.atlas_baker_smart_uv
        props.uv_src_name            = ctx.scene.atlas_baker_uv_src
        props.atlas_uv_name          = ctx.scene.atlas_baker_uv_atlas
        props.bake_image_name        = ctx.scene.atlas_baker_img_name
        props.final_material_name    = ctx.scene.atlas_baker_mat_name
        props.allowed_images_csv     = ctx.scene.atlas_baker_allowed_csv
        props.save_folder            = ctx.scene.atlas_baker_save_dir
        props.apply_transforms       = ctx.scene.atlas_baker_apply_xforms

# ----------------------- Scene properties (UI state) -----------------------

def _init_scene_props():
    sc = bpy.types.Scene
    sc.atlas_baker_atlas_size = EnumProperty(
        name="Atlas Size", items=[('1024','1024',''), ('2048','2048',''), ('4096','4096',''), ('8192','8192','')], default='4096'
    )
    sc.atlas_baker_margin  = FloatProperty(name="Margin", default=0.02, min=0.0, max=0.2)
    sc.atlas_baker_keep_td = BoolProperty(name="Keep Texel Density", default=False)
    sc.atlas_baker_smart_uv= BoolProperty(name="Smart UV if Missing", default=True)
    sc.atlas_baker_uv_src  = StringProperty(name="Source UV", default="UV_SRC")
    sc.atlas_baker_uv_atlas= StringProperty(name="Atlas UV", default="UVMap_Atlas")
    sc.atlas_baker_img_name= StringProperty(name="Image Name", default="Combined_Albedo")
    sc.atlas_baker_mat_name= StringProperty(name="Material Name", default="Combined_Material")
    sc.atlas_baker_allowed_csv = StringProperty(
        name="Allowed Images (CSV)", default="Image_0,Image_3,Image_9",
        description="Only these image nodes may influence Base Color while baking. Leave blank to allow all."
    )
    sc.atlas_baker_save_dir= StringProperty(name="Save PNG Folder", subtype='DIR_PATH', default="")
    sc.atlas_baker_apply_xforms = BoolProperty(name="Apply Rot/Scale", default=True)

def _clear_scene_props():
    sc = bpy.types.Scene
    del sc.atlas_baker_atlas_size
    del sc.atlas_baker_margin
    del sc.atlas_baker_keep_td
    del sc.atlas_baker_smart_uv
    del sc.atlas_baker_uv_src
    del sc.atlas_baker_uv_atlas
    del sc.atlas_baker_img_name
    del sc.atlas_baker_mat_name
    del sc.atlas_baker_allowed_csv
    del sc.atlas_baker_save_dir
    del sc.atlas_baker_apply_xforms

# ----------------------- Register -----------------------

classes = (
    ATLASBAKER_OT_combine_and_bake,
    ATLASBAKER_PT_panel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    _init_scene_props()

def unregister():
    _clear_scene_props()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
