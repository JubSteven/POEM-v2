from copy import deepcopy
import numpy as np
import trimesh
import torch
import warnings
import functools
import importlib
import matplotlib as mpl
from lib.utils.transform import caculate_align_mat


def import_open3d(func):
    """Summary
    import open3d before set CUDA_VISIBLE_DEVICES will cause the latter not work (@BUG)!
    Solution: import open3d in runtime using this decorator.

    Args:
        func (function): function to be wrapped

    Returns:
        function: wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        o3d = importlib.import_module("open3d")
        # print(f"Open3D is imported")
        return func(*args, **kwargs, o3d=o3d)

    return wrapper


class VizContext:

    @import_open3d
    def __init__(self, non_block=False, o3d=None) -> None:
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.running = True

        def shutdown_callback(vis):
            self.reset()
            self.running = False
            self.vis.close()
            self.deinit()

        self.vis.register_key_callback(ord("Q"), shutdown_callback)
        self.geometry_to_viz = dict()

        self.non_block = non_block

    def register_key_callback(self, key, callback):
        self.vis.register_key_callback(ord(key), callback)

    def init(self, point_size=10.0):
        self.vis.create_window()
        self.vis.get_render_option().point_size = point_size
        self.vis.get_render_option().background_color = np.asarray([1, 1, 1])

    def deinit(self):
        self.vis.destroy_window()

    def paint_color_on(self, pts, colors=None):
        if colors is None:
            colors = np.ones_like(pts) * [0.9, 0.9, 0.9]
        elif isinstance(colors, (list, tuple)) and len(colors) == 3:
            colors = np.ones_like(pts) * colors
            # if any of colors greater than 1
            if np.any(colors) > 1:
                colors = colors / 255.0
        elif isinstance(colors, str):
            colors = np.ones_like(pts) * mpl.colors.to_rgb(colors)
        elif isinstance(colors, np.ndarray):
            if colors.ndim == 1 and colors.shape[0] == 3:
                colors = np.ones_like(pts) * colors.reshape(1, 3)
            elif colors.ndim == 2 and colors.shape[1] == 3:  # (NPts, 3)
                pass
        else:
            raise ValueError(f"Unknown color type or shape, "
                             f"support str #ffffff, list/tuple of 3, np.ndarray of shape (3,) or (NPts, 3)")
        print
        return colors

    @import_open3d
    def update_by_mesh(self, geo_key, verts, faces, normals=None, vcolors=None, update=True, o3d=None):
        # already exist and not update
        if self.geometry_to_viz.get(geo_key) and not update:
            return

        if isinstance(verts, torch.Tensor):
            verts = verts.detach().cpu().numpy()
        assert len(verts.shape) == 2 and verts.shape[1] == 3, f"verts.shape: {verts.shape}"

        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        assert len(faces.shape) == 2 and faces.shape[1] == 3, f"faces.shape: {faces.shape}"

        if normals is not None:
            if isinstance(normals, torch.Tensor):
                normals = normals.detach().cpu().numpy()
            assert len(normals.shape) == 2 and normals.shape[1] == 3, f"normals.shape: {normals.shape}"

        vcolors = self.paint_color_on(verts, vcolors)

        if self.geometry_to_viz.get(geo_key) is None:  # create content
            mesh_to_create = o3d.geometry.TriangleMesh()
            mesh_to_create.vertices = o3d.utility.Vector3dVector(verts)
            mesh_to_create.triangles = o3d.utility.Vector3iVector(faces)
            mesh_to_create.vertex_colors = o3d.utility.Vector3dVector(vcolors)
            if normals is not None:
                mesh_to_create.vertex_normals = o3d.utility.Vector3dVector(normals)
            else:
                mesh_to_create.compute_vertex_normals()
                mesh_to_create.compute_triangle_normals()

            self.geometry_to_viz[geo_key] = mesh_to_create  # save to dict
            self.vis.add_geometry(mesh_to_create)
        else:  # update content
            mesh_to_update = self.geometry_to_viz[geo_key]  # retrieve from dict, type: o3d.geometry.TriangleMesh
            mesh_to_update.vertices = o3d.utility.Vector3dVector(verts)
            mesh_to_update.triangles = o3d.utility.Vector3iVector(faces)
            mesh_to_update.vertex_colors = o3d.utility.Vector3dVector(vcolors)
            if normals is not None:
                mesh_to_update.vertex_normals = o3d.utility.Vector3dVector(normals)
            else:
                mesh_to_update.compute_vertex_normals()
                mesh_to_update.compute_triangle_normals()
            self.vis.update_geometry(mesh_to_update)

    @import_open3d
    def update_by_pc(self, geo_key, pcs, normals=None, pcolors=None, update=True, o3d=None):
        # already exist and not update
        if self.geometry_to_viz.get(geo_key) and not update:
            return

        if isinstance(pcs, torch.Tensor):
            pcs = pcs.detach().cpu().numpy()
        assert len(pcs.shape) == 2 and pcs.shape[1] == 3, f"pcs.shape: {pcs.shape}"

        if normals is not None:
            if isinstance(normals, torch.Tensor):
                normals = normals.detach().cpu().numpy()
            assert len(normals.shape) == 2 and normals.shape[1] == 3, f"normals.shape: {normals.shape}"

        pcolors = self.paint_color_on(pcs, pcolors)

        if self.geometry_to_viz.get(geo_key) is None:  # create content
            pc_to_creat = o3d.geometry.PointCloud()
            pc_to_creat.points = o3d.utility.Vector3dVector(pcs)
            pc_to_creat.colors = o3d.utility.Vector3dVector(pcolors)
            if normals is not None:
                pc_to_creat.normals = o3d.utility.Vector3dVector(normals)
            self.geometry_to_viz[geo_key] = pc_to_creat  # save to dict
            self.vis.add_geometry(pc_to_creat)
        else:  # update content
            pcs_to_update = self.geometry_to_viz[geo_key]  # retrieve from dict, type: o3d.geometry.PointCloud
            pcs_to_update.points = o3d.utility.Vector3dVector(pcs)
            pcs_to_update.colors = o3d.utility.Vector3dVector(pcolors)
            if normals is not None:
                pcs_to_update.normals = o3d.utility.Vector3dVector(normals)
            self.vis.update_geometry(pcs_to_update)

    def reset(self):
        self.remove_all_geometry()
        self.running = True

    @import_open3d
    def remove_all_geometry(self, o3d=None):
        for k, geo in self.geometry_to_viz.items():
            if isinstance(geo, (o3d.geometry.TriangleMesh, o3d.geometry.PointCloud)):
                self.vis.remove_geometry(geo, reset_bounding_box=False)
            else:
                warnings.warn(f"Unknown geometry type: {type(geo)}")
        self.geometry_to_viz = dict()

    def add_geometry(self, geo):
        self.vis.add_geometry(geo)

    def add_geometry_list(self, geo_list):
        for geo in geo_list:
            self.vis.add_geometry(geo, reset_bounding_box=False)

    def update_geometry_list(self, geo_list):
        for geo in geo_list:
            self.vis.update_geometry(geo)

    def update_geometry(self, geo):
        self.vis.update_geometry(geo)

    def step(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        self.vis.run()

    def remove_geometry(self, geo):
        self.vis.remove_geometry(geo)

    def remove_geometry_list(self, geo_list):
        for geo in geo_list:
            self.vis.remove_geometry(geo, reset_bounding_box=False)

    def condition(self):
        return self.running and (not self.non_block)


@import_open3d
def open3d_version(o3d=None):
    print("Open3D version:", o3d.__version__)


@import_open3d
def cvt_from_trimesh(mesh: trimesh.Trimesh, o3d=None):
    vertices = mesh.vertices
    faces = mesh.faces
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([
        [0.8, 0.8, 0.8],
    ] * len(vertices)))
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


@import_open3d
def create_coord_system_can(scale=1, transf=None, o3d=None):
    axis_list = []
    cylinder_radius = 0.0015 * scale
    cone_radius = 0.002 * scale
    cylinder_height = 0.05 * scale
    cone_height = 0.008 * scale
    resolution = int(20 * scale)
    cylinder_split = 4
    cone_split = 1

    x = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    x.paint_uniform_color([255 / 255.0, 0 / 255.0, 0 / 255.0])
    align_x = caculate_align_mat(np.array([1, 0, 0]))
    x = x.rotate(align_x, center=(0, 0, 0))
    x.compute_vertex_normals()
    axis_list.append(x)

    y = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    y.paint_uniform_color([0 / 255.0, 255 / 255.0, 0 / 255.0])

    align_y = caculate_align_mat(np.array([0, 1, 0]))
    y = y.rotate(align_y, center=(0, 0, 0))
    y.compute_vertex_normals()
    axis_list.append(y)

    z = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    z.paint_uniform_color([0 / 255.0, 0 / 255.0, 255 / 255.0])
    align_z = caculate_align_mat(np.array([0, 0, 1]))
    z = z.rotate(align_z, center=(0, 0, 0))
    z.compute_vertex_normals()
    axis_list.append(z)

    if transf is not None:
        assert transf.shape == (4, 4), "transf must be 4x4 Transformation matrix"
        for i, axis in enumerate(axis_list):
            axis.rotate(transf[:3, :3], center=(0, 0, 0))
            axis.translate(transf[:3, 3].T)
            axis_list[i] = axis

    return axis_list


@import_open3d
def o3d_arrow(arrow_origin, arrow_direction, o3d=None):
    # Set up the arrow parameters
    arrow_length = 0.01
    arrow_radius = 0.001

    # Create the arrow geometry
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=arrow_radius,
                                                   cone_radius=2 * arrow_radius,
                                                   cylinder_height=arrow_length * 0.7,
                                                   cone_height=arrow_length * 0.3)

    arrow.compute_vertex_normals()

    # Set the arrow pose
    arrow_transform = np.eye(4)
    arrow_transform[:3, 3] = arrow_origin
    arrow_transform[:3, :3] = caculate_align_mat(arrow_direction)

    arrow.transform(arrow_transform)
    return arrow
