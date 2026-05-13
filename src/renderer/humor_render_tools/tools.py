# adapted from HuMoR
import torch
import numpy as np
from tqdm import tqdm
import trimesh
import json
import matplotlib.pyplot as plt
from .parameters import colors, smpl_connections
from .mesh_viewer import MeshViewer

c2c = lambda tensor: tensor.detach().cpu().numpy()  # noqa

# Define the mapping from SMPL part names to the visualization categories
SMPL_TO_BODY_PART = {
    "leftArm": "left_arm",
    "leftForeArm": "left_arm",
    "leftHand": "left_arm",
    "leftHandIndex1": "left_arm",
    "leftShoulder": "left_arm",
    
    "rightArm": "right_arm",
    "rightForeArm": "right_arm",
    "rightHand": "right_arm",
    "rightHandIndex1": "right_arm",
    "rightShoulder": "right_arm",

    "leftUpLeg": "left_leg",
    "leftLeg": "left_leg",
    "leftToeBase": "left_leg",
    "leftFoot": "left_leg",

    "rightUpLeg": "right_leg",
    "rightLeg": "right_leg",
    "rightToeBase": "right_leg",
    "rightFoot": "right_leg",

    "spine": "spine",
    "spine1": "spine",
    "spine2": "spine",
    
    "neck": "head",
    "head": "head",
    
    "hips": "trajectory"
}

def load_smpl_vertex_segmentation(segmentation_file):
    """
    Load the SMPL vertex segmentation from a JSON file and map it to high-level body parts.
    
    Args:
        segmentation_file: Path to the segmentation JSON file.
    
    Returns:
        Dict with body parts as keys and lists of vertex indices as values.
    """
    try:
        with open(segmentation_file, 'r') as f:
            segmentation_data = json.load(f)

        # Initialize body part map
        body_part_map = {key: [] for key in ["left_arm", "right_arm", "left_leg", "right_leg", "spine", "head", "trajectory"]}

        # Assign vertex indices to the mapped body part categories
        for smpl_part, vertices in segmentation_data.items():
            if smpl_part in SMPL_TO_BODY_PART:
                body_part = SMPL_TO_BODY_PART[smpl_part]
                body_part_map[body_part].extend(vertices)

        return body_part_map

    except Exception as e:
        print(f"Error loading segmentation file: {str(e)}")
        return None

class ActionColorizer:
    """Class to handle dynamic coloring based on actions."""
    
    def __init__(self, annotations):
        """
        Initialize the action colorizer.
        
        Args:
            annotations: Dictionary of body part annotations
        """
        self.annotations = annotations
        self.action_colors = {}
        self.generate_action_colors()
        
    def generate_action_colors(self):
        """Generate distinguishable colors for different actions."""
        # Collect all unique actions
        unique_actions = set()
        for body_part, annotations in self.annotations.items():
            if isinstance(annotations, list):  # Skip non-annotation keys like 'duration'
                for ann in annotations:
                    if "text" in ann:
                        unique_actions.add(ann["text"])
        
        # Generate colormap
        n_colors = max(1, len(unique_actions))
        cmap = plt.cm.get_cmap('hsv', n_colors)
        self.action_colors = {action: cmap(i)[:3] for i, action in enumerate(unique_actions)}
        
        # Special case for 'unknown'
        if "unknown" in unique_actions:
            self.action_colors["unknown"] = (0.3, 0.3, 0.3)  # Dark Gray
    
    def get_annotation_at_time(self, body_part, time):
        """Get annotation for a body part at a specific time."""
        if body_part not in self.annotations:
            return {"text": "unknown", "confidence": 1, "source": "none"}
            
        for ann in self.annotations[body_part]:
            if ann["start"] <= time < ann["end"]:
                return ann
        
        return {"text": "unknown", "confidence": 1, "source": "none"}

def viz_smpl_seq(
    pyrender,
    out_path,
    body,
    #
    start=None,
    end=None,
    #
    imw=720,
    imh=720,
    fps=20,
    use_offscreen=True,
    follow_camera=True,
    progress_bar=tqdm,
    #
    contacts=None,
    render_body=True,
    render_joints=False,
    render_skeleton=False,
    render_ground=True,
    ground_plane=None,
    wireframe=False,
    RGBA=False,
    joints_seq=None,
    joints_vel=None,
    vtx_list=None,
    points_seq=None,
    points_vel=None,
    static_meshes=None,
    camera_intrinsics=None,
    img_seq=None,
    point_rad=0.015,
    skel_connections=smpl_connections,
    img_extn="png",
    ground_alpha=1.0,
    body_alpha=None,
    mask_seq=None,
    cam_offset=[0.0, 2.2, 0.9],  # [0.0, 4.0, 1.25],
    ground_color0=[0.8, 0.9, 0.9],
    ground_color1=[0.6, 0.7, 0.7],
    skel_color=[0.5, 0.5, 0.5],  # [0.0, 0.0, 1.0],
    joint_rad=0.015,
    point_color=[0.0, 0.0, 1.0],
    joint_color=[0.0, 1.0, 0.0],
    contact_color=[1.0, 0.0, 0.0],
    vertex_color=colors["vertex"],
    render_bodies_static=None,
    render_points_static=None,
    cam_rot=None,
    colored_parts=False,  # Whether to color different body parts with different colors
    segmentation_file=None,  # Path to SMPL segmentation file
    action_colorizer=None,  # Action colorizer instance
    title=None,  # Added title parameter
):
    """
    Visualizes the body model output of a smpl sequence with dynamic action-based coloring.
    
    - body : body model output from SMPL forward pass (where the sequence is the batch)
    - joints_seq : list of torch/numy tensors/arrays
    - points_seq : list of torch/numpy tensors
    - camera_intrinsics : (fx, fy, cx, cy)
    - ground_plane : [a, b, c, d]
    - render_bodies_static is an integer, if given renders all bodies at once but only every x steps
    - colored_parts : Whether to color different body parts with different colors
    - segmentation_file : Path to SMPL segmentation file with vertex mapping
    - action_colorizer : ActionColorizer instance for dynamic coloring
    - title : Optional title to display at the top of the video
    """

    if contacts is not None and torch.is_tensor(contacts):
        contacts = c2c(contacts)

    if render_body or vtx_list is not None:
        faces = c2c(body.f)
        
        # Create body mesh sequence with dynamic coloring based on actions
        if colored_parts and segmentation_file is not None and action_colorizer is not None:
            try:
                body_mesh_seq = []
                for i in range(body.v.size(0)):
                    vertices = c2c(body.v[i])
                    
                    # Calculate current frame
                    time = i
                    
                    # Load the body part segmentation
                    body_part_map = load_smpl_vertex_segmentation(segmentation_file)
                    
                    if body_part_map is None:
                        raise Exception("Could not load body part map")
                        
                    # Initialize vertex colors to a default gray
                    nv = vertices.shape[0]
                    vertex_colors = np.ones((nv, 3)) * 0.7  # Default gray
                    
                    # For unknown annotations, use dark gray
                    unknown_color = action_colorizer.action_colors.get("unknown", (0.3, 0.3, 0.3))
                    
                    # Apply colors based on body part and current action
                    for body_part, indices in body_part_map.items():
                        # Get current annotation for this body part
                        ann = action_colorizer.get_annotation_at_time(body_part, time)
                        
                        if ann:
                            if body_part == "head" and ann["text"] == "unknown":
                                spine_ann = action_colorizer.get_annotation_at_time("spine", time)
                                base_color = action_colorizer.action_colors.get(spine_ann["text"], (0.7, 0.7, 0.7))
                            else:
                                if ann["text"] == "unknown":
                                    base_color = unknown_color
                                else:
                                    base_color = action_colorizer.action_colors.get(ann["text"], (0.7, 0.7, 0.7))
                            
                            # Adjust alpha based on confidence (1-5)
                            confidence = ann.get("confidence", 1)
                            
                            # Apply color to vertices
                            for idx in indices:
                                if idx < nv:
                                    vertex_colors[idx] = base_color
                    
                    # Add alpha channel if specified
                    if body_alpha is not None:
                        vtx_alpha = np.ones((vertex_colors.shape[0], 1)) * body_alpha
                        vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
                    
                    # Create and append trimesh object
                    body_mesh = trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces,
                        vertex_colors=vertex_colors,
                        process=False
                    )
                    body_mesh_seq.append(body_mesh)
                
            except Exception as e:
                print(f"Error in dynamic coloring: {str(e)}. Using default coloring.")
                # Fallback to default coloring
                nv = body.v.size(1)
                vertex_colors = np.tile(vertex_color, (nv, 1))
                if body_alpha is not None:
                    vtx_alpha = np.ones((vertex_colors.shape[0], 1)) * body_alpha
                    vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
                
                body_mesh_seq = [
                    trimesh.Trimesh(
                        vertices=c2c(body.v[i]),
                        faces=faces,
                        vertex_colors=vertex_colors,
                        process=False,
                    )
                    for i in range(body.v.size(0))
                ]
        else:
            # Original coloring method
            nv = body.v.size(1)
            vertex_colors = np.tile(vertex_color, (nv, 1))
            if body_alpha is not None:
                vtx_alpha = np.ones((vertex_colors.shape[0], 1)) * body_alpha
                vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
            
            body_mesh_seq = [
                trimesh.Trimesh(
                    vertices=c2c(body.v[i]),
                    faces=faces,
                    vertex_colors=vertex_colors,
                    process=False,
                )
                for i in range(body.v.size(0))
            ]

    if render_joints and joints_seq is None:
        # only body joints
        joints_seq = [c2c(body.Jtr[i, :22]) for i in range(body.Jtr.size(0))]
    elif render_joints and torch.is_tensor(joints_seq[0]):
        joints_seq = [c2c(joint_frame) for joint_frame in joints_seq]

    if joints_vel is not None and torch.is_tensor(joints_vel[0]):
        joints_vel = [c2c(joint_frame) for joint_frame in joints_vel]
    if points_vel is not None and torch.is_tensor(points_vel[0]):
        points_vel = [c2c(joint_frame) for joint_frame in points_vel]

    mv = MeshViewer(
        pyrender,
        width=imw,
        height=imh,
        use_offscreen=use_offscreen,
        follow_camera=follow_camera,
        camera_intrinsics=camera_intrinsics,
        img_extn=img_extn,
        default_cam_offset=cam_offset,
        default_cam_rot=cam_rot,
    )
    if render_body and render_bodies_static is None:
        mv.add_mesh_seq(body_mesh_seq, progress_bar=progress_bar)
    elif render_body and render_bodies_static is not None:
        mv.add_static_meshes(
            [
                body_mesh_seq[i]
                for i in range(len(body_mesh_seq))
                if i % render_bodies_static == 0
            ]
        )
    if render_joints and render_skeleton:
        mv.add_point_seq(
            joints_seq,
            color=joint_color,
            radius=joint_rad,
            contact_seq=contacts,
            connections=skel_connections,
            connect_color=skel_color,
            vel=joints_vel,
            contact_color=contact_color,
            render_static=render_points_static,
        )
    elif render_joints:
        mv.add_point_seq(
            joints_seq,
            color=joint_color,
            radius=joint_rad,
            contact_seq=contacts,
            vel=joints_vel,
            contact_color=contact_color,
            render_static=render_points_static,
        )

    if vtx_list is not None:
        mv.add_smpl_vtx_list_seq(
            body_mesh_seq, vtx_list, color=[0.0, 0.0, 1.0], radius=0.015
        )

    if points_seq is not None:
        if torch.is_tensor(points_seq[0]):
            points_seq = [c2c(point_frame) for point_frame in points_seq]
        mv.add_point_seq(
            points_seq,
            color=point_color,
            radius=point_rad,
            vel=points_vel,
            render_static=render_points_static,
        )

    if static_meshes is not None:
        mv.set_static_meshes(static_meshes)

    if img_seq is not None:
        mv.set_img_seq(img_seq)

    if mask_seq is not None:
        mv.set_mask_seq(mask_seq)

    if render_ground:
        xyz_orig = None
        if ground_plane is not None:
            if render_body:
                xyz_orig = body_mesh_seq[0].vertices[0, :]
            elif render_joints:
                xyz_orig = joints_seq[0][0, :]
            elif points_seq is not None:
                xyz_orig = points_seq[0][0, :]

        mv.add_ground(
            ground_plane=ground_plane,
            xyz_orig=xyz_orig,
            color0=ground_color0,
            color1=ground_color1,
            alpha=ground_alpha,
        )

    mv.set_render_settings(
        out_path=out_path,
        wireframe=wireframe,
        RGBA=RGBA,
        single_frame=(
            render_points_static is not None or render_bodies_static is not None
        ),
    )  # only does anything for offscreen rendering
    try:
        mv.animate(fps=fps, start=start, end=end, progress_bar=progress_bar, action_colorizer=action_colorizer, title=title)
    except RuntimeError as err:
        print("Could not render properly with the error: %s" % (str(err)))

    del mv