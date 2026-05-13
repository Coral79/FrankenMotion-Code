import numpy as np
import torch
from .humor_render_tools.tools import viz_smpl_seq, ActionColorizer
from smplx.utils import Struct
from .video import Video
import os
from tqdm import tqdm

# os.environ["PYOPENGL_PLATFORM"] = "egl"

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
FACE_PATH = os.path.join(THIS_FOLDER, "humor_render_tools/smplh.faces")
FACES = torch.from_numpy(np.int32(np.load(FACE_PATH)))

# Default path for SMPL segmentation file - you can modify this path
SEGMENTATION_FILE = os.path.join(THIS_FOLDER, "humor_render_tools/smpl_vert_segmentation.json")

class HumorRenderer:
    def __init__(self, fps=20.0, colored_parts=False, segmentation_file=None, **kwargs):
        """
        Initialize the HumorRenderer with options for body part coloring.
        
        Args:
            fps: Frames per second for rendering
            colored_parts: Whether to color different body parts with different colors
            segmentation_file: Path to the SMPL segmentation file (defaults to included file)
            **kwargs: Additional rendering parameters to pass to viz_smpl_seq
        """
        self.kwargs = kwargs
        self.fps = fps
        self.colored_parts = colored_parts
        
        # Use custom segmentation file if provided, otherwise use default
        self.segmentation_file = segmentation_file if segmentation_file else SEGMENTATION_FILE
        
        # Check if segmentation file exists
        if self.colored_parts and not os.path.exists(self.segmentation_file):
            print(f"Warning: SMPL segmentation file not found at {self.segmentation_file}")
            print("Body part coloring will be disabled")
            self.colored_parts = False

    def __call__(self, vertices, output, annotations=None, **kwargs):
        """
        Render the SMPL sequence with optional body part coloring.
        
        Args:
            vertices: SMPL vertices to render
            output: Output path for the rendered video
            annotations: Dictionary of annotations for dynamic coloring
            **kwargs: Additional rendering parameters
        """
        params = self.kwargs | kwargs
        fps = self.fps
        if "fps" in params:
            fps = params.pop("fps")
            
        # Add segmentation parameters
        params["colored_parts"] = kwargs.get("colored_parts", self.colored_parts)
        params["segmentation_file"] = kwargs.get("segmentation_file", self.segmentation_file)
        
        # Process annotations if provided for dynamic coloring
        if annotations and params["colored_parts"]:
            # Convert the input format to the expected format
            processed_annotations = {}
            duration = 0
            
            for body_part, action_list in annotations.items():
                if body_part != 'len()':  # Skip the len() entry
                    processed_annotations[body_part] = []
                    current_time = 0
                    
                    for action_tuple in action_list:
                        action_name = action_tuple[0]
                        action_duration = action_tuple[1]
                        
                        # Create annotation entry with proper format
                        processed_annotations[body_part].append({
                            "text": action_name,
                            "start": current_time,
                            "end": current_time + action_duration,
                            "confidence": 5 if action_name != "unknown" else 1,
                            "source": "input_annotation"
                        })
                        
                        current_time += action_duration
                        duration = max(duration, current_time)
                        
            # Add total duration
            processed_annotations["duration"] = duration
            
            # Create action colorizer with the processed annotations
            action_colorizer = ActionColorizer(processed_annotations)
            params["action_colorizer"] = action_colorizer
        
        render(vertices, output, fps, **params)


def render(vertices, out_path, fps, progress_bar=tqdm, colored_parts=False, 
           segmentation_file=None, action_colorizer=None, **kwargs):
    """
    Render SMPL sequence with optional body part coloring.
    
    Args:
        vertices: SMPL vertices to render
        out_path: Output path for the rendered video
        fps: Frames per second for rendering
        progress_bar: Progress bar function
        colored_parts: Whether to color different body parts with different colors
        segmentation_file: Path to the SMPL segmentation file
        action_colorizer: ActionColorizer instance for dynamic coloring
        **kwargs: Additional rendering parameters
    """
    # Put the vertices at the floor level
    ground = vertices[..., 2].min()
    vertices[..., 2] -= ground

    import pyrender

    # remove title if it exists ##To Change
    # kwargs.pop("title", None)

    # Create output folder
    out_folder = os.path.splitext(out_path)[0]

    if isinstance(vertices, np.ndarray):
        verts = torch.from_numpy(vertices)
    else:
        verts = vertices
        
    body_pred = Struct(v=verts, f=FACES)

    # Add the coloring parameters to kwargs
    if colored_parts:
        kwargs["colored_parts"] = colored_parts
        kwargs["segmentation_file"] = segmentation_file
        
        # Add action colorizer if available
        if action_colorizer:
            kwargs["action_colorizer"] = action_colorizer

    # Call the visualization function
    viz_smpl_seq(
        pyrender, out_folder, body_pred, fps=fps, progress_bar=progress_bar, **kwargs
    )

    video = Video(out_folder, fps=fps)
    video.save(out_path)
    video.save(out_path)