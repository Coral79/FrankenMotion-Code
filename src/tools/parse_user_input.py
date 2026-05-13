import os
import json
import logging
from typing import Dict, List, Any, Tuple
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def parse_input_file(filepath: str) -> Dict[str, Any]:
    """
    Auto-detect file format (JSON or TXT) and parse.
    
    Args:
        filepath: Path to input file
        
    Returns:
        Parsed input dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return parse_json_format(data)
    elif ext == '.txt':
        with open(filepath, 'r') as f:
            text = f.read()
        return parse_txt_format(text)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .json or .txt")


def parse_json_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse JSON format input.
    
    Expected structure:
    {
        "duration": 5.0,
        "body_parts": {
            "left_arm": [{"start": 0.0, "end": 2.0, "text": "wave"}, ...],
            ...
        },
        "sequence_caption": "..." (optional)
    }
    """
    if "duration" not in data:
        raise ValueError("Input must contain 'duration' field")
    
    if "body_parts" not in data:
        raise ValueError("Input must contain 'body_parts' field")
    
    return data


def parse_txt_format(text: str) -> Dict[str, Any]:
    """
    Parse TXT format input.
    
    Expected structure:
    duration: 5.0
    
    [left_arm]
    0.0-2.0: wave
    2.0-5.0: raise
    
    [sequence_caption]
    A person waves
    """
    lines = [line.strip() for line in text.split('\n')]
    
    result = {
        "body_parts": {}
    }
    
    current_section = None
    
    for line in lines:
        # Skip empty lines
        if not line:
            continue
        
        # Parse duration
        if line.startswith("duration:"):
            result["duration"] = float(line.split(":", 1)[1].strip())
            continue
        
        # Parse section headers [section_name]
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue
        
        # Parse section content
        if current_section:
            if current_section == "sequence_caption":
                # Sequence caption is just text, not time-based
                result["sequence_caption"] = line
            else:
                # Parse time-based annotation: "0.0-2.0: wave"
                if ":" not in line:
                    logger.warning(f"Skipping malformed line: {line}")
                    continue
                
                time_range, text_label = line.split(":", 1)
                time_range = time_range.strip()
                text_label = text_label.strip()
                
                if "-" not in time_range:
                    logger.warning(f"Skipping malformed time range: {time_range}")
                    continue
                
                start_str, end_str = time_range.split("-", 1)
                start = float(start_str.strip())
                end = float(end_str.strip())
                
                # Add to body_parts
                if current_section not in result["body_parts"]:
                    result["body_parts"][current_section] = []
                
                result["body_parts"][current_section].append({
                    "start": start,
                    "end": end,
                    "text": text_label
                })
    
    if "duration" not in result:
        raise ValueError("Input must contain 'duration' field")
    
    return result


def validate_against_model(raw_input: Dict[str, Any], cfg: DictConfig) -> None:
    """
    Validate that user input is compatible with model configuration.
    Unsupported body parts are ignored with a warning.
    
    Args:
        raw_input: Parsed user input
        cfg: Model configuration
    """
    # Get supported body parts from config
    supported_body_parts = cfg.data.text_encoder.body_part_order
    
    # Check if user provided unsupported body parts and remove them
    unsupported_parts = []
    for part in list(raw_input["body_parts"].keys()):  # Use list() to avoid runtime error during iteration
        if part not in supported_body_parts:
            unsupported_parts.append(part)
            del raw_input["body_parts"][part]
    
    if unsupported_parts:
        logger.warning(
            f"Model doesn't support body parts: {unsupported_parts}. "
            f"Ignoring them. Supported parts: {supported_body_parts}"
        )
    
    # Check sequence_caption support
    has_sequence_caption = (
        hasattr(cfg.data.text_encoder, 'load_global') and 
        cfg.data.text_encoder.load_global and 
        hasattr(cfg.data.text_encoder, 'load_global_type') and
        cfg.data.text_encoder.load_global_type == "caption"
    )
    
    if "sequence_caption" in raw_input and not has_sequence_caption:
        logger.warning(
            "Model doesn't support sequence_caption (load_global not enabled or "
            "load_global_type != 'caption'). Ignoring sequence_caption."
        )
        del raw_input["sequence_caption"]
    
    # Check if any body parts remain after filtering
    if not raw_input["body_parts"]:
        raise ValueError(
            f"No supported body parts found in input. "
            f"Supported parts: {supported_body_parts}"
        )
    
    logger.info(f"Input validation passed. Body parts to use: {list(raw_input['body_parts'].keys())}")

def fill_time_gaps(annotations: List[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
    """
    Fill time gaps in annotations with "unknown" labels.
    
    Args:
        annotations: List of {"start": float, "end": float, "text": str}
        duration: Total duration in seconds
        
    Returns:
        Filled annotations with no time gaps
    """
    if not annotations:
        return [{"start": 0.0, "end": duration, "text": "unknown"}]
    
    # Sort by start time
    sorted_anns = sorted(annotations, key=lambda x: x['start'])
    filled = []
    
    current_time = 0.0
    for ann in sorted_anns:
        # Gap before this annotation
        if ann['start'] > current_time:
            filled.append({
                "start": current_time,
                "end": ann['start'],
                "text": "unknown"
            })
        
        filled.append(ann)
        current_time = ann['end']
    
    # Gap at the end
    if current_time < duration:
        filled.append({
            "start": current_time,
            "end": duration,
            "text": "unknown"
        })
    
    return filled


def merge_consecutive_unknown(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments with "unknown" label.
    
    Args:
        annotations: List of annotations
        
    Returns:
        Annotations with consecutive unknowns merged
    """
    if not annotations:
        return annotations
    
    merged = []
    i = 0
    
    while i < len(annotations):
        if annotations[i]['text'].lower() == 'unknown':
            # Find consecutive unknowns
            start = annotations[i]['start']
            end = annotations[i]['end']
            i += 1
            
            while i < len(annotations) and annotations[i]['text'].lower() == 'unknown':
                end = annotations[i]['end']
                i += 1
            
            merged.append({"start": start, "end": end, "text": "unknown"})
        else:
            merged.append(annotations[i])
            i += 1
    
    return merged


def fill_and_merge_annotations(raw_input: Dict[str, Any], cfg: DictConfig, fps: int) -> Dict[str, Any]:
    """
    Fill missing body parts and time gaps, then merge consecutive unknowns.
    
    Args:
        raw_input: Parsed user input
        cfg: Model configuration
        fps: Frames per second
        
    Returns:
        Filled input with all body parts and time periods covered
    """
    duration = raw_input["duration"]
    supported_body_parts = cfg.data.text_encoder.body_part_order
    
    filled_input = {
        "duration": duration,
        "body_parts": {}
    }
    
    # Process each supported body part
    for part in supported_body_parts:
        if part in raw_input["body_parts"]:
            # Fill time gaps
            annotations = raw_input["body_parts"][part]
            filled = fill_time_gaps(annotations, duration)
            # Merge consecutive unknowns
            merged = merge_consecutive_unknown(filled)
            filled_input["body_parts"][part] = merged
        else:
            # Body part not mentioned - fill entire duration with unknown
            filled_input["body_parts"][part] = [
                {"start": 0.0, "end": duration, "text": "unknown"}
            ]
    
    # Copy sequence_caption if present
    if "sequence_caption" in raw_input:
        filled_input["sequence_caption"] = raw_input["sequence_caption"]
    
    return filled_input


def convert_to_annotation_format(filled_input: Dict[str, Any], fps: int) -> Dict[str, Any]:
    """
    Convert filled input to annotation format compatible with TextEmbeddings.load_from_annotation.
    
    Args:
        filled_input: Filled user input
        fps: Frames per second
        
    Returns:
        Annotation dict in format:
        {
            "path": "user_input",
            "start": 0.0,
            "end": duration,
            "annotations": [
                {"bodypart": "left_arm", "start": 0.0, "end": 2.0, "text": "wave"},
                ...
            ]
        }
    """
    duration = filled_input["duration"]
    
    annotation_dict = {
        "path": "user_input",
        "start": 0.0,
        "end": duration,
        "annotations": []
    }
    
    # Flatten body_parts dict to annotations list
    for bodypart, segments in filled_input["body_parts"].items():
        for segment in segments:
            annotation_dict["annotations"].append({
                "bodypart": bodypart,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
    
    # Add sequence_caption as a special annotation if present
    if "sequence_caption" in filled_input:
        annotation_dict["annotations"].append({
            "bodypart": "sequence_caption",
            "start": 0.0,
            "end": duration,
            "text": filled_input["sequence_caption"]
        })
        annotation_dict["sequence_caption"] = filled_input["sequence_caption"]
    
    return annotation_dict


def parse_and_validate_user_input(filepath: str, cfg: DictConfig, fps: int = 20) -> Dict[str, Any]:
    """
    Main entry point: parse, validate, fill, and convert user input.
    
    Args:
        filepath: Path to input file (JSON or TXT)
        cfg: Model configuration
        fps: Frames per second (default: 20)
        
    Returns:
        Annotation dict compatible with TextEmbeddings.load_from_annotation
    """
    logger.info(f"Parsing user input from: {filepath}")
    
    # Parse file
    raw_input = parse_input_file(filepath)
    
    # Validate against model
    validate_against_model(raw_input, cfg)
    
    # Fill gaps and merge unknowns
    filled_input = fill_and_merge_annotations(raw_input, cfg, fps)
    
    # Convert to annotation format
    annotation_dict = convert_to_annotation_format(filled_input, fps)
    
    logger.info(
        f"Successfully parsed user input: duration={filled_input['duration']}s, "
        f"body_parts={list(filled_input['body_parts'].keys())}"
    )
    
    return annotation_dict