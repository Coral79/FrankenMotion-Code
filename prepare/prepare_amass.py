"""
Prepare AMASS motion data for FrankenMotion training.

This script runs the full preprocessing pipeline:
1. Fix FPS to 20.0 and remove hand parameters
2. Mirror motions for data augmentation
3. Extract SMPL joint positions
4. Compute SMPL-RiFKE features (205-dimensional representation)

Prerequisites:
- Download AMASS dataset (SMPL-H G format) from https://amass.is.tue.mpg.de/
- Place the raw data in datasets/motions/AMASS/
- Download SMPL-H model files and place in deps/smplh/

Usage:
    python prepare/prepare_amass.py [--amass_dir PATH] [--output_dir PATH] [--smplh_dir PATH]
"""

import os
import sys
import argparse

# Add amasstools to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "amasstools"))

from fix_fps import fix_fps
from smpl_mirroring import mirror_smpl
from extract_joints import extract_joints
from get_smplrifke import get_features


def main():
    parser = argparse.ArgumentParser(description="Prepare AMASS data for FrankenMotion")
    parser.add_argument(
        "--amass_dir",
        type=str,
        default="datasets/motions/AMASS",
        help="Path to raw AMASS dataset (SMPL-H G format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/motions",
        help="Base output directory for processed data",
    )
    parser.add_argument(
        "--smplh_dir",
        type=str,
        default="deps/smplh",
        help="Path to SMPL-H model files",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Target FPS (default: 20.0)",
    )
    parser.add_argument(
        "--force_redo",
        action="store_true",
        help="Force reprocessing of existing files",
    )
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fps_name = f"AMASS_{args.fps}_fps_nh"
    fps_dir = os.path.join(args.output_dir, fps_name)
    joints_name = f"{fps_name}_smpljoints_neutral_nobetas"
    joints_dir = os.path.join(args.output_dir, joints_name)
    rifke_name = f"{fps_name}_smplrifke"

    # Step 1: Fix FPS and remove hands
    print("=" * 60)
    print("Step 1: Fix FPS and remove hand parameters")
    print("=" * 60)
    fix_fps(args.amass_dir, fps_dir, args.fps, force_redo=args.force_redo)

    # Step 2: Mirror motions
    print("\n" + "=" * 60)
    print("Step 2: Mirror motions for data augmentation")
    print("=" * 60)
    mirror_dir = os.path.join(fps_dir, "M")
    mirror_smpl(fps_dir, mirror_dir, force_redo=args.force_redo)

    # Step 3: Extract joints
    print("\n" + "=" * 60)
    print("Step 3: Extract SMPL joint positions")
    print("=" * 60)
    extract_joints(
        base_folder=fps_dir,
        new_base_folder=joints_dir,
        smplh_folder=args.smplh_dir,
        jointstype="smpljoints",
        batch_size=4096,
        gender="neutral",
        use_betas=False,
        device=device,
        force_redo=args.force_redo,
    )

    # Step 4: Compute SMPL-RiFKE features
    print("\n" + "=" * 60)
    print("Step 4: Compute SMPL-RiFKE features (205-dim)")
    print("=" * 60)
    get_features(
        folder=args.output_dir,
        base_name=fps_name,
        joints_name=joints_name,
        new_name=rifke_name,
        force_redo=args.force_redo,
    )

    print("\n" + "=" * 60)
    print("Done! Processed data is in:")
    print(f"  {os.path.join(args.output_dir, rifke_name)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
