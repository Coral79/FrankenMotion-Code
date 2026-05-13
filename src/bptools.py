import einops
import torch
import numpy as np

##Coverted to our part-level setup
BODY_PARTS_LST = [
    "left arm", "right arm",
    "left leg", "right leg",
    "head", "spine",
    "trajectory",   # new
]


# fmt: off
JOINT_NAMES = {
    "smpljoints": [
        "pelvis",
        "left_hip", "right_hip", "spine1",
        "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3",
        "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hand", "right_hand",
    ]
}
# fmt: on
JOINT_NAMES["guoh3djoints"] = JOINT_NAMES["smpljoints"][:-2]

JOINT_NAMES_IDX = {
    jointstype: {x: i for i, x in enumerate(JOINT_NAMES[jointstype])}
    for jointstype in JOINT_NAMES
}


# fmt: off
MAPPING = {
    "left arm":  ["left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand"],
    "right arm": ["right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand"],
    "left leg":  ["left_hip", "left_knee", "left_ankle", "left_foot"],
    "right leg": ["right_hip", "right_knee", "right_ankle", "right_foot"],
    "head":      ["neck", "head"],
    "spine":     ["spine1", "spine2", "spine3"],
    "trajectory":["pelvis"],   # pelvis-only group
}
# fmt: on

INV_MAPPING = {x: bp for bp in BODY_PARTS_LST for x in MAPPING[bp]}

BODY_PARTS = {
    jointstype: {
        bp: np.array(
            [i for x, i in JOINT_NAMES_IDX[jointstype].items() if INV_MAPPING[x] == bp]
        )
        for bp in BODY_PARTS_LST
    }
    for jointstype in JOINT_NAMES
}

# The indexes are precomputed for faster inference
# see the __main__ below, for computing those

# fmt: off
BP_INDEXES = {
    "smplrifke":
    {
        'left arm': 
        np.array([ 82,  83,  84,  85,  86,  87, 100, 101, 102, 103, 104, 105, 112,
                   113, 114, 115, 116, 117, 124, 125, 126, 127, 128, 129, 172, 173,
                   174, 181, 182, 183, 187, 188, 189, 193, 194, 195, 199, 200, 201]), 
        
        'right arm': 
        np.array([ 88,  89,  90,  91,  92,  93, 106, 107, 108, 109, 110, 111, 118,
                   119, 120, 121, 122, 123, 130, 131, 132, 133, 134, 135, 175, 176,
                   177, 184, 185, 186, 190, 191, 192, 196, 197, 198, 202, 203, 204]), 
        
        'left leg': 
        np.array([ 10,  11,  12,  13,  14,  15,  28,  29,  30,  31,  32,  33,  46,
                   47,  48,  49,  50,  51,  64,  65,  66,  67,  68,  69, 136, 137,
                   138, 145, 146, 147, 154, 155, 156, 163, 164, 165]), 
        
        'right leg': 
        np.array([ 16,  17,  18,  19,  20,  21,  34,  35,  36,  37,  38,  39,  52,
                   53,  54,  55,  56,  57,  70,  71,  72,  73,  74,  75, 139, 140,
                   141, 148, 149, 150, 157, 158, 159, 166, 167, 168]), 
        
        'head': 
        np.array([ 76,  77,  78,  79,  80,  81,  94,  95,  96,  97,  98,  99, 169,
                   170, 171, 178, 179, 180]), 
        
        'spine': 
        np.array([ 22,  23,  24,  25,  26,  27,  40,  41,  42,  43,  44,  45,  58,
                   59,  60,  61,  62,  63, 142, 143, 144, 151, 152, 153, 160, 161,
                   162]), 
        
        'trajectory': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    },
    
    "guoh3dfeats":
    {
        'left arm': 
        np.array([ 40,  41,  42,  49,  50,  51,  55,  56,  57,  61,  62,  63, 139,
                   140, 141, 142, 143, 144, 157, 158, 159, 160, 161, 162, 169, 170,
                   171, 172, 173, 174, 181, 182, 183, 184, 185, 186, 232, 233, 234,
                   241, 242, 243, 247, 248, 249, 253, 254, 255]), 
        
        'right arm': 
        np.array([ 43,  44,  45,  52,  53,  54,  58,  59,  60,  64,  65,  66, 145,
                   146, 147, 148, 149, 150, 163, 164, 165, 166, 167, 168, 175, 176,
                   177, 178, 179, 180, 187, 188, 189, 190, 191, 192, 235, 236, 237,
                   244, 245, 246, 250, 251, 252, 256, 257, 258]), 
        
        'left leg': 
        np.array([  4,   5,   6,  13,  14,  15,  22,  23,  24,  31,  32,  33,  67,
                   68,  69,  70,  71,  72,  85,  86,  87,  88,  89,  90, 103, 104,
                   105, 106, 107, 108, 121, 122, 123, 124, 125, 126, 196, 197, 198,
                   205, 206, 207, 214, 215, 216, 223, 224, 225, 259, 260]), 
        
        'right leg': 
        np.array([  7,   8,   9,  16,  17,  18,  25,  26,  27,  34,  35,  36,  73,
                   74,  75,  76,  77,  78,  91,  92,  93,  94,  95,  96, 109, 110,
                   111, 112, 113, 114, 127, 128, 129, 130, 131, 132, 199, 200, 201,
                   208, 209, 210, 217, 218, 219, 226, 227, 228, 261, 262]), 
        
        'head': 
        np.array([ 37,  38,  39,  46,  47,  48, 133, 134, 135, 136, 137, 138, 151,
                   152, 153, 154, 155, 156, 229, 230, 231, 238, 239, 240]), 
        
        'spine': 
        np.array([ 10,  11,  12,  19,  20,  21,  28,  29,  30,  79,  80,  81,  82,
                   83,  84,  97,  98,  99, 100, 101, 102, 115, 116, 117, 118, 119,
                   120, 202, 203, 204, 211, 212, 213, 220, 221, 222]), 
        
        'trajectory': 
        np.array([  0,   1,   2,   3, 193, 194, 195])
    }
}
# fmt: on


def get_indexes_body_parts(featsname):
    return BP_INDEXES[featsname]


# Was used beforehand
# but it is way faster to compute the indices first
# and only do slicing
# The function below are still included as a reference
# to have a way to generate indices above
def _split_by_body_parts_smplrifke(x):
    assert x.shape[-1] == 205
    body_parts = BODY_PARTS["smpljoints"]

    (
        root_data,
        poses_local_flatten,
        joints_local_flatten,
    ) = einops.unpack(x, [[4], [132], [69]], "b k *")

    poses_local = einops.rearrange(poses_local_flatten, "b k (l t) -> b k l t", t=6)
    joints_local = einops.rearrange(joints_local_flatten, "b k (l t) -> b k l t", t=3)

    output_bp = {}
    for bp in BODY_PARTS_LST:

        # --- NEW: trajectory = root (4) + pelvis rotation (6D from poses_local joint 0) ---
        if bp == "trajectory":
            pelvis_rot6d = einops.rearrange(
                poses_local[:, :, 0:1],  # pelvis is joint 0 in poses_local
                "b k l t -> b k (l t)"
            )
            output_bp[bp] = torch.cat((root_data, pelvis_rot6d), axis=2)  # 4 + 6 = 10
            continue
        # ------------------------------------------------------------------------------

        # offset because of the pelvis
        # only for joints_local (pelvis was dropped there)
        idx = body_parts[bp]
        offset_idx = idx - 1

        # remove left and right arm from the rotations
        if any(idx >= 22):
            assert "arm" in bp
            idx = idx[idx < 22]

        # remove the pelvis index (shouldn't occur now if pelvis maps only to trajectory)
        if any(offset_idx < 0):
            offset_idx = offset_idx[offset_idx >= 0]

        bp_feats = torch.cat(
            (
                einops.rearrange(poses_local[:, :, idx], "b k l t -> b k (l t)"),
                einops.rearrange(joints_local[:, :, offset_idx], "b k l t -> b k (l t)"),
            ),
            axis=2,
        )

        # --- REMOVED: no longer append root_data to legs (root lives in trajectory now) ---
        # if bp == "legs":
        #     bp_feats = torch.cat((bp_feats, root_data), axis=2)

        output_bp[bp] = bp_feats

    assert sum([output_bp[bp].shape[2] for bp in BODY_PARTS_LST]) == 205
    return output_bp



def _split_by_body_parts_guoh3dfeats(x):
    assert x.shape[-1] == 263
    import einops
    import torch

    (
        root_data,
        ric_data,
        rot_data,
        local_velocity,
        foot_contact,
    ) = einops.unpack(x, [[4], [21 * 3], [21 * 6], [22 * 3], [4]], "b k *")

    body_parts = BODY_PARTS["guoh3djoints"]  # 21 joints (pelvis excluded for ric/rot)

    # reshape
    ric_data       = einops.rearrange(ric_data,       "b k (l t) -> b k l t", t=3)
    rot_data       = einops.rearrange(rot_data,       "b k (l t) -> b k l t", t=6)
    local_velocity = einops.rearrange(local_velocity, "b k (l t) -> b k l t", t=3)

    FOOT_CONTACT_SPLIT = {"left leg":[0,1], "right leg":[2,3]}

    output_bp = {}
    for bp in BODY_PARTS_LST:
        # ---- trajectory: root (4) + pelvis velocity (3) ----
        if bp == "trajectory":
            pelvis_vel = einops.rearrange(
                local_velocity[:, :, 0:1],  # pelvis is joint 0 in the 22-joint vel block
                "b k l t -> b k (l t)"
            )  # -> [b, k, 3]
            output_bp[bp] = torch.cat([root_data, pelvis_vel], dim=2)  # 4 + 3 = 7
            continue

        idx = body_parts[bp]
        offset_idx = idx - 1  # ric/rot exclude pelvis

        if any(offset_idx < 0):
            idx = idx[idx >= 1]
            offset_idx = offset_idx[offset_idx >= 0]

        part_chunks = [
            einops.rearrange(ric_data[:, :, offset_idx], "b k l t -> b k (l t)"),
            einops.rearrange(rot_data[:, :, offset_idx], "b k l t -> b k (l t)"),
            einops.rearrange(local_velocity[:, :, idx],  "b k l t -> b k (l t)"),
        ]
        if bp in ("left leg", "right leg"):
            part_chunks.append(foot_contact[:, :, FOOT_CONTACT_SPLIT[bp]])

        output_bp[bp] = torch.cat(part_chunks, dim=2)

    assert sum([output_bp[bp].shape[2] for bp in BODY_PARTS_LST]) == 263
    return output_bp



if __name__ == "__main__":
    smplrifke = {
        x: y.numpy()[0, 0]
        for x, y in _split_by_body_parts_smplrifke(
            torch.arange(205)[None, None]
        ).items()
    }
    print(f"{smplrifke=}")
    print("")

    guoh3dfeats = {
        x: y.numpy()[0, 0]
        for x, y in _split_by_body_parts_guoh3dfeats(
            torch.arange(263)[None, None]
        ).items()
    }
    print(f"{guoh3dfeats=}")
    print("")
