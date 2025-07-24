import numpy as np


def lookat_to_extr(eye, target, up):
    # Normalize vectors
    def normalize(v):
        return v / np.maximum(np.linalg.norm(v), np.finfo(v.dtype).eps)

    # Calculate forward, right, and up vectors
    f = normalize(target - eye)
    r = normalize(np.cross(f, up))
    u = normalize(np.cross(r, f))

    # Create the rotation matrix
    R = np.column_stack([r, u, -f])

    # Construct the extrinsic matrix - note the use of .T for correct orientation
    extr = np.identity(4)
    extr[:3, :3] = R.T
    extr[:3, 3] = np.dot(R.T, -eye)

    return extr


def extr_to_lookat(extrinsic_matrix):
    # Extract rotation matrix and translation vector
    R = extrinsic_matrix[:3, :3].T  # Transpose back to get original R
    T = extrinsic_matrix[:3, 3]

    # Calculate eye position
    eye = -np.dot(R, T)

    # Reconstruct forward, right, and up vectors
    r = R[:, 0]
    u = R[:, 1]
    f = -R[:, 2]

    # Target is a point along the forward direction from the eye
    target = eye + f
    # Up vector is simply the up vector used to construct R
    up = u

    return eye.copy(), target.copy(), up.copy()


def flip_cam_extr(cam_extr):
    eye, target, up = extr_to_lookat(cam_extr)
    # print(eye, target, up)
    # up_point = eye + up
    eye[0] = -eye[0]
    target[0] = -target[0]
    up[0] = -up[0]
    res = lookat_to_extr(eye, target, up)
    return res