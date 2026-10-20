"""Conversion between the OpenCV and USD camera axis conventions.

Every extrinsic this project measures is OpenCV-convention by construction:
``cv2.calibrateHandEye`` returns the transform of a frame whose axes are
``+X right, +Y down, +Z forward`` (forward being the direction the lens looks),
because that is the frame ``cv2.projectPoints`` and the intrinsic matrix are
defined in. Composing it with forward kinematics does not change that.

RoboICL asks for the other one. ``train_calibration_profile`` pins
``train_camera_convention`` to ``camera_to_world_usd``, and the model prompt
tells the policy to map a world point into the camera with::

    u = T_world_to_camera @ p
    v = [u.x, -u.y, -u.z]      # v.z > 0 means "in front of the camera"

That formula only holds for USD axes: ``+X right, +Y up, -Z forward``. Feeding
it an OpenCV extrinsic negates the two axes that were already correct, so every
point on the table comes back with ``v.z < 0`` -- the policy concludes the whole
workspace is behind the camera.

The two conventions differ by a 180-degree rotation about the camera's own X
axis, so the fix is a right-multiplication::

    T_usd = T_opencv @ diag(1, -1, -1, 1)

Right, not left: the flip is expressed in the camera frame, not the world frame.
The translation column is untouched (both conventions put the origin at the
optical centre) and the rotation stays proper -- ``det(diag(1, -1, -1)) == 1`` --
so the result still passes RoboICL's orthonormality check. That check is also
the reason this bug is worth a module rather than an inline matrix multiply:
``camera_arrays`` validates that the extrinsic is a rigid transform, which a
wrongly-converted one still is, so nothing downstream raises. The error is
silent and shows up only as a policy that reasons about space incorrectly.

The live ``get_world_pose`` shim converts at the boundary where it declares the
USD convention expected by RoboICL.
"""

from __future__ import annotations

import numpy as np

# Flips Y and Z, expressed in the camera frame. Its own inverse.
OPENCV_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])


def opencv_to_usd(transform) -> np.ndarray:
    """Convert an OpenCV-convention camera-to-world transform to USD axes.

    Accepts anything array-like of shape ``(4, 4)`` and returns a new array;
    the input is never modified.
    """
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    return matrix @ OPENCV_TO_USD


# The transform is an involution, so one direction covers both. The alias
# exists so call sites can name the direction they mean.
usd_to_opencv = opencv_to_usd
