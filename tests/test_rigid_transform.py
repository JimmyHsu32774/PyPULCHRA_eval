import numpy as np
from pypulchra.function import fit_rigid_transform, make_local_frame

# Author: Yitian
def test_rigid_transform_notebook_cases():
    # Y5 tests: the public example plus boundary cases.
    example_moving = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    example_target = example_moving + np.array([2.0, 3.0, 4.0])
    example_R, example_t = fit_rigid_transform(example_moving, example_target)
    assert np.allclose(example_moving @ example_R + example_t, example_target)
    assert np.allclose(example_R.T @ example_R, np.eye(3))
    assert np.isclose(np.linalg.det(example_R), 1.0)
    
    angle = np.radians(37.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                       [0.0, 1.5, 0.0], [0.5, 0.5, 2.0]])
    target = points @ rotation.T + np.array([3.0, -1.0, 2.0])
    R, t = fit_rigid_transform(points, target)
    assert np.allclose(points @ R + t, target, atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R.T @ R, np.eye(3))
    
    # a mirrored copy cannot be matched by a proper rotation: det stays +1, residual stays large
    mirror = points * np.array([-1.0, 1.0, 1.0])
    R_mirror, t_mirror = fit_rigid_transform(points, mirror)
    residual = np.sqrt(np.mean(np.sum((points @ R_mirror + t_mirror - mirror) ** 2, axis=1)))
    assert np.isclose(np.linalg.det(R_mirror), 1.0)
    assert residual > 0.5
    
    # collinear points, NaN coordinates and a shape mismatch are rejected
    collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    for moving, reference in ((collinear, collinear),
                              (points, np.full((4, 3), np.nan)),
                              (points, points[:3])):
        try:
            fit_rigid_transform(moving, reference)
        except ValueError:
            pass
        else:
            raise AssertionError('expected ValueError')
    
    # the function must not modify its inputs
    moving_copy = points.copy()
    target_copy = target.copy()
    fit_rigid_transform(points, target)
    assert np.array_equal(points, moving_copy) and np.array_equal(target, target_copy)
    print('Y5: example and boundary tests passed')
