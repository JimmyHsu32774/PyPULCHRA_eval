import numpy as np
from pypulchra.function import fit_rigid_transform, make_local_frame

# Author: Yitian
def test_local_frame_notebook_cases():
    # Y6 tests: the public example plus boundary cases.
    example_origin, example_axes = make_local_frame([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert np.allclose(example_origin, [0.0, 0.0, 0.0])
    assert np.allclose(example_axes, np.eye(3))
    assert np.isclose(np.linalg.det(example_axes), 1.0)
    
    origin, axes = make_local_frame([1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 3.0, 3.0])
    assert np.allclose(origin, [1.0, 2.0, 3.0])
    assert np.allclose(axes, np.eye(3))
    assert np.isclose(np.linalg.det(axes), 1.0)
    
    # a rotated/translated frame stays orthonormal and keeps local distances
    origin, axes = make_local_frame([-4.0, 2.0, 1.0], [-2.0, 2.0, 3.0], [-4.0, 4.0, 1.0])
    assert np.allclose(axes[:, 0], np.array([2.0, 0.0, 2.0]) / np.sqrt(8.0))
    assert np.allclose(axes.T @ axes, np.eye(3))
    assert np.isclose(np.linalg.det(axes), 1.0)
    local = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mapped = local @ axes.T + origin
    assert np.isclose(np.linalg.norm(mapped[1] - mapped[0]), 1.0)
    assert np.isclose(np.linalg.norm(mapped[2] - mapped[0]), 1.0)
    assert np.isclose(np.dot(mapped[1] - origin, mapped[2] - origin), 0.0)
    
    # zero forward vector, collinear points and NaN are rejected
    for arguments in (([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                      ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
                      ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, np.nan, 0.0])):
        try:
            make_local_frame(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError('expected ValueError')
    
    # the returned origin is a copy, not a view of the input
    input_origin = np.array([1.0, 1.0, 1.0])
    returned, axes = make_local_frame(input_origin, [2.0, 1.0, 1.0], [1.0, 2.0, 1.0])
    returned[0] = 99.0
    assert input_origin[0] == 1.0
    print('Y6: example and boundary tests passed')
