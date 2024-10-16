from model.fp.pnet2_layers.python_modules import pygather_point, pyfarthest_point_sample, pyquery_ball_point
import tensorflow as tf
import numpy as np
import numpy.testing as npt


def test_pyfarthest_point_sample():
    """
    Test the farthest point sampling function.
    """
    input_xyz = tf.convert_to_tensor(
        [
            [
                [0, 0, 0],
                [0, 2, 0],
                [2, 0, 0],
                [2, 2, 1],
                [1, 1, 0],
            ]
        ],
        dtype=tf.float32,
    )

    centroids = pyfarthest_point_sample(1, input_xyz)
    assert centroids.shape == (1, 1)
    npt.assert_array_equal(centroids.numpy(), np.array([[3]]))
    
    centroids = pyfarthest_point_sample(3, input_xyz)
    assert centroids.shape == (1, 3)
    npt.assert_array_equal(centroids.numpy(), np.array([[3, 0, 1]]))
    
    centroids = pyfarthest_point_sample(5, input_xyz)
    assert centroids.shape == (1, 5)
    npt.assert_array_equal(centroids.numpy(), np.array([[3, 0, 1, 2, 4]]))
    
    centroids = pyfarthest_point_sample(9, input_xyz)
    assert centroids.shape == (1, 9)
    npt.assert_array_equal(centroids.numpy(), np.array([[3, 0, 1, 2, 4, 0, 0, 0, 0]]))


def test_pygather_point():
    """
    Test the gather point function.
    """
    input_points = tf.convert_to_tensor(
        [
            [
                [0, 0, 0],
                [0, 2, 0],
                [2, 0, 0],
                [2, 2, 1],
                [1, 1, 0],
            ]
        ],
        dtype=tf.float32,
    )
    input_idx = tf.convert_to_tensor([[3, 0, 1]], dtype=tf.int32)
    output = pygather_point(input_points, input_idx)
    assert output.shape == (1, 3, 3)
    npt.assert_array_equal(output.numpy(), np.array([[[2, 2, 1], [0, 0, 0], [0, 2, 0]]]))
  
    # B = 2
    input_points = tf.convert_to_tensor(
        [
            [
                [0, 0, 0],
                [0, 2, 0],
                [2, 0, 0],
                [2, 2, 1],
                [1, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 2, 0],
                [2, 0, 0],
                [2, 2, 1],
                [1, 1, 0],
            ]
        ],
        dtype=tf.float32,
    )
    input_idx = tf.convert_to_tensor([[1, 4, 0], [3, 0, 2]], dtype=tf.int32)
    output = pygather_point(input_points, input_idx)
    assert output.shape == (2, 3, 3)
    npt.assert_array_equal(output.numpy(), np.array([[[0, 2, 0], [1, 1, 0], [0, 0, 0]], [[2, 2, 1], [0, 0, 0], [2, 0, 0]]]))


def test_pyquery_ball_point():
    """
    Test the query ball point function.
    """
    input_points = tf.convert_to_tensor(
        [
            [
                [0, 0, 0],
                [0, 2, 0],
                [2, 0, 0],
                [2, 2, 1],
                [1, 1, 0],
            ]
        ],
        dtype=tf.float32,
    )
    input_anchors = tf.convert_to_tensor(
        [
            [
                [0, 0, 0]
            ]
        ],
        dtype=tf.float32,
    )
    input_radius = 1.5
    group_idx, pts_cnt = pyquery_ball_point(input_radius, 2, input_points, input_anchors)
    assert group_idx.shape == (1, 1, 2)
    npt.assert_array_equal(group_idx.numpy(), np.array([[[0, 4]]]))
    group_idx, pts_cnt = pyquery_ball_point(input_radius, 4, input_points, input_anchors)
    assert group_idx.shape == (1, 1, 4)
    npt.assert_array_equal(group_idx.numpy(), np.array([[[0, 4, 0, 0]]]))



def pygroup_point(points, idx):
    """Retrieves the points corresponding to the given indices.

    This function gathers the 3D coordinates of the selected points using the provided indices.

    Args:
        points: Tensor of shape [B, N, 3], the original point cloud before grouping.
        idx: Tensor of shape [B, S, nsample], indices of the grouped points from query ball selection.

    Returns:
        grouped_points: Tensor of shape [B, S, nsample, 3], grouped points with their 3D coordinates (F,T,A).
    """
    # Gather the points based on the provided indices
    grouped_points = tf.gather(points, idx, batch_dims=1)

    return grouped_points

def test_pygroup_point():
    """
    Test the group point function.
    """
    input_points = tf.convert_to_tensor(
        [
            [
                [0, 0, 0],
                [0, 2, 0],
                [2, 0, 0],
                [2, 2, 1],
                [1, 1, 0],
            ]
        ],
        dtype=tf.float32,
    )
    input_anchors = tf.convert_to_tensor(
        [
            [
                [0, 0, 0]
            ]
        ],
        dtype=tf.float32,
    )
    input_radius = 1.5
    group_idx, pts_cnt = pyquery_ball_point(input_radius, 2, input_points, input_anchors)
    grouped_points = pygroup_point(input_points, group_idx)
    assert grouped_points.shape == (1, 1, 2, 3)
    npt.assert_array_equal(grouped_points.numpy(), np.array([[[[0, 0, 0], [1, 1, 0]]]]))
    group_idx, pts_cnt = pyquery_ball_point(input_radius, 4, input_points, input_anchors)
    grouped_points = pygroup_point(input_points, group_idx)
    assert grouped_points.shape == (1, 1, 4, 3)
    npt.assert_array_equal(grouped_points.numpy(), np.array([[[[0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]]]]))
  
    # B = 2
    # TODO
    input_points = tf.convert_to_tensor(
        [
            [
                [0, 0, 0],
                [0, 2, 0],
                [2, 0, 0],
                [2, 2, 1],
                [1, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 2, 0],
                [2, 0, 0],
                [2, 2, 1],
                [1, 1, 0],
            ]
        ],
        dtype=tf.float32,
    )
    input_anchors = tf.convert_to_tensor(
        [
            [
                [0, 0, 0]
            ],
            [
                [0, 0, 0]
            ]
        ],
        dtype=tf.float32,
    )
    input_radius = 1.5
    group_idx, pts_cnt = pyquery_ball_point(input_radius, 2, input_points, input_anchors)
    grouped_points = pygroup_point(input_points, group_idx)
    assert grouped_points.shape == (2, 1, 2, 3)
    npt.assert_array_equal(grouped_points.numpy(), np.array([[[[0, 0, 0], [1, 1, 0]]], [[[0, 0, 0], [1, 1, 0]]]]))
    group_idx, pts_cnt = pyquery_ball_point(input_radius, 4, input_points, input_anchors)
    grouped_points = pygroup_point(input_points, group_idx)
    assert grouped_points.shape == (2, 1, 4, 3)
    npt.assert_array_equal(grouped_points.numpy(), np.array([[[[0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]]]]))
