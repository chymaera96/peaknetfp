import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from tensorflow.keras.layers import MaxPool1D, Layer


@tf.function
def pygather_point(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: Contains the idx of the selected points for each element
             in the batch. [B, S]
    Return:
        new_points: indexed points data, [B, S, C]
    """
    B = tf.shape(points)[0]
    batch_indices = tf.range(B, dtype=tf.int32)[:, tf.newaxis]  # [B, 1]
    batch_indices = tf.tile(batch_indices, [1, tf.shape(idx)[1]])  # [B, S]
    indices = tf.stack([batch_indices, idx], axis=-1)  # [B, S, 2]
    
    return tf.gather_nd(points, indices)




# @tf.function
def pyfarthest_point_sample(npoint, xyz):
    """
    Farthest point sampling (FPS) is used to choose a subset of points such that the
    last point is the most distant point in metric distance from the set of points.
    Process:
    - Initialization: It starts by selecting the point with the highest amplitude from the entire
        point cloud. This point becomes the first element in your sampled subset.
    - Iterative Selection: In each subsequent iteration, the algorithm finds the
        point in the remaining point cloud that is farthest away from all the points
            already selected in the previous steps.
    - Repeat until npoint number of points have been selected. If there are less points
        in the cloud that npoint, it duplicates the first point until npoints.
    Args:
            npoint: (int) number of points
            xyz: (batch_size,num_points,3) shape
    Returns:
            (batch_size, npoint)
    """

    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = tf.shape(xyz)[0], tf.shape(xyz)[1], tf.shape(xyz)[2]
    centroids = tf.TensorArray(dtype=tf.int32, size=npoint, dynamic_size=False, clear_after_read=False)
    distance = tf.fill([B, N], 1e10)
    
    # Select the point with the highest amplitude (largest norm)
    farthest = tf.argmax(xyz[:,:,2], axis=-1, output_type=tf.int32)
    
    batch_indices = tf.range(B, dtype=tf.int32)

    for i in tf.range(npoint):
        centroids = centroids.write(i, farthest)
        indices = tf.stack([batch_indices, farthest], axis=1)
        centroid = tf.gather_nd(xyz, indices)
        centroid = tf.expand_dims(centroid, axis=1)  # [B, 1, 3]
        dist = tf.reduce_sum(tf.square(xyz - centroid), axis=-1)
        
        distance = tf.where(dist < distance, dist, distance)  # Reassign distance
        farthest = tf.argmax(distance, axis=-1, output_type=tf.int32)
    
    return tf.transpose(centroids.stack(), perm=[1, 0])  # [B, npoint]


def pysquare_distance(src, dst):
    """
    Calculate Euclidean squared distance between each pair of points.
    
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: squared distances, [B, N, M]
    """
    B = tf.shape(src)[0]
    N = tf.shape(src)[1]
    M = tf.shape(dst)[1]

    dist = -2 * tf.linalg.matmul(src, dst, transpose_b=True)
    dist += tf.reshape(tf.reduce_sum(tf.square(src), axis=-1), [B, N, 1])
    dist += tf.reshape(tf.reduce_sum(tf.square(dst), axis=-1), [B, 1, M])

    return dist


@tf.function
def pyquery_ball_point(radius, nsample, xyz1, xyz2):
    """
    Finds all points that are within a radius to the query point (with an upper
    limit of K). ball query's local neighborhood guarantees a fixed region
    scale thus making local feature more generalizable across space.
    When the number of points to select is lower than nsample, the idx array is
    filled with	duplicates. This ensures valid indices even if there are fewer
    points than nsample within the radius.
    for example: idx[1,80,:] is [ 3,  4, 13, 33, 89,  3,  3,  3]. 3 being the
                 duplicate idx

    Args:
            radius: Radius of the sphere around each point in xyz2 to search for points in xyz1
            nsample: maximum number of points to find within the radius for each point
            xyz1: Points before farthest sampling (input to layer). [B, N, 3]
            xyz2: Points after farthest sampling. [B, S, 3]
    Returns:
            idx (batch_size,npoints,nsample): Idx of the points selected
            pts_cnt (batch_size,npoints): Number of points selected for each point in xyz2
    """
    B, N, _ = tf.shape(xyz1)[0], tf.shape(xyz1)[1], tf.shape(xyz1)[2]
    S = tf.shape(xyz2)[1]
   
    # Compute squared distances [B, S, N]
    sqrdists = pysquare_distance(xyz2, xyz1)  # Uses the optimized function we created

    # Mask out points that are outside the radius
    mask = sqrdists > (radius ** 2)

    # Initialize indices: [B, S, N] with point indices from 0 to N-1
    group_idx = tf.tile(tf.range(N, dtype=tf.int32)[tf.newaxis, tf.newaxis, :], [B, S, 1])
    group_idx = tf.where(mask, N, group_idx)  # Replace invalid indices with N

    # Sort indices along the last dimension (sort by distance)
    group_idx = tf.sort(group_idx, axis=-1)[:, :, :nsample]  # Keep only nsample points

    # Get first valid index for each query point to use for duplicates
    group_first = tf.tile(group_idx[:, :, 0:1], [1, 1, nsample])

    # Replace invalid indices (N) with the first valid index
    mask = group_idx == N
    group_idx = tf.where(mask, group_first, group_idx)

    # Count valid points (not equal to N)
    pts_cnt = tf.reduce_sum(tf.cast(group_idx != N, tf.int32), axis=-1)

    return group_idx, pts_cnt


@tf.function
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
