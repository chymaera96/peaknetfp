import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, BatchNormalization
from scipy.spatial.distance import cdist

from .python_modules import (
    pyfarthest_point_sample,
    pygather_point,
    pyquery_ball_point,
    pygroup_point,
)


class UnknownGroupingException(Exception):
    pass


def knn_point(k, xyz1, xyz2):
    """
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    """
    b = tf.shape(xyz1)[0]
    n = tf.shape(xyz1)[1]
    c = tf.shape(xyz1)[2]
    m = tf.shape(xyz2)[1]
    # tile operation replicates xyz1 m times
    xyz1 = tf.tile(tf.reshape(xyz1, (b, 1, n, c)), [1, m, 1, 1])
    # tile operation replicates xyz2 n times
    xyz2 = tf.tile(tf.reshape(xyz2, (b, m, 1, c)), [1, 1, n, 1])
    dist = tf.reduce_sum((xyz1 - xyz2) ** 2, -1)

    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])
    val = tf.slice(out, [0, 0, 0], [-1, -1, k])
    # val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx


@tf.function
def sample_and_group(npoint, radius, nsample, xyz, points, grouping, use_xyz=True):
    """
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        grouping: str -- knn or queryball
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    """
    new_xyz = pygather_point(
        xyz, pyfarthest_point_sample(npoint, xyz)
    )  # (batch_size, npoint, 3)
    if grouping == "knn":
        _, idx = knn_point(k=nsample, xyz1=xyz, xyz2=new_xyz)
    elif grouping == "queryball":
        idx, pts_cnt = pyquery_ball_point(
            radius=radius, nsample=nsample, xyz1=xyz, xyz2=new_xyz
        )
    else:
        raise UnknownGroupingException
    grouped_xyz = pygroup_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(
        tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1]
    )  # translation normalization
    if points is not None:
        grouped_points = pygroup_point(
            points, idx
        )  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat(
                [grouped_xyz, grouped_points], axis=-1
            )  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


@tf.function
def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    """
    batch_size = tf.shape(xyz)[0]
    nsample = tf.shape(xyz)[1]

    new_xyz = tf.zeros((batch_size, 1, 3), dtype=tf.float32)  # (batch_size, 1, 3)
    grouped_xyz = tf.reshape(
        xyz, (batch_size, 1, nsample, 3)
    )  # (batch_size, 1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


@tf.function
def sample_peaks(peaks, npoint, sampling):
    if sampling.lower() == "fps":
        pyfps = pyfarthest_point_sample(npoint, peaks)
        py_new_peaks = pygather_point(peaks, pyfps)
        new_peaks = py_new_peaks
    elif sampling.lower() == "topn":
        # select the topn points. The order of the points is given, amplitude descending
        # for the first layer and for the second PointNetAFP layer is given by
        # the sampling method used in layer 1 (fps or topn).
        new_peaks = peaks[:, :npoint, :]
    return new_peaks


class Conv2d(Layer):
    def __init__(
        self,
        filters,
        strides=[1, 1],
        activation=tf.nn.relu,
        padding="VALID",
        initializer="glorot_normal",
        bn=False,
    ):
        super(Conv2d, self).__init__()

        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.initializer = initializer
        self.bn = bn

    def build(
        self, input_shape
    ):  # Called when calling fingerprinter() with some data for the first time or fingerprinter.build() explicitly
        self.w = self.add_weight(
            shape=(1, 1, input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name="pnet_conv",
        )
        if self.bn:
            self.bn_layer = BatchNormalization()
        super(Conv2d, self).build(input_shape)

    def call(self, inputs, training=True):
        points = tf.nn.conv2d(
            inputs, filters=self.w, strides=self.strides, padding=self.padding
        )
        if self.bn:
            points = self.bn_layer(points, training=training)
        if self.activation:
            points = self.activation(points)
        return points
