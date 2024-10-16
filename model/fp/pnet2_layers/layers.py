import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization

from . import utils


class Pointnet_SA(Layer):
	''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
			sampling: str -- sampling technique
			grouping: str -- knn or queryball
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
	def __init__(
		self, npoint, radius, nsample, mlp, sampling, grouping,
		group_all=False, use_xyz=True, activation=tf.nn.relu, bn=False
	):
		super(Pointnet_SA, self).__init__()
		self.npoint = npoint
		self.radius = radius
		self.nsample = nsample
		self.mlp = mlp
		self.sampling = sampling
		self.grouping = grouping
		self.group_all = group_all
		self.use_xyz = use_xyz
		self.activation = activation
		self.bn = bn
		self.mlp_list = []

	def build(self, input_shape):
		for i, n_filters in enumerate(self.mlp):
			self.mlp_list.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))
		super(Pointnet_SA, self).build(input_shape)

	def call(self, xyz, points, training=True):
		if points is not None:
			if len(points.shape) < 3:
				points = tf.expand_dims(points, axis=0)
		if self.group_all:
			new_xyz, new_points = utils.sample_and_group_all(xyz, points, self.use_xyz)
		else:
			new_xyz, new_points = utils.sample_and_group(
				npoint=self.npoint,
				radius=self.radius,
				nsample=self.nsample,
				xyz=xyz,
				points=points,
				grouping=self.grouping,
				use_xyz=self.use_xyz
			)
		for mlp_layer in self.mlp_list:
			new_points = mlp_layer(new_points, training=training)
		new_points = tf.math.reduce_max(new_points, axis=2, keepdims=True)
		# TODO? check original code for other pooling techniques:
		# https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/utils/pointnet_util.py#L126
		return new_xyz, tf.squeeze(new_points)


class Pointnet_SA_MSG(Layer):
	''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG).
	Input == (N x (d+C)) matrix that is from N points with d-dim coordinates and C-dim
	point feature. It outputs a (N' x (d+C')) matrix of N' subsampled points with d-dim
	coordinates and new C'-dim feature vectors summarizing the local context.

	- Sampling Layer: Farthest point sampling (FPS) is used to choose a subset of points
	such that the last point is the most distant point in metric distance from the set
	of points.
	- Grouping Layer: The input is a set of points (N x (d + C)) and the coordinates of
	the centroids (N' x d). The output are groups of point sets of size (N' x K x (d+C))
	and each group corresponds to a local region and K is the number of points in the
	neighborhood of centroid points. Note that K varies across groups but the suceeding
	PointNet layer is able to convert flexible number of points into a fixed length
	local region feature vector.
	- PointNet Layer: Input are N' local regions of points with data size (N' x K x (d+C)).
	Each region in the output is abstracted by its centroid and local feature that
	encodes the centroid's neighborhood. Output data size is (N' x (d+C')). The coordinates
	of points in local region are firstly translated into a local frame relative to the
	centroid point.
        Input:
            npoint: int32 -- #points sampled in farthest point sampling
            radius_list: list of float32 -- search radius in local region
            nsample_list: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
			activation: tf.nn -- activation function
			bn: bool -- batchnormalization
			sampling: str -- sampling technique
			grouping: str -- knn or queryball
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor -- points selected after the sampling process
            new_points_concat: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor -- 
    '''
	def __init__(
		self, npoint, radius_list, nsample_list, mlp, sampling,
		grouping, use_xyz=True, activation=tf.nn.relu, bn=False
	):
		super(Pointnet_SA_MSG, self).__init__()

		self.npoint = npoint
		self.radius_list = radius_list
		self.nsample_list = nsample_list
		self.mlp = mlp
		self.use_xyz = use_xyz
		self.activation = activation
		self.bn = bn
		self.mlp_list = []
		self.sampling = sampling
		self.grouping = grouping

	def build(self, input_shape):
		for i in range(len(self.radius_list)):
			tmp_list = []
			for n_filters in self.mlp[i]:
				tmp_list.append(utils.Conv2d(n_filters,
								             activation=self.activation,
											 bn=self.bn))
			self.mlp_list.append(tmp_list)
		super(Pointnet_SA_MSG, self).build(input_shape)

	def call(self, xyz, points, training=True):

		if points is not None:
			if len(points.shape) < 3:
				points = tf.expand_dims(points, axis=0)
		
		new_xyz = utils.sample_peaks(xyz, self.npoint, sampling=self.sampling)

		new_points_list = []
		for i, nsample in enumerate(self.nsample_list):
			# nsample = maximum number of points to take within each ball radius.
			radius = 1
			if self.grouping.lower() == "knn":
				_, idx = utils.knn_point(k=nsample,
							 			 xyz1=xyz,
							 			 xyz2=new_xyz)
			elif self.grouping.lower() == "queryball":
				radius = self.radius_list[i]  # radius of each ball to look for points.
				idx, pts_cnt = utils.pyquery_ball_point(radius=radius,
										  			  nsample=nsample,
										  			  xyz1=xyz,
										  		 	  xyz2=new_xyz) # idx are the labels of points assigned to query balls.
			grouped_xyz = utils.pygroup_point(xyz, idx) # points grouped in queryballs. (B,npoints,nsample,3)
			# 1. Insert a tensor with a length 1 axis inserted at index axis (2) to new_xyz. --> (B,npoint,1,3)
			# 2. Take the tensor and duplicate it (tile) until nsample. --> (B,npoint,nsample,3)
			# 3. subtract this new tensor to the grouped one.
   			# This operation performs a normalization and centering operation of grouped_xyz features.
			# print(f'debug, grouped_xyz shape {grouped_xyz.shape}, new_xyz shape {new_xyz.shape}, nsample {nsample}, radius {radius}')
			# print(f'debug, expanded shape: {tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]).shape}')
			grouped_xyz_norm = (grouped_xyz - tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])) / radius

			if points is not None:
				grouped_points = utils.pygroup_point(points, idx)
				if self.use_xyz:
					# Append xyz points to the last dim of grouped_points
					grouped_points = tf.concat([grouped_points, grouped_xyz_norm], axis=-1)
			else:
				grouped_points = grouped_xyz_norm

			for mlp_layer in self.mlp_list[i]:
				#recursive --> keeps adding information on the last dim
				grouped_points = mlp_layer(grouped_points, training=training) # (B, npoints, nsample, MLP[i][-1])
			# gets the maximum value along axis=2 of grouped_points. axis 2 
            # corresponds to the number of peaks selected (nsample). This operation
			# takes the maximum peak values of each group.
			new_points = tf.math.reduce_max(grouped_points, axis=2)
			new_points_list.append(new_points)

		new_points_concat = tf.concat(new_points_list, axis=-1)

		return new_xyz, new_points_concat
