# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" nnfp.py

'Neural Audio Fingerprint for High-specific Audio Retrieval based on 
Contrastive Learning', https://arxiv.org/abs/2010.11910

USAGE:
    
    Please see test() in the below.
    
"""
import numpy as np
import tensorflow as tf
assert tf.__version__ >= "2.0"

from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                                     Dropout, ELU, Flatten, LayerNormalization)
from  model.fp.pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG

class ConvLayer(tf.keras.layers.Layer):
    """
    Separable convolution layer
    
    Arguments
    ---------
    hidden_ch: (int)
    strides: [(int, int), (int, int)]
    norm: 'layer_norm1d' for normalization on Freq axis. (default)
          'layer_norm2d' for normalization on on FxT space 
          'batch_norm' or else, batch-normalization
    
    Input
    -----
    x: (B,F,T,1)
    
    [Conv1x3]>>[ELU]>>[BN]>>[Conv3x1]>>[ELU]>>[BN]
    
    Output
    ------
    x: (B,F,T,C) with {F=F/stride, T=T/stride, C=hidden_ch}
    
    """
    def __init__(self,
                 hidden_ch=128,
                 strides=[(1,1),(1,1)],
                 norm='layer_norm2d'):
        super(ConvLayer, self).__init__()
        self.conv2d_1x3 = Conv2D(hidden_ch,
                                 kernel_size=(1, 3),
                                 strides=strides[0],
                                 padding='SAME',
                                 dilation_rate=(1, 1),
                                 kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros')
        self.conv2d_3x1 = Conv2D(hidden_ch,
                                 kernel_size=(3, 1),
                                 strides=strides[1],
                                 padding='SAME',
                                 dilation_rate=(1, 1),
                                 kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros')
        
        if norm == 'layer_norm1d':
            self.BN_1x3 = LayerNormalization(axis=-1)
            self.BN_3x1 = LayerNormalization(axis=-1)
        elif norm == 'layer_norm2d':
            self.BN_1x3 = LayerNormalization(axis=(1, 2, 3))
            self.BN_3x1 = LayerNormalization(axis=(1, 2, 3))
        else:
            self.BN_1x3 = BatchNormalization(axis=-1) # Fix axis: 2020 Apr20
            self.BN_3x1 = BatchNormalization(axis=-1)
            
        self.forward = tf.keras.Sequential([self.conv2d_1x3,
                                            ELU(),
                                            self.BN_1x3,
                                            self.conv2d_3x1,
                                            ELU(),
                                            self.BN_3x1
                                            ])

       
    def call(self, x):
        return self.forward(x)


class DivEncLayer(tf.keras.layers.Layer):
    """
    Multi-head projection a.k.a. 'divide and encode' layer:
        
    • The concept of 'divide and encode' was discovered  in Lai et.al.,
     'Simultaneous Feature Learning and Hash Coding with Deep Neural Networks',
      2015. https://arxiv.org/abs/1504.03410
    • It was also adopted in Gfeller et.al. 'Now Playing: Continuo-
      us low-power music recognition', 2017. https://arxiv.org/abs/1711.10958
    
    Arguments
    ---------
    q: (int) number of slices as 'slice_length = input_dim / q'
    unit_dim: [(int), (int)]
    norm: 'layer_norm1d' or 'layer_norm2d' uses 1D-layer normalization on the feature.
          'batch_norm' or else uses batch normalization. Default is 'layer_norm2d'.

    Input
    -----
    x: (B,1,1,C)
    
    Returns
    -------
    emb: (B,Q)
    
    """
    def __init__(self, q=128, unit_dim=[32, 1], norm='batch_norm'):
        super(DivEncLayer, self).__init__()

        self.q = q
        self.unit_dim = unit_dim
        self.norm = norm
        
        if norm in ['layer_norm1d', 'layer_norm2d']:
            self.BN = [LayerNormalization(axis=-1) for i in range(q)]
        else:
            self.BN = [BatchNormalization(axis=-1) for i in range(q)]
            
        self.split_fc_layers = self._construct_layers() 


    def build(self, input_shape):
        # Prepare output embedding variable for dynamic batch-size 
        self.slice_length = int(input_shape[-1] / self.q)

 
    def _construct_layers(self):
        layers = list()
        for i in range(self.q): # q: num_slices
            layers.append(tf.keras.Sequential([Dense(self.unit_dim[0], activation='elu'),
                                               #self.BN[i],
                                               Dense(self.unit_dim[1])]))
        return layers

 
    @tf.function
    def _split_encoding(self, x_slices):
        """
        Input: (B,Q,S)
        Returns: (B,Q)
        
        """
        out = list()
        for i in range(self.q):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
        return tf.concat(out, axis=1)

    
    def call(self, x): # x: (B,1,1,1024)
        x = tf.reshape(x, shape=[x.shape[0], self.q, -1]) # (B,Q,S); Q=num_slices; S=slice length; (B,128,8 or 16)
        return self._split_encoding(x)
    
    
class FingerPrinter(tf.keras.Model):
    """
    Fingerprinter: 'Neural Audio Fingerprint for High-specific Audio Retrieval
        based on Contrastive Learning', https://arxiv.org/abs/2010.11910
    
    IN >> [Convlayer]x8 >> [DivEncLayer] >> [L2Normalizer] >> OUT 
    
    Arguments
    ---------
    input_shape: tuple (int), not including the batch size
    front_hidden_ch: (list)
    front_strides: (list)
    emb_sz: (int) default=128
    fc_unit_dim: (list) default=[32,1]
    norm: 'layer_norm1d' for normalization on Freq axis. 
          'layer_norm2d' for normalization on on FxT space (default).
          'batch_norm' or else, batch-normalization.
    
    • Note: batch-normalization will not work properly with TPUs.
                    
    
    Input
    -----
    x: (B,F,T,1)
    
        
    Returns
    -------
    emb: (B,Q) 
    
    """
    def __init__(self,
                 cfg,
                 input_shape=(256,32,1),
                 front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                 front_strides=[[(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)]],
                 fc_unit_dim=[32,1]):
        super(FingerPrinter, self).__init__()
        self.front_hidden_ch = front_hidden_ch
        self.front_strides = front_strides
        self.emb_sz = cfg["MODEL"]["EMB_SZ"]
        self.norm = cfg["NEURALFP"]["BN"]
        self.mixed_precision = cfg["TRAIN"]["MIXED_PRECISION"]
        
        # Front (sep-)conv layers
        self.n_clayers = len(front_strides)
        self.front_conv = tf.keras.Sequential(name='ConvLayers')
        # Adjust the last hidden channel to be divisible by emb_sz for div_enc layer
        if ((front_hidden_ch[-1] % self.emb_sz) != 0):
            front_hidden_ch[-1] = ((front_hidden_ch[-1]//self.emb_sz) + 1) * self.emb_sz                
        # Add conv layers
        for i in range(self.n_clayers):
            self.front_conv.add(ConvLayer(hidden_ch=front_hidden_ch[i],
                                          strides=front_strides[i],
                                          norm=self.norm))
        self.front_conv.add(Flatten()) # (B,F',T',C) >> (B,D)
            
        # Divide & Encoder layer
        self.div_enc = DivEncLayer(q=self.emb_sz, unit_dim=fc_unit_dim, norm=self.norm)

        
    @tf.function
    def call(self, inputs):
        x = self.front_conv(inputs) # (B,D) with D = (T/2^4) x last_hidden_ch
        x = self.div_enc(x) # (B,Q)

        # Convert the output to float32 for:
        #   1) avoiding underflow at l2_normalize
        #   2) avoiding underflow at loss calculation
        if self.mixed_precision:
            x = Activation('linear', dtype='float32')(x)

        # L2-normalization of the final embedding
        x = tf.math.l2_normalize(x, axis=1)
        return x


class PointNetAFP(tf.keras.Model):
    """
    'PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space',
    https://doi.org/10.48550/arXiv.1706.02413

    Code adapted from https://github.com/dgriffiths3/pointnet2-tensorflow2

    This contains the abstraction layers that takes a set of points as input and outputs
    a single vector.

    Arguments
    ---------
    bn: (bool) default=False. BatchNormalization.
    activation: (Layer) default=tf.nn.relu. Activation layer.
    mixed_precision: (bool) default=False. Mixed precision for faster training.

    Input
    -----
    x: (BSZ, NUM_PEAKS, 3)

    Returns
    -------
    emb: (BSZ,EMB_SZ)

    """
    def __init__(self, cfg, bn=False, activation=tf.nn.relu):
        super(PointNetAFP, self).__init__()

        self.activation = activation
        self.bn = bn
        self.keep_prob = 0.4
        self.kernel_initializer = 'glorot_normal'
        self.kernel_regularizer = None
        self.mixed_precision = False
        self.emb_sz = cfg['MODEL']['EMB_SZ']

        self.init_network(cfg)


    def init_network(self, cfg):
        self.layer1 = Pointnet_SA_MSG(
            npoint=cfg["POINTNET2"]["LAYER1"]["NPOINT"],
            radius_list=cfg["POINTNET2"]["LAYER1"]["RADIUS_LIST"],
            nsample_list=cfg["POINTNET2"]["LAYER1"]["NSAMPLE_LIST"],
            mlp=cfg["POINTNET2"]["LAYER1"]["MLP"],
            sampling=cfg["POINTNET2"]["SAMPLING"],
            grouping=cfg["POINTNET2"]["GROUPING"],
            activation=self.activation,
            bn = self.bn,
        )

        self.layer2 = Pointnet_SA_MSG(
            npoint=cfg["POINTNET2"]["LAYER2"]["NPOINT"],
            radius_list=cfg["POINTNET2"]["LAYER2"]["RADIUS_LIST"],
            nsample_list=cfg["POINTNET2"]["LAYER2"]["NSAMPLE_LIST"],
            mlp=cfg["POINTNET2"]["LAYER2"]["MLP"],
            sampling=cfg["POINTNET2"]["SAMPLING"],
            grouping=cfg["POINTNET2"]["GROUPING"],
            activation=self.activation,
            bn = self.bn,
        )
        mlp_l3 = cfg["POINTNET2"]["LAYER3"]["MLP"] + [self.emb_sz]
        self.layer3 = Pointnet_SA(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=mlp_l3, #first dim of this has to match last dim of previous layer
            sampling=cfg["POINTNET2"]["SAMPLING"],
            grouping=cfg["POINTNET2"]["GROUPING"],
            group_all=True,
            activation=self.activation,
            bn = self.bn
        )

    @tf.function
    def call(self, xyz):
        xyz, points = self.layer1(xyz, None, training=self.trainable)  # (bsz, npoint, 3), (bsz, npoint, N) -> N sum of last dims of mlps first layer
        xyz, points = self.layer2(xyz, points, training=self.trainable) # (bsz, npoint, 3), (bsz, npoint, M) -> M sum of last dims of mlps second layer
        xyz, points = self.layer3(xyz, points, training=self.trainable) # (bsz, 1, 3), (bsz, self.emb_sz) -> self.emb_sz=last mlp layer3
        # xyz here is a tensor full of 0s.
        emb = tf.reshape(points, (xyz.shape[0], -1))  # (bsz, self.emb_sz)
        # L2-normalization of the final embedding
        emb = tf.math.l2_normalize(emb, axis=1)
        return emb


def get_fingerprinter(cfg, trainable=False):
    """
    Input length : 1s or 2s

    Arguements
    ----------
    cfg : (dict)
        created from the '.yaml' located in /config dicrectory

    Returns
    -------
    <tf.keras.Model> FingerPrinter object
    """

    if cfg["MODEL"]["ARCH"] == "nnfp":
        stft_hop = cfg["MODEL"]["STFT_HOP"]
        n_mels = cfg["MODEL"]["N_MELS"]
        n_channels = 1
        n_samples = cfg["MODEL"]["DUR"] * cfg["MODEL"]["FS"]
        if n_samples // stft_hop != 0:
            n_frames = int(n_samples // stft_hop) + 1
        else:
            n_frames = int(n_samples / stft_hop)
        input_shape = (n_mels, n_frames, n_channels)
        fc_unit_dim = [32, 1]
        m = FingerPrinter(cfg=cfg,
                          input_shape=input_shape,
                          fc_unit_dim=fc_unit_dim)
    elif cfg["MODEL"]["ARCH"] == "pointnet2":
        m = PointNetAFP(cfg=cfg)

    m.trainable = trainable
    return m


def test():
    input_1s = tf.constant(np.random.randn(3,256,32,1), dtype=tf.float32) # BxFxTx1
    fprinter = FingerPrinter(emb_sz=128, fc_unit_dim=[32, 1], norm='layer_norm2d')
    emb_1s = fprinter(input_1s) # BxD

    input_2s = tf.constant(np.random.randn(3,256,63,1), dtype=tf.float32) # BxFxTx1
    fprinter = FingerPrinter(emb_sz=128, fc_unit_dim=[32, 1], norm='layer_norm2d')
    emb_2s = fprinter(input_2s)
    #%timeit -n 10 fprinter(_input) # 27.9ms
"""
NeuralFP stats
Total params: 19,224,576
Trainable params: 19,224,576
Non-trainable params: 0
"""
