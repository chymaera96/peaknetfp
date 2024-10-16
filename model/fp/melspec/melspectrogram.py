# -*- coding: utf-8 -*-
"""melsprctrogram.py"""    
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda, Permute
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank
import math


class Melspec_layer(Model):
    """
    A wrapper class, based on the implementation:
        https://github.com/keunwoochoi/kapre
        
    Input:
        (B,1,T)
    Output:
        (B,C,T,1) with C=Number of mel-bins
    
    USAGE:
        
        See get_melspec_layer() in the below.
        
    """
    def __init__(
            self,
            input_shape=(1, 8000),
            amplitude_norm=False,
            n_fft=1024,
            stft_hop=256,
            n_mels=256,
            fs=8000,
            dur=1.,
            f_min=300.,
            f_max=4000.,
            amin=1e-5, # minimum amp.
            dynamic_range=80.,
            name='Mel-spectrogram',
            trainable=False,
            **kwargs
            ):
        super(Melspec_layer, self).__init__(name=name, trainable=False, **kwargs)
        
        self.mel_fb_kwargs = {
            'sample_rate': fs,
            'n_freq': n_fft // 2 + 1,
            'n_mels': n_mels,
            'f_min': f_min,
            'f_max': f_max,
            }
        self.n_fft = n_fft
        self.stft_hop = stft_hop
        self.n_mels = n_mels
        self.amin = amin
        self.dynamic_range = dynamic_range
        self.amplitude_norm = amplitude_norm
        
        # 'SAME' Padding layer
        self.pad_l = n_fft // 2
        self.pad_r = n_fft // 2
        self.padded_input_shape = (1, int(fs * dur) + self.pad_l + self.pad_r)
        self.pad_layer = Lambda(
            lambda z: tf.pad(z, tf.constant([[0, 0], [0, 0],
                                             [self.pad_l, self.pad_r]]))
            )
        
        # Construct log-power Mel-spec layer
        self.m = self.construct_melspec_layer(input_shape, name)

        # Permute layer
        self.p = tf.keras.Sequential(name='Permute')
        self.p.add(Permute((3, 2, 1), input_shape=self.m.output_shape[1:]))
        
        super(Melspec_layer, self).build((None, input_shape[0], input_shape[1]))
        
        
    def construct_melspec_layer(self, input_shape, name):
        m = tf.keras.Sequential(name=name)
        m.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        m.add(self.pad_layer)
        m.add(
            STFT(
                n_fft=self.n_fft,
                hop_length=self.stft_hop,
                pad_begin=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                pad_end=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                input_data_format='channels_first',
                output_data_format='channels_first')
            )
        m.add(
            Magnitude()
            )
        m.add(
            ApplyFilterbank(type='mel',
                            filterbank_kwargs=self.mel_fb_kwargs,
                            data_format='channels_first'
                            )
            )
        return m
        

    @tf.function
    def call(self, x):
        # Amplitude Mel-Spectrogram
        x = self.m(x) # (BSZ, n_channels, T, F)
        # Clip x below amin. This is to avoid log(0) in the next step
        x = tf.maximum(x, self.amin)
        # log-power Mel-spectrogram
        # x_ref = tf.reduce_max(x) # Reference is the maximum value of x
        x_ref = tf.reduce_max(x, axis=[-1,-2], keepdims=True)  # Reference is the maximum value along the frequency axis
        x = 20 * tf.math.log(x/x_ref) / math.log(10) # tf.math.log is natural logarithm. We convert to log10 by dividing by ln(10)
        # Clip x below from -dynamic_range dB
        x = tf.maximum(x, -1 * self.dynamic_range)
        # Normalize x to be in [0, 1]
        if self.amplitude_norm:
            x = 1 + (x/self.dynamic_range)
        return self.p(x) # Permute((3,2,1))

def get_melspec_layer(cfg, trainable=False):
    fs = cfg['MODEL']['FS']
    dur = cfg['MODEL']['DUR']
    n_fft = cfg['MODEL']['STFT_WIN']
    stft_hop = cfg['MODEL']['STFT_HOP']
    n_mels = cfg['MODEL']['N_MELS']
    f_min = cfg['MODEL']['F_MIN']
    f_max = cfg['MODEL']['F_MAX']
    amplitude_norm = cfg['MODEL']['NORM_MEL_AMPlITUDE']

    input_shape = (1, int(fs * dur))
    l = Melspec_layer(input_shape=input_shape,
                      amplitude_norm=amplitude_norm,
                      n_fft=n_fft,
                      stft_hop=stft_hop,
                      n_mels=n_mels,
                      fs=fs,
                      dur=dur,
                      f_min=f_min,
                      f_max=f_max)
    l.trainable = trainable
    return l

def get_melspec_layer_data(data, cfg, trainable=False):
    fs = cfg['MODEL']['FS']
    dur = cfg['MODEL']['DUR']
    n_fft = cfg['MODEL']['STFT_WIN']
    stft_hop = cfg['MODEL']['STFT_HOP']
    n_mels = cfg['MODEL']['N_MELS']
    f_min = cfg['MODEL']['F_MIN']
    f_max = cfg['MODEL']['F_MAX']
    amplitude_norm = cfg['MODEL']['NORM_MEL_AMPlITUDE']

    input_shape = (data.shape[1], data.shape[2])

    l = Melspec_layer(input_shape=input_shape,
                      amplitude_norm=amplitude_norm,
                      n_fft=n_fft,
                      stft_hop=stft_hop,
                      n_mels=n_mels,
                      fs=fs,
                      dur=dur,
                      f_min=f_min,
                      f_max=f_max)
    l.trainable = trainable
    return l