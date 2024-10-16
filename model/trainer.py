# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" trainer.py """
import tensorflow as tf
import random
import numpy as np
import os
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import sys

from tensorflow.keras.utils import Progbar
from tensorflow.keras.mixed_precision import Policy, set_global_policy, LossScaleOptimizer

from model.dataset import Dataset
from model.fp.melspec.melspectrogram import get_melspec_layer, get_melspec_layer_data
from model.fp.specaug_chain.specaug_chain import get_specaug_chain_layer
from model.fp.nnfp import get_fingerprinter
from model.fp.NTxent_loss_single_gpu import NTxentLoss
from model.fp.online_triplet_loss import OnlineTripletLoss
from model.fp.lamb_optimizer import LAMB
from model.utils.experiment_helper import ExperimentHelper
from model.utils.mini_search_subroutines import mini_search_eval


SEED = 13
PADDING_VALUE = -1.0

def set_seed(seed: int = SEED):
    """Set seed for reproducibility."""
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set as {seed}")

def build_fp(cfg):
    """ Build fingerprinter """
    # melspec_layer: log-power-Mel-spectrogram layer, S.
    melspec_layer = get_melspec_layer(cfg, trainable=False)

    # specaug_layer: spec-augmentation layer.
    specaug_layer = get_specaug_chain_layer(cfg, trainable=False)
    assert(specaug_layer.bypass==False) # Detachable by setting specaug_layer.bypass.

    # Enable mixed precision after specaug is built.
    if cfg["TRAIN"]["MIXED_PRECISION"]:
        set_global_policy(Policy('mixed_float16'))
        tf.print('Mixed precision enabled.')

    # fingerprinter: fingerprinter g(f(.)).
    fingerprinter = get_fingerprinter(cfg, trainable=False)
    return melspec_layer, specaug_layer, fingerprinter

@tf.function
def generate_uniform_peaks(batch_size, npoints):
    """Generate a random peaks that follow a uniform distribution in the melspectrogram
    domain.

    Args:
        batch_size (int)
        npoints (int) -- Number of points (peaks) that we take for each batch_id
    Returns:
        tf.tensor (B,npoints,3)
    """
    uniform_t = tf.random.uniform(
        shape=[batch_size,npoints,3],
        minval=[0,0,0],
        maxval=[255,31,1]) #TODO Change these values according to the config FFT parameters
    t1 = tf.math.round(uniform_t[:,:,0])
    t2 = tf.math.round(uniform_t[:,:,1])
    return tf.stack([t1,t2,uniform_t[:,:,2]], axis=2)

@tf.function
def extract_peak_tensors_from_mel(peaks_mel):
    """ Transforms a melspectrogram with only the peaks to a list of tensors in which
    each tensor contains the (BatchID, Freq, Timeframe, Amplitude) for each peak in the
    melspectrogram. The list has the same length as the batch size.
    Args:
        peaks_mel: (tf.tensor) -- Melspetrogram with only peaks. (B,F,T,1)
    Returns:
        peaks_per_batch: (list(tf.tensor)) -- List of Eager tensors. Length of the list
            is the batch size. Each tensor in the list has shape (n_peaks, 4) where
            n_peaks is the number of peaks in that batch_id, and 4 are (B,F,T,A) where
            A is the peak amplitude in the melspectrogram.
    """
    peaks_sparse = tf.sparse.from_dense(peaks_mel)
    peaks_indices_float = tf.cast(peaks_sparse.indices, dtype=tf.float32)
    peaks_coords = tf.stack([peaks_indices_float[:,0], peaks_indices_float[:,1], peaks_indices_float[:,2], peaks_sparse.values], axis=1)
    _, unique_batches = tf.unique(peaks_coords[:, 0])
    peaks_per_batch = tf.dynamic_partition(peaks_coords, unique_batches, num_partitions=peaks_mel.shape[0])
    return peaks_per_batch

@tf.function
def sort_peaks(peaks_per_batch, ordering="amplitude_desc"):
    """Sort peaks according to a specific ordering.

    Args:
        peaks_per_batch: (list(tf.tensor)) -- List of Eager tensors. Length of the list
                is the batch size. Each tensor in the list has shape (n_peaks, 4) where
                n_peaks is the number of peaks in that batch_id, and 4 are (B,F,T,A) where
                A is the peak amplitude in the melspectrogram.
        ordering: (str) -- Ordering type.
    Returns:
        list(tf.tensor) -- List of Eager tensors. List of peaks ordered by amplitude value
                in descending order. Length of the list is the batch size.
                Each tensor in the list has shape (n_peaks, 4) where n_peaks is the number
                of peaks in that batch_id, and 4 are (B,F,T,A) where A is the peak
                amplitude in the melspectrogram.
    """
    def _sort_matrix_by_last_column(matrix):
        amplitudes = matrix[:, -1]
        # Get the indices of amplitudes in descending order
        sorted_indices = tf.argsort(amplitudes, direction='DESCENDING')
        # Use these indices to sort the entire matrix
        sorted_matrix = tf.gather(matrix, sorted_indices)
        return sorted_matrix
    if ordering == "amplitude_desc":
        sorted_peaks = [_sort_matrix_by_last_column(matrix) for matrix in peaks_per_batch]
    elif ordering == "none":
        sorted_peaks = peaks_per_batch
    else:
        sys.exit("peak ordering mode not implemented. Please use one of the"
                 "available orderings: 'amplitude_desc'")
    return sorted_peaks


@tf.function
def pad_peaks(sorted_peaks, npoints):
    """0-Pads all the batch elements to have the same number of peaks.

    Args:
        sorted_peaks: (list(tf.tensor)) -- List of Eager tensors. List of peaks
                ordered by amplitude value in descending order. Length of the list
                is the batch size. Each tensor in the list has shape (n_peaks, 4)
                where n_peaks is the number of peaks in that batch_id, and 4 are
                (B,F,T,A) where A is the peak amplitude in the melspectrogram.
        npoints: (int) -- Number of points per sample.
    Returns:
        padded_peaks (tf.tensor) -- Tensor with shape (B, npoints, 3).
    """

    batch_size = len(sorted_peaks)

    # Create a padded tensor with the desired shape
    # padded_peaks = tf.zeros((batch_size, npoints, 4), dtype=sorted_peaks[0].dtype)  # all zeros
    padded_peaks = tf.stack(  # keep batch_size value
        [tf.concat([tf.fill([npoints, 1], float(i)),
                    tf.zeros((npoints, 3))], axis=1) for i in range(batch_size)])

    # Use tf.tensor_scatter_nd_update to insert the peaks into the padded tensor
    for i, peaks in enumerate(sorted_peaks):
        limit = tf.minimum(tf.shape(peaks)[0], npoints)
        indices = tf.reshape(tf.range(limit), [-1, 1])
        updates = peaks[:limit]
        indices = tf.concat([tf.fill([limit, 1], i), indices], axis=1)
        padded_peaks = tf.tensor_scatter_nd_update(padded_peaks, indices, updates)

    return padded_peaks


@tf.function
def sample_peaks(padded_peaks, uniform_peaks):
    """ Substitute padded valued peaks for random uniform peaks.

    Args:
        sliced_peaks (tf.tensor) -- Tensor with shape (B, npoints, 3). Contains padded
            peaks.
    Returns:
        sampled_peaks (tf.tensor) -- Tensor with shape (B, npoints, 3). Padded peaks
            are substituted with random uniform peaks.
    """
    peaks = tf.where(padded_peaks == PADDING_VALUE, uniform_peaks, padded_peaks)
    return peaks


def normalize_peaks(peaks, cfg):
    """Normalize peak space so F,T,A values are in the [0,1] range each.
    (Amplitude comes already normalized, though)

    Args:
        peaks (N,4): Peaks.
        cfg (dict): configuration loaded from the yml file.

    Returns:
        peaks_norm (N,4): Peaks normalized.
    """
    n_mels = cfg["MODEL"]["N_MELS"]
    stft_hop = cfg["MODEL"]["STFT_HOP"]
    n_samples = cfg["MODEL"]["DUR"] * cfg["MODEL"]["FS"]
    timefreq_ratio = cfg["PREPROCESSING"]["TIMEFREQ_RATIO"]
    assert timefreq_ratio >=0 and timefreq_ratio <= 1, "Invalid time-freq ratio."
    if n_samples // n_mels != 0:
        n_frames = int(n_samples // stft_hop ) + 1
    else:
        n_frames = int(n_samples / stft_hop)
    if timefreq_ratio == 0:
        freq_max = n_mels - 1
    else:
        freq_max = (n_mels - 1) * timefreq_ratio
    max_val = tf.constant([freq_max, n_frames-1, 1.0])
                           # Reshape the norm_factors tensor to match the last dimension of the matrix
    max_val = tf.cast(max_val, tf.float32)

    peaks_norm = []
    for p in peaks:
        batch_id = tf.cast(p[:,0], tf.float32)
        p_norm = p[:,1:]/tf.cast(max_val, tf.float32)
        peaks_norm.append(tf.concat([batch_id[:,None], p_norm], axis=1, name="concat"))
    return peaks_norm


@tf.function
def sample_peaks_matrix(peaks, npeaks):
    """
    Get a matrix with constant dimmensions of all the peaks by 0padding.

    Arguments
    ---------
    peaks: (ndarray)
        list of tensors in which each tensor contains the (BatchID, Freq,
        Timeframe, Amplitude) for each peak in the melspectrogram.
    npeaks: (int)
        Number of peaks per spectrogram.
    """
    padded_peaks = pad_peaks(peaks, npeaks)
    return tf.gather(padded_peaks, indices=[1, 2, 3], axis=-1)


@tf.function
def tiled_peaks_matrix(sorted_peaks, npoints):
    """
    Get a matrix with constant dimmensions of all the peaks in a batch sampling
    from duplicating peaks if needed.

    Args:
        sorted_peaks: (list(tf.tensor)) -- List of Eager tensors. List of peaks
                ordered by amplitude value in descending order. Length of the list
                is the batch size. Each tensor in the list has shape (n_peaks, 4)
                where n_peaks is the number of peaks in that batch_id, and 4 are
                (B,F,T,A) where A is the peak amplitude in the melspectrogram.
        npoints: (int) -- Number of points per sample.
    Returns:
        padded_peaks (tf.tensor) -- Tensor with shape (B, npoints, 3).
    """

    def _pad_or_trim(peaks):
        n_peaks = tf.shape(peaks)[0]

        def trim():
            return peaks[:npoints, :]

        def pad():
            repeat_ntimes = tf.math.ceil(npoints / n_peaks)
            repeats = tf.tile(peaks, [repeat_ntimes, 1])[:npoints]
            return repeats

        return tf.cond(n_peaks >= npoints, trim, pad)

    padded_peaks = tf.stack([_pad_or_trim(peaks) for peaks in sorted_peaks])
    return tf.gather(padded_peaks, indices=[1, 2, 3], axis=-1)

@tf.function
def extract_peaks(melspec, kernel_size=3):
    """ Extract peaks from melspectrogram.

    Args:
        melspec (tf.tensor) -- Tensor with shape (B, F, T, 1). Contains all the
            melspectrograms of the batch samples.
    Returns:
        peaks (list[peaks_sample]) -- List of all the peaks of all samples
            sorted in descending order of amplitude. peaks_sample has shape
            [N, 4] where N is the number of extracted peaks (variable to each
            sample) and 4 is (B,F,T,A)
        peaks_mel (tf.tensor) -- Peaks located in a spectrogram shaped matrix.
    """
    pooled_result = tf.nn.pool(melspec,
                               window_shape=[kernel_size, kernel_size],
                               strides=[1,1],
                               pooling_type='MAX',
                               padding='SAME')
    raw_peaks = tf.equal(melspec, pooled_result)
    peaks_mel = tf.where(raw_peaks, melspec, tf.zeros_like(melspec))
    peaks = extract_peak_tensors_from_mel(peaks_mel=peaks_mel)
    return peaks, peaks_mel

@tf.function
def get_peaks_from_mel(melspec, cfg):
    """ Complete pipeline for extracting the peaks that will be fed to the PointNetAFP
    network

    Args:
        melspec (tf.tensor) -- Tensor with shape (B, F, T, 1). Contains all the
            melspectrograms of the batch samples.
        cfg (dict) -- experiment config
    Returns:
        feat (tf.tensor) -- (B, npeaks, 3) tensor matrix where axis 2 equals to F,T,A.
                            note: A can be substituted by 1s if USE_AMPLITUDE=FALSE.
    """
    model = cfg["MODEL"]["ARCH"]
    peaks, peaks_mel = extract_peaks(melspec=melspec, kernel_size=3)
    peaks = sort_peaks(peaks,
                       ordering=cfg["PREPROCESSING"]["PEAK_ORDERING"].lower())
    if cfg["PREPROCESSING"]["NORM_COORDS"]:
        peaks = normalize_peaks(peaks, cfg)
    if model == "nnfp":
        feat = peaks_mel
    elif model == "pointnet2":
        npeaks = cfg["PREPROCESSING"]["NPEAKS"]
        # NOTE: To include here fps, we would need a Tensor of constant dimensions
        if cfg["PREPROCESSING"]["PEAK_SAMPLING"].lower() == "0padding":
            feat = sample_peaks_matrix(peaks, npeaks)
        elif cfg["PREPROCESSING"]["PEAK_SAMPLING"].lower() == "tile":
            feat = tiled_peaks_matrix(peaks, npeaks)
        if not cfg["PREPROCESSING"]["USE_AMPLITUDE"]:
            ones_matrix = tf.ones_like(feat[:, :, 2])
            feat = tf.stack([feat[:,:,0],
                             feat[:,:,1],
                             ones_matrix], axis=2)
    return feat

@tf.function
def stretch_batch(mel_x, n_anchors, cfg):
    max_factor = 1 + round(cfg["TD_AUG"]["MAX_STRETCH_AUG"]/100,3)
    melspec_nframes = int(np.ceil(cfg["MODEL"]["FS"]/cfg["MODEL"]["STFT_HOP"]))
    max_length_melspec = int(melspec_nframes * max_factor)  # 64
    mel_x_refs = pad_mel_to_fixed_size(mel_x[:n_anchors], nframes=max_length_melspec)
    mel_x_queries, _ = stretch_melspec(mel_x[n_anchors:], cfg)
    mel_x_queries = pad_mel_to_fixed_size(mel_x_queries, nframes=max_length_melspec)
    mel_x = tf.concat([mel_x_refs, mel_x_queries], axis=0)
    return mel_x

@tf.function
def get_melspec_excerpt(melspec, nframes):
    "Extract a random segment of melspecs in batch"
    diff = int(tf.shape(melspec)[2] - nframes)
    tf.debugging.assert_greater(
        diff,
        0,
        message="The number of frames should be smaller than the length of melspec."
    )
    start_frame = tf.random.uniform(
        shape=[], minval=0, maxval=diff, dtype=tf.int32
    )
    melspec = melspec[:,:,start_frame:(start_frame+nframes),:]
    return melspec

@tf.function
def stretch_batch_1s(mel_x, n_anchors, cfg):
    melspec_nframes = int(np.ceil(cfg["MODEL"]["FS"]/cfg["MODEL"]["STFT_HOP"]))
    mel_x_refs = mel_x[:n_anchors]
    mel_x_queries, sf = stretch_melspec(mel_x[n_anchors:], cfg)
    if sf < 1:
        mel_x_queries = get_melspec_excerpt(mel_x_queries, nframes=melspec_nframes)
    elif sf > 1:
        mel_x_queries = pad_mel_to_fixed_size(mel_x_queries, nframes=melspec_nframes)
    mel_x = tf.concat([mel_x_refs, mel_x_queries], axis=0)
    return mel_x

@tf.function
def stretch_melspec(mel_x, cfg):
    """Apply the stretching to the melspectrogram as image resizing."""
    max_stretch = cfg["TD_AUG"]["MAX_STRETCH_AUG"]
    melspec_nframes = int(np.ceil(cfg["MODEL"]["FS"]/cfg["MODEL"]["STFT_HOP"]))
    factor = 1 + round(random.uniform(-max_stretch/2, max_stretch)/100,3)
    mel_x = tf.image.resize(
        mel_x,
        [cfg["MODEL"]["N_MELS"], int(np.ceil(melspec_nframes/factor))])
    return mel_x, factor

@tf.function
def pad_mel_to_fixed_size(melspec, nframes):
    melspec_shape = tf.shape(melspec)
    pad_amount = nframes - melspec_shape[2]
    # Define the padding dimensions: (before, after) for each axis
    paddings = tf.stack([[0, 0], [0, 0], [0, pad_amount], [0, 0]])
    padded_tensor = tf.pad(melspec, paddings, mode='CONSTANT', constant_values=0)
    return padded_tensor

def stretch_pysox(X, cfg, melspec_layer, specaug_layer):
    """Stretch the audio with pysox. It is really slow and the training is not
    feasible as it is c oded now (cuda compatibility). Leaving the code here just
    in case we want to revisit it"""
    max_factor = 2
    max_length = int(cfg["MODEL"]["FS"] * cfg["MODEL"]["DUR"] * max_factor)  # 16000
    xa = pad_tensor_to_fixed_size(X[0], max_length)
    xp = pad_tensor_to_fixed_size(X[1], max_length)
    # -------- anchors --------
    mel_x_anchors = melspec_layer(xa)  # (BSZ, F, T, 1)
    # if pointnet: feat = (BSZ, npeaks, 3)
    if cfg['MODEL']['DATA_TYPE'] == "peaks":
        feat_anchors = get_peaks_from_mel(melspec=mel_x_anchors, cfg=cfg)
    else:
        if cfg['NEURALFP']['USE_SPECAUG']:
            feat_anchors = specaug_layer(mel_x_anchors)  # (nA+nP, F, T, 1)
    # -------- pairs --------
    mel_x_pairs = melspec_layer(xp)  # (BSZ, F, T, 1)
    # if pointnet: feat = (BSZ, npeaks, 3)
    if cfg['MODEL']['DATA_TYPE'] == "peaks":
        feat_pairs = get_peaks_from_mel(melspec=mel_x_pairs, cfg=cfg)
    else:
        if cfg['NEURALFP']['USE_SPECAUG']:
            feat_pairs = specaug_layer(mel_x_pairs)  # (nA+nP, F, T, 1)
    feat = tf.concat([feat_anchors, feat_pairs], axis=0)
    return feat

@tf.function
def pad_tensor_to_fixed_size(melspec, max_length):
    current_shape = tf.shape(melspec)
    pad_amount = max_length - current_shape[2]
    # Define the padding dimensions: (before, after) for each axis
    paddings = tf.stack([[0, 0], [0, 0], [0, pad_amount]])
    padded_tensor = tf.pad(melspec, paddings, mode='CONSTANT', constant_values=0)
    return padded_tensor

@tf.function
def train_step(X, melspec_layer, specaug_layer, fingerprinter, loss_obj, helper, cfg):
    """ Train step """
    # X: (Xa, Xp)
    # Xa: anchors or originals, s.t. [xa_0, xa_1,...]
    # Xp: augmented replicas, s.t. [xp_0, xp_1] with xp_n = rand_aug(xa_n).
    n_anchors = len(X[0])
    X = tf.concat(X, axis=0)
    mel_x = melspec_layer(X)  # (BSZ, F, T, 1)
    # if pointnet: feat = (BSZ, npeaks, 3)
    if cfg['TD_AUG']['USE_SPECAUG'] == True:
        mel_x = specaug_layer(mel_x)  # (nA+nP, F, T, 1)
    if cfg["TD_AUG"]["MELSPEC_STRETCH"] == True:  # melspec stretching
        mel_x = stretch_batch_1s(mel_x, n_anchors, cfg)
    if cfg['MODEL']['DATA_TYPE'] == "peaks":
        feat = get_peaks_from_mel(melspec=mel_x, cfg=cfg)
    else:
        feat = mel_x
    fingerprinter.trainable = True
    with tf.GradientTape() as t:
        emb = fingerprinter(feat) # (BSZ, Dim)
        loss, sim_mtx, _ = loss_obj.compute_loss(emb[:n_anchors, :],
                                                 emb[n_anchors:, :]) # {emb_org, emb_rep}
        if fingerprinter.mixed_precision:
            scaled_loss = helper.optimizer.get_scaled_loss(loss)
    if fingerprinter.mixed_precision:
        scaled_g = t.gradient(scaled_loss, fingerprinter.trainable_variables)
        g = helper.optimizer.get_unscaled_gradients(scaled_g)
    else:
        g = t.gradient(loss, fingerprinter.trainable_variables)
    helper.optimizer.apply_gradients(zip(g, fingerprinter.trainable_variables))
    avg_loss = helper.update_tr_loss(loss) # To tensorboard.
    return avg_loss, sim_mtx # avg_loss: average within the current epoch


@tf.function
def val_step(X, melspec_layer, fingerprinter, loss_obj, helper, cfg):
    """ Validation step """
    n_anchors = len(X[0])
    X = tf.concat(X, axis=0)
    mel_x = melspec_layer(X)  # (BSZ, F, T, 1)
    # if pointnet: feat = (BSZ, npeaks, 3)
    if cfg["TD_AUG"]["MELSPEC_STRETCH"] == True:  # melspec stretching
        mel_x = stretch_batch_1s(mel_x, n_anchors, cfg)
    if cfg['MODEL']['DATA_TYPE'] == "peaks":
        feat = get_peaks_from_mel(melspec=mel_x, cfg=cfg)
    else:
        feat = mel_x
    fingerprinter.trainable = False
    emb = fingerprinter(feat)  # (BSZ, Dim)
    loss, sim_mtx, _ = loss_obj.compute_loss(emb[:n_anchors, :],
                                             emb[n_anchors:, :]) # {emb_org, emb_rep}
    avg_loss = helper.update_val_loss(loss) # To tensorboard.
    return avg_loss, sim_mtx

@tf.function
def test_step(X, melspec_layer, fingerprinter, cfg):
    """ Test step used for mini-search-validation """
    fingerprinter.trainable = False
    n_anchors = len(X[0])
    X = tf.concat(X, axis=0)  # (VAL_BATCH_SZ, 1, 8000)
    mel_x = melspec_layer(X)  # (nA+nP, F, T, 1)
    if cfg["TD_AUG"]["MELSPEC_STRETCH"] == True:  # melspec stretching
        mel_x = stretch_batch_1s(mel_x, n_anchors, cfg)
    if fingerprinter.name == 'point_net_afp':
        # if pointnet: feat = (BSZ, npeaks, 3)
        feat = get_peaks_from_mel(melspec=mel_x, cfg=cfg)
        emb_gf = fingerprinter(feat)
        return None, None, emb_gf
    else: #(fingerprinter.name == "finger_printer")
        feat = mel_x
        if fingerprinter.mixed_precision:
            act = tf.keras.layers.Activation('linear', dtype='float32')
            emb_f = fingerprinter.front_conv(feat)  # (BSZ, Dim)
            emb_f_FP32 = act(emb_f)
            emb_f_postL2 = tf.math.l2_normalize(emb_f_FP32, axis=1)
            emb_gf = fingerprinter.div_enc(emb_f)
            emb_gf = act(emb_gf)
            emb_gf = tf.math.l2_normalize(emb_gf, axis=1)
            return emb_f_FP32, emb_f_postL2, emb_gf # f(.), L2(f(.)), L2(g(f(.))
        else:
            emb_f = fingerprinter.front_conv(feat)  # (BSZ, Dim)
            emb_f_postL2 = tf.math.l2_normalize(emb_f, axis=1)
            emb_gf = fingerprinter.div_enc(emb_f)
            emb_gf = tf.math.l2_normalize(emb_gf, axis=1)
            return emb_f, emb_f_postL2, emb_gf # f(.), L2(f(.)), L2(g(f(.))


def mini_search_validation(ds, melspec_layer, fingerprinter, cfg, mode='argmin',
                           scopes=[1, 3, 5, 9, 11, 19], max_n_samples=3000):
    """ Mini-search-validation """
    fingerprinter.trainable = False
    (db, query, emb, dim) = (dict(), dict(), dict(), dict())
    bsz = ds.bsz
    n_anchor = bsz // 2
    n_iter = min(len(ds), max_n_samples // bsz)
    if fingerprinter.name == "point_net_afp":
        key_strs = ['g(f)']
        dim['g(f)'] = fingerprinter.emb_sz
    else:
        # Construct mini-DB
        key_strs = ['f', 'L2(f)', 'g(f)']
        dim['f'] = dim['L2(f)'] = fingerprinter.front_hidden_ch[-1]
        dim['g(f)'] = fingerprinter.emb_sz
    for k in key_strs:
        (db[k], query[k]) = (tf.zeros((0, dim[k])), tf.zeros((0, dim[k])))
    for i in range(n_iter):
        X = ds.__getitem__(i)
        emb['f'], emb['L2(f)'], emb['g(f)'] = test_step(X=X,
                                                        melspec_layer=melspec_layer,
                                                        fingerprinter=fingerprinter,
                                                        cfg=cfg)
        for k in key_strs:
            db[k] = tf.concat((db[k], emb[k][:n_anchor, :]), axis=0)
            query[k] = tf.concat((query[k], emb[k][n_anchor:, :]), axis=0)

    # db["g(f)"].shape == [1440,128]
    # Search test
    accs_by_scope = dict()
    for k in key_strs:
        tf.print(f'======= mini-search-validation: \033[31m{mode} \033[33m{k} \033[0m=======' + '\033[0m')
        query[k] = tf.expand_dims(query[k], axis=1) # (nQ, d) --> (nQ, 1, d)
        accs_by_scope[k], _ = mini_search_eval(
            query[k], db[k], scopes, mode, display=True)
    return accs_by_scope, scopes, key_strs


def trainer(cfg, checkpoint_name):
    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tf.print(f"{time_now}  |  Starting the training.")

    # Initialize the datasets
    tf.print('-----------Initializing the datasets-----------')
    # Dataloader
    dataset = Dataset(cfg)
    train_ds = dataset.get_train_ds(cfg['DATA_SEL']['REDUCE_ITEMS_P'])

    # Build models.
    tf.print('-----------Building the model------------------')
    melspec_layer, specaug_layer, fingerprinter = build_fp(cfg)

    # Learning schedule
    tf.print('-----------Setting lr schedule-----------------')
    total_nsteps = cfg['TRAIN']['MAX_EPOCH'] * len(train_ds)
    if cfg['TRAIN']['LR_SCHEDULE'].upper() == 'COS':
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=float(cfg['TRAIN']['LR']),
            decay_steps=total_nsteps,
            alpha=1e-06)
    elif cfg['TRAIN']['LR_SCHEDULE'].upper() == 'COS-RESTART':
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=float(cfg['TRAIN']['LR']),
            first_decay_steps=int(total_nsteps * 0.1),
            num_periods=0.5,
            alpha=2e-06)
    else:
        lr_schedule = float(cfg['TRAIN']['LR'])

    # Optimizer
    if cfg['TRAIN']['OPTIMIZER'].upper() == 'LAMB':
        opt = LAMB(learning_rate=lr_schedule)
    elif cfg['TRAIN']['OPTIMIZER'].upper() == 'ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        raise NotImplementedError(cfg['TRAIN']['OPTIMIZER'])
    if cfg['TRAIN']['MIXED_PRECISION']:
        opt = LossScaleOptimizer(opt)

    # Experiment helper: see utils.experiment_helper.py for details.
    helper = ExperimentHelper(
        checkpoint_name=checkpoint_name,
        optimizer=opt,
        model_to_checkpoint=fingerprinter,
        cfg=cfg)

    # Loss objects
    if cfg['LOSS']['LOSS_MODE'].upper() == 'NTXENT': # Default
        loss_obj_train = NTxentLoss(
            n_org=cfg['BSZ']['TR_N_ANCHOR'],
            n_rep=cfg['BSZ']['TR_BATCH_SZ'] - cfg['BSZ']['TR_N_ANCHOR'],
            tau=cfg['LOSS']['TAU'])
        loss_obj_val = NTxentLoss(
            n_org=cfg['BSZ']['VAL_N_ANCHOR'],
            n_rep=cfg['BSZ']['VAL_BATCH_SZ'] - cfg['BSZ']['VAL_N_ANCHOR'],
            tau=cfg['LOSS']['TAU'])
    elif cfg['LOSS']['LOSS_MODE'].upper() == 'ONLINE-TRIPLET': # Now-playing
        loss_obj_train = OnlineTripletLoss(
            bsz=cfg['BSZ']['TR_BATCH_SZ'],
            n_anchor=cfg['BSZ']['TR_N_ANCHOR'],
            mode = 'semi-hard',
            margin=cfg['LOSS']['MARGIN'])
        loss_obj_val = OnlineTripletLoss(
            bsz=cfg['BSZ']['VAL_BATCH_SZ'],
            n_anchor=cfg['BSZ']['VAL_N_ANCHOR'],
            mode = 'all', # use 'all' mode for validation
            margin=0.)
    else:
        raise NotImplementedError(cfg['LOSS']['LOSS_MODE'])

    # Initialize variables
    sim_mtx = None

    # Training loop
    ep_start = helper.epoch
    ep_max = cfg['TRAIN']['MAX_EPOCH']
    if ep_start != 1:
        tf.print("... continuing training ...")
        assert ep_start <= ep_max, f"MAX_EPOCH={ep_max} must be => to {ep_start} "\
        f"(where training was left at)."

    tf.print('-----------Training starts---------------------')
    for ep in range(ep_start, ep_max + 1):
        tf.print(f'EPOCH: {ep}/{ep_max}')
        tf.print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        train_ds = dataset.get_train_ds(cfg['DATA_SEL']['REDUCE_ITEMS_P'])
        progbar = Progbar(len(train_ds))
        # Train
        # if shuffle==True --> shuffles the data at the beginning of each epoch.
        enq = tf.keras.utils.OrderedEnqueuer(train_ds,
                                             use_multiprocessing=True,
                                             shuffle=train_ds.shuffle)
        # """ Parallelism to speed up preprocessing.............. """
        # CPU workers prepare the batches to be taken by the GPU.
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                  max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
        i = 0
        while i < len(enq.sequence):
            X = next(enq.get()) # X: Tuple(Xa, Xp)
            tr_avg_loss, sim_mtx = train_step(X=X,
                                           melspec_layer=melspec_layer,
                                           specaug_layer=specaug_layer,
                                           fingerprinter=fingerprinter,
                                           loss_obj=loss_obj_train,
                                           helper=helper,
                                           cfg=cfg)
            progbar.add(1, values=[("tr loss", tr_avg_loss)])
            i += 1
        enq.stop()
        # """ End of Parallelism................................. """

        # print network summary at the first epoch.
        if ep==1:
            if cfg["MODEL"]["ARCH"]=="nnfp":
                tf.print("NNFP convolutional layers summary:")
                tf.print(fingerprinter.front_conv.summary())
            elif fingerprinter.name == "point_net_afp":
                tf.print("PointNet model summary:")
                tf.print(fingerprinter.summary())

        if cfg['TRAIN']['SAVE_IMG'] and (sim_mtx is not None):
            helper.write_image_tensorboard('tr_sim_mtx', sim_mtx.numpy())

        # Validate
        val_ds = dataset.get_val_ds(max_song=cfg['TRAIN']['VAL_SIZE']) # original 250, max 500.
        progbar = Progbar(len(val_ds))
        # """ Parallelism to speed up preprocessing.............. """
        enq = tf.keras.utils.OrderedEnqueuer(val_ds,
                                             use_multiprocessing=True,
                                             shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                  max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
        i = 0
        while i < len(enq.sequence):
            X = next(enq.get()) # X: Tuple(Xa, Xp)
            val_avg_loss, sim_mtx = val_step(X=X,
                                  melspec_layer=melspec_layer,
                                  fingerprinter=fingerprinter,
                                  loss_obj=loss_obj_val,
                                  helper=helper,
                                  cfg=cfg)
            progbar.add(1, values=[("val loss", val_avg_loss)])
            i += 1
        enq.stop()
        # """ End of Parallelism................................. """

        if cfg['TRAIN']['SAVE_IMG'] and (sim_mtx is not None):
            helper.write_image_tensorboard('val_sim_mtx', sim_mtx.numpy())

        # On epoch end
        tf.print('tr_loss:{:.4f}, val_loss:{:.4f}'.format(helper._tr_loss.result(),
                                                          helper._val_loss.result()))
        helper.update_on_epoch_end(save_checkpoint_now=True)

        # Mini-search-validation (optional)
        if cfg['TRAIN']['MINI_TEST_IN_TRAIN']:
            accs_by_scope, scopes, key_strs = mini_search_validation(ds=val_ds,
                                                                     melspec_layer=melspec_layer,
                                                                     fingerprinter=fingerprinter,
                                                                     cfg=cfg)
            for k in key_strs:
                helper.update_minitest_acc(accs_by_scope[k], scopes, k)
