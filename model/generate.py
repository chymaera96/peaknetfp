# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" generate.py """
import csv
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from model.dataset import Dataset
from model.trainer import get_peaks_from_mel, pad_mel_to_fixed_size
from model.fp.melspec.melspectrogram import get_melspec_layer
from model.fp.nnfp import get_fingerprinter


def build_fp(cfg):
    """ Build fingerprinter """
    # melspec_layer: log-power-Mel-spectrogram layer, S.
    melspec_layer = get_melspec_layer(cfg, trainable=False)

    # fingerprinter: fingerprinter g(f(.)).
    fingerprinter = get_fingerprinter(cfg, trainable=False)
    return melspec_layer, fingerprinter


def load_checkpoint(checkpoint_root_dir, checkpoint_name, checkpoint_index,
                    fingerprinter):
    """ Load a trained fingerprinter """
    # Create checkpoint
    checkpoint = tf.train.Checkpoint(model=fingerprinter)
    checkpoint_dir = os.path.join(checkpoint_root_dir, checkpoint_name)
    c_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir,
                                           max_to_keep=None)

    # Load
    if checkpoint_index == None:
        tf.print("\x1b[1;32mArgument 'checkpoint_index' was not specified.\x1b[0m")
        tf.print('\x1b[1;32mSearching for the latest checkpoint...\x1b[0m')
        latest_checkpoint = c_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint_index = int(latest_checkpoint.split(sep='ckpt-')[-1])
            status = checkpoint.restore(latest_checkpoint)
            status.expect_partial()
            tf.print(f'---Restored from {c_manager.latest_checkpoint}---')
        else:
            raise FileNotFoundError(f'Cannot find checkpoint in {checkpoint_dir}')
    else:
        checkpoint_fpath = os.path.join(checkpoint_dir, 'ckpt-' + str(checkpoint_index))
        status = checkpoint.restore(checkpoint_fpath) # Let TF to handle error cases.
        status.expect_partial()
        tf.print(f'---Restored from {checkpoint_fpath}---')
    return checkpoint_index


def prevent_overwrite(key, target_path):
    if (key == 'dummy_db') & os.path.exists(target_path):
        answer = input(f'{target_path} exists. Overwrite (y/N)? ')
        if answer.lower() not in ['y', 'yes']: sys.exit()


def get_data_source(cfg, source: str, skip_dummy: bool, source_type: str, query_dir:str=None):
    dataset = Dataset(cfg)
    ds = dict()
    if source:
        tf.print("Getting data from \033[33m'custom source'\033[0m.")
        ds['custom_source'] = dataset.get_custom_db_ds(source, source_type)
    else:
        if skip_dummy:
            tf.print("Excluding \033[33m'dummy_db'\033[0m from source.")
        else:
            tf.print("Getting \033[33m'dummy_db'\033[0m data.")
            ds['dummy_db'] = dataset.get_test_dummy_db_ds()

        if dataset.datasel_test_query_db in ['unseen_icassp', 'unseen_syn']:
            tf.print("Getting \033[33m'unseen_icassp'\033[0m test_query_db data.")
            ds['query'], ds['db'] = dataset.get_test_query_db_ds()
        elif dataset.datasel_test_query_db == 'unseen_icassp_stretch':
            tf.print("Getting \033[33m'unseen_icassp_stretch'\033[0m test_query_db data.")
            tf.print(f"Looking into \033[33m'{query_dir}'\033[0m.")
            ds['query'], ds['db'] = dataset.get_test_query_db_ds_stretch(query_dir)
        else:
            raise ValueError(dataset.datasel_test_query_db)

    tf.print(f'\x1b[1;32mData source: {ds.keys()}\x1b[0m',
             f'{dataset.datasel_test_query_db}')
    return ds


@tf.function
def test_step(X, melspec_layer, fingerprinter, cfg):
    """ Test step used for generating fingerprint """
    # X is Xa only here, not (Xa, Xp) as in training and validation. 
    fingerprinter.trainable = False
    n_anchors = len(X[0])
    X = tf.concat(X, axis=0)  # (VAL_BATCH_SZ, 1, 8000)
    mel_x = melspec_layer(X)  # (nA+nP, F, T, 1)
    if fingerprinter.name == 'point_net_afp':
        feat = get_peaks_from_mel(melspec=mel_x, cfg=cfg)
    else:
        feat = mel_x
    emb = fingerprinter(feat)  # (BSZ, emb_sz)
    return emb


def generate_fingerprint(cfg,
                         checkpoint_name,
                         checkpoint_index,
                         source,
                         output_root_dir,
                         skip_dummy,
                         source_type,
                         query_dir=None):
    """ Generate fingerprints from a trained model checkpoint.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    checkpoint_dir : str, optional
        Directory containing the checkpoints. (default: "")
        If not specified, load from the directory specified in the config file.
    checkpoint_index : int, optional
        Index of the checkpoint to load from. (default: 0)
        0 means the latest checkpoint. 97 means the 97th epoch.
    source : str, optional
        Path to the custom source. (default: "")
        If not specified, load from the default source specified in the config file.
    output_root_dir : str, optional
        Root directory for the output. (default: "")
        If not specified, load from the default directory specified in the config file.
    skip_dummy : bool, optional
        Whether generating the skip dummy_db. (default: False)
    source_type : str, optional
        In case of a custom dataset, specify the type of source
    query_dir : str, optional
        In case of stretch, we have different query folders for the same dataset.
        One for each stretching factor.
    
    After run, the output (generated fingerprints) directory will be:
      .
      └──output_root_dir (default=./logs/emb)
            └── checkpoint_name
                └── checkpoint_index
                    ├── db.mm
                    ├── db_shape.npy
                    ├── dummy_db.mm
                    ├── dummy_db_shape.npy
                    ├── query.mm
                    └── query_shape.npy
    """
    # Build and load checkpoint
    melspec_layer, fingerprinter = build_fp(cfg)
    checkpoint_root_dir = os.path.join(cfg['DIR']['LOG_ROOT_DIR'], 'checkpoint')
    checkpoint_index = load_checkpoint(checkpoint_root_dir, checkpoint_name,
                                       checkpoint_index, fingerprinter)

    # Get data source
    # ds = {'key1': <Dataset>, 'key2': <Dataset>, ...}
    if query_dir != None:
        ds = get_data_source(cfg, source, skip_dummy, source_type, query_dir=query_dir)
    else:
        ds = get_data_source(cfg, source, skip_dummy, source_type)

    # Make output directory
    if not output_root_dir:
        output_root_dir = os.path.join(
            cfg['DIR']['OUTPUT_ROOT_DIR'],
            checkpoint_name,
            str(checkpoint_index)
        )

    os.makedirs(output_root_dir, exist_ok=True)
    if not skip_dummy:
        prevent_overwrite('dummy_db', os.path.join(output_root_dir, 'dummy_db.mm'))

    # Generate
    sz_check = dict() # for warning message
    for key in ds.keys():
        bsz = int(cfg['BSZ']['TS_BATCH_SZ'])  # Do not use ds.bsz here.
        # n_items = len(ds[key]) * bsz
        n_items = ds[key].n_samples
        emb_sz = cfg['MODEL']['EMB_SZ']
        """
        Why use "memmap"?

        • First, we need to store a huge uncompressed embedding vectors until
          constructing a compressed DB with IVF-PQ (using FAISS). Handling a
          huge ndarray is not a memory-safe way: "memmap" consume 0 memory.

        • Second, Faiss-GPU does not support reconstruction of DB from
          compressed DB (index). In eval/eval_faiss.py, we need uncompressed
          vectors to calaulate sequence-level matching score. The created
          "memmap" will be reused at that point.

        Reference:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

        """
        # Create memmap, and save shapes
        assert n_items > 0
        arr_shape = (n_items, emb_sz)
        mm_name = key
        if source:
            source_bname = Path(source).stem
            tf.print(f"Generating fingerprint from {source_type}.")
            if source_type == "file":
                mm_name = source_bname

        arr = np.memmap(os.path.join(output_root_dir, f'{mm_name}.mm'),
                        dtype='float32',
                        mode='w+',
                        shape=arr_shape)
        np.save(os.path.join(output_root_dir, f'{mm_name}_shape.npy'), arr_shape)

        # Fingerprinting loop
        tf.print(
            f"=== Generating fingerprint from \x1b[1;32m'{key}'\x1b[0m " +
            f"bsz={bsz}, {n_items} items, d={emb_sz}"+ " ===")
        progbar = Progbar(len(ds[key]))

        # Parallelism to speed up preprocessing-------------------------
        enq = tf.keras.utils.OrderedEnqueuer(ds[key],
                                              use_multiprocessing=True,
                                              shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                  max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
        i = 0

        if source and (source_type == "file"):
            segments_csv = os.path.join(output_root_dir, f"{source_bname}_{key}_segments.csv")
        else:
            segments_csv = os.path.join(output_root_dir, f"{key}_segments.csv")

        with open(segments_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['segment_id', 'filename', 'intra_segment_id', 'offset_min', 'offset_max'] # from model/utils/dataloader_keras.py line 117
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for ii, seg in enumerate(ds[key].fns_event_seg_list):
                writer.writerow({'segment_id': ii,
                                 'filename': seg[0],
                                 'intra_segment_id': seg[1],
                                 'offset_min': seg[2],
                                 'offset_max': seg[3]})
        
        while i < len(enq.sequence):
            progbar.update(i)
            X, _ = next(enq.get())
            emb = test_step(X, melspec_layer, fingerprinter, cfg)
            arr[i * bsz:(i + 1) * bsz, :] = emb.numpy() # Writing on disk.
            i += 1
        progbar.update(i, finalize=True)
        enq.stop()
        # End of Parallelism--------------------------------------------
        sz_check[key] = len(arr)
        # if source_type == "file":
        #     np.save(os.path.join(output_root_dir, f'{source_bname}.npy'),
        #                 arr)
        arr.flush(); del(arr) # Close memmap
        tf.print(f'    Succesfully stored {arr_shape[0]} segment embeddings to {output_root_dir} ===')

    if 'custom_source' in ds.keys():
        pass;
    elif sz_check['db'] != sz_check['query']:
        print("\033[93mWarning: 'db' and 'query' size does not match. This can cause a problem in evaluataion stage.\033[0m")
    return
