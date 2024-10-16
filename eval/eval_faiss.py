# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" eval_faiss.py """
import os
import sys
import time
import glob
import click
import curses
import numpy as np
import faiss
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from eval.utils.get_index_faiss import get_index
from eval.utils.print_table import PrintTable


def load_memmap_data(source_dir,
                     fname,
                     append_extra_length=None,
                     shape_only=False,
                     display=True):
    """
    Load data and datashape from the file path.

    • Get shape from [source_dir/fname_shape.npy}.
    • Load memmap data from [source_dir/fname.mm].

    Parameters
    ----------
    source_dir : (str)
    fname : (str)
        File name except extension.
    append_empty_length : None or (int)
        Length to appened empty vector when loading memmap. If activate, the
        file will be opened as 'r+' mode.
    shape_only : (bool), optional
        Return only shape. The default is False.
    display : (bool), optional
        The default is True.

    Returns
    -------
    (data, data_shape)

    """
    path_shape = os.path.join(source_dir, fname + '_shape.npy')
    path_data = os.path.join(source_dir, fname + '.mm')
    data_shape = np.load(path_shape)
    if shape_only:
        return data_shape

    if append_extra_length:
        data_shape[0] += append_extra_length
        data = np.memmap(path_data, dtype='float32', mode='r+',
                         shape=(data_shape[0], data_shape[1]))
    else:
        data = np.memmap(path_data, dtype='float32', mode='r',
                         shape=(data_shape[0], data_shape[1]))
    if display:
        print(f'Load {data_shape[0]:,} items from \033[32m{path_data}\033[0m.')
    return data, data_shape


def query_id_from_db_id(db_seg_id, query_info, db_info, stretch_factor):
    """Retrieve query segment id from db segment id"""
    track_id = db_info[db_info["segment_id"]==db_seg_id].iloc[0]["track_id"]
    db_intra_segment_id = db_info[db_info["segment_id"]==db_seg_id].iloc[0]["intra_segment_id"]
    # hack to not get an error for this reference
    last_db_intra = db_info[db_info["track_id"]=="067474.wav"]["intra_segment_id"].max()
    if (db_intra_segment_id == last_db_intra) and (stretch_factor > 1):
        # Last segment of one reference, pick last query segment.
        query_seg_id = query_info[query_info["track_id"]==track_id]["segment_id"].max()
    else:
        query_intra_segment_id = int(np.floor(db_intra_segment_id/stretch_factor))
        query_seg_id = query_info[
            (query_info["track_id"]==track_id) &
            (query_info["intra_segment_id"] == query_intra_segment_id)
            ].iloc[0]["segment_id"]
    return query_seg_id


def get_ref_from_db_id(db_seg_id, db_segments_info, dummy_db_shape):
    """Retrieve track id for the song-level search"""
    seg_id = db_seg_id - dummy_db_shape[0]
    if seg_id >= 0:
        if seg_id not in db_segments_info['segment_id'].values:
            raise ValueError(f"Segment id {seg_id} not found in db_segments_info.")
        db_info = db_segments_info[db_segments_info["segment_id"] == seg_id]
        if db_info.shape[0] > 1:
            raise ValueError("ERROR: Segment id pointing to multiple db entries.")
        track_id = db_info.iloc[0].track_id
    else:
        track_id = "dummy"
    return track_id

def estimate_stretching_factor(query_segments, reference_segments):
    """Estimate stretching factor from IDs of query_segments and reference_segments
    """
    if query_segments.size != reference_segments.size:
        raise ValueError("missmatch between query_segments and reference_segments!")
    if query_segments.size == 1 and reference_segments.size == 1:
        return 1, 1
    else:
        # Calculate consecutive differences
        query_diffs = query_segments[:, np.newaxis] - query_segments
        ref_diffs = reference_segments[:, np.newaxis] - reference_segments
        # stretching_factors = np.where(ref_diffs == 0, 0, query_diffs / ref_diffs)
        # if stretching_factors.any():
        #     return np.mean(stretching_factors[stretching_factors != 0]), np.median(stretching_factors[stretching_factors != 0])

        # Avoid division by zero by setting stretching_factors to zero where ref_diffs is zero
        with np.errstate(divide='ignore', invalid='ignore'):
            stretching_factors = np.where(ref_diffs != 0, query_diffs / ref_diffs, 0)
        non_zero_factors = stretching_factors[stretching_factors != 0]
        if non_zero_factors.size > 0:
            return np.mean(non_zero_factors), np.median(non_zero_factors)    
        return 1, 1

"""
q   1   2   3   4       -->      -1   -2  -3  /   1   -1   -2  /   2   1   -1  /   3    2   1
r   3   1   6   8       -->       2   -3  -5  /  -2   -5   -7  /   3   5   -2  /   5    7   2
                        factor  -.5  .66  .6  / -.5   .2  .28  / .66  .2   .5  /  .6  .28  .5
                        mean: 0.29
                        median: 0.39
q   1   2   3   4       -->       -1   -2  -3  /    1    -1     -2  /    2      1  -1  /   3     2  1
r   3   40   6   8      -->      -37   -3  -5  /   37    34     32  /    3    -34  -2  /   5   -32  2
                        factor  .027  .66  .6  / .027 -.029   -.06  /  .66  -.029  .5  /  .6  -.06 .5
                        mean: 0.28
                        median: 0.26
q   1   2   3   4       -->     -1  -2  -3  /   1  -1  -2  /  2   1  -1  /   3   2   1
r   10  10  11  11      -->      0  -1  -1  /   0  -1  -1  /  1   1   0  /   1   1   0
                        factor   -   2   3  /   -   1   2  /  2   1   -  /   3   2   -
                        mean: 2
                        median: 2
"""

@click.command()
@click.argument('emb_dir', required=True,type=click.STRING)
@click.option('--emb_dummy_dir', default=None, type=click.STRING,
              help="Specify a directory containing 'dummy_db.mm' and "
              "'dummy_db_shape.npy' to use. Default is EMB_DIR.")
@click.option('--index_type', '-i', default='ivfpq', type=click.STRING,
              help="Index type must be one of {'L2', 'IVF', 'IVFPQ', "
              "'IVFPQ-RR', 'IVFPQ-ONDISK', HNSW'}")
@click.option('--nogpu', default=False, is_flag=True,
              help='Use this flag to use CPU only.')
@click.option('--stretch_factor', '-sf', default=1, type=click.FLOAT,
              help='Specify stretching factor of the queries.')
@click.option('--max_train', default=1e7, type=click.INT,
              help='Max number of items for index training. Default is 1e7.')
@click.option('--test_seq_len', default='1 3 5 9 11 19', type=click.STRING,
              help="A set of different number of segments to test. "
              "Numbers are separated by spaces. Default is '1 3 5 9 11 19', "
              "which corresponds to '1s, 2s, 3s, 5s, 6s, 10s'. (with 1 sec segment "
              "duration and 0.5 sec hop duration.)")
@click.option('--test_ids', '-t', default='icassp', type=click.STRING,
              help="One of {'all', 'icassp', 'path/file.npy', (int)}. If 'all', " +
              "test all IDs from the test. If 'icassp', use the 2,000 " +
              "sequence starting point IDs of 'eval/test_ids_icassp.npy' " +
              "located in ./eval directory. You can also specify the 1-D array "
              "file's location. Any numeric input N (int) > 0 will randomly "
              "select N IDs. Default is 'icassp'.")
@click.option('--k_probe', '-k', default=20, type=click.INT,
              help="Top k search for each segment. Default is 20")
@click.option('--display_interval', '-dp', default=10, type=click.INT,
              help="Display interval. Default is 10, which updates the table" +
              " every 10 queries.")
def eval_faiss(emb_dir,
               emb_dummy_dir=None,
               index_type='ivfpq',
               nogpu=False,
               stretch_factor=None,
               max_train=1e7,
               test_ids='icassp',
               test_seq_len='1 3 5 9 11 19',
               k_probe=20,
               display_interval=5):
    """
    Segment/sequence-wise audio search experiment and evaluation: implementation based on FAISS.

    ex) python eval.py EMB_DIR --index_type ivfpq

    EMB_DIR: Directory where {query, db, dummy_db}.mm files are located. The 'raw_score.npy' and 'test_ids.npy' will be also created in the same directory.



    """
    test_seq_len = np.asarray(list(map(int, test_seq_len.split())))  # '1 3 5' --> [1, 3, 5]

    # Load items from {query, db, dummy_db}
    query, query_shape = load_memmap_data(emb_dir, 'query')
    db, db_shape = load_memmap_data(emb_dir, 'db')
    if emb_dummy_dir is None:
        emb_dummy_dir = emb_dir
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir, 'dummy_db')
    """ ----------------------------------------------------------------------
    FAISS index setup

        dummy: 10 items.
        db: 5 items.
        query: 5 items, corresponding to 'db'.

        index.add(dummy_db); index.add(db) # 'dummy_db' first

               |------ dummy_db ------|
        index: [d0, d1, d2,..., d8, d9, d11, d12, d13, d14, d15]
                                       |--------- db ----------|

                                       |--------query ---------|
                                       [q0,  q1,  q2,  q3,  q4]

    • The set of ground truth IDs for q[i] will be (i + len(dummy_db))

    ---------------------------------------------------------------------- """
    # Create and train FAISS index
    index = get_index(index_type=index_type,
                      train_data=dummy_db,
                      train_data_shape=dummy_db.shape,
                      use_gpu=(not nogpu),
                      max_nitem_train=max_train)

    # Add items to index
    start_time = time.time()

    print("Adding items to the index...")
    index.add(dummy_db)
    print(f'{len(dummy_db)} items from dummy DB')
    index.add(db)
    print(f'{len(db)} items from reference DB')

    t = time.time() - start_time
    print(f'Added total {index.ntotal} items to DB. {t:>4.2f} sec.')

    """ ----------------------------------------------------------------------
    We need to prepare a merged {dummy_db + db} memmap:

    • Calcuation of sequence-level matching score requires reconstruction of
      vectors from FAISS index.
    • Unfortunately, current faiss.index.reconstruct_n(id_start, id_stop)
      supports only CPU index.
    • We prepare a fake_recon_index through the on-disk method.

    ---------------------------------------------------------------------- """
    # Prepare fake_recon_index
    del dummy_db
    start_time = time.time()

    fake_recon_index, index_shape = load_memmap_data(emb_dummy_dir,
                                                     'dummy_db',
                                                     append_extra_length=db_shape[0],
                                                     display=False)
    fake_recon_index[dummy_db_shape[0]:dummy_db_shape[0] + db_shape[0], :] = db[:, :]
    fake_recon_index.flush()

    t = time.time() - start_time
    print(f'Created fake_recon_index, total {fake_recon_index.shape[0]} items. {t:>4.2f} sec.')

    # Get test_ids
    print(f'test_id: \033[93m{test_ids}\033[0m,  ', end='')
    if test_ids.lower() == 'all':
        # Will use all segments in query/db set as starting point and 
        # evaluate the performance for each test_seq_len.
        test_ids = np.arange(0, len(query) - max(test_seq_len), 1)
    elif test_ids.lower() == 'icassp':
        test_ids = np.load(glob.glob('./**/test_ids_icassp2021.npy', recursive=True)[0])
    elif test_ids.isnumeric():
        # Will use random segments in query/db set as starting point and
        # evaluate the performance for each test_seq_len. This does not guarantee
        # getting a sample from each track
        test_ids = np.random.permutation(len(query) - max(test_seq_len))[:int(test_ids)]
    else:
        test_ids = np.load(test_ids)

    n_test = len(test_ids)
    gt_ids  = test_ids + dummy_db_shape[0]
    print(f'n_test: \033[93m{n_test:n}\033[0m')

    """ Segment/sequence-level search & evaluation """
    # Define metric
    top1_exact = np.zeros((n_test, len(test_seq_len))).astype(int) # (n_test, test_seg_len)
    top1_near = np.zeros((n_test, len(test_seq_len))).astype(int)
    top3_exact = np.zeros((n_test, len(test_seq_len))).astype(int)
    top10_exact = np.zeros((n_test, len(test_seq_len))).astype(int)
    top1_song = np.zeros((n_test, len(test_seq_len))).astype(int)
    top3_song = np.zeros((n_test, len(test_seq_len))).astype(int)
    top10_song = np.zeros((n_test, len(test_seq_len))).astype(int)

    if stretch_factor is not None:
        query_segments_info = pd.read_csv(os.path.join(emb_dir, "query_segments.csv"))
        db_segments_info = pd.read_csv(os.path.join(emb_dir, "db_segments.csv"))
        query_segments_info["track_id"] = query_segments_info.apply(lambda x: x["filename"].split('/')[-1], axis=1)
        db_segments_info["track_id"] = db_segments_info.apply(lambda x: x["filename"].split('/')[-1], axis=1)
    else:
        db_segments_info = pd.read_csv(os.path.join(emb_dir, "nosource_query_segments.csv"))

    scr = curses.initscr()
    pt = PrintTable(scr=scr,
                    test_seq_len=test_seq_len,
                    row_names=['Top1 exact', 'Top1 near', 'Top3 exact', 'Top10 exact',
                               'Top1 song', 'Top3 song', 'Top10 song'])
    start_time = time.time()
    for ti, test_id in enumerate(test_ids):
        gt_id = gt_ids[ti]
        gt_info = db_segments_info[db_segments_info["segment_id"]==test_id]
        if gt_info.shape[0] == 1:
            gt_track_id = gt_info.iloc[0]["track_id"]
            gt_segids = db_segments_info[db_segments_info[
                "track_id"]==gt_track_id]["segment_id"].values + dummy_db_shape[0]
        else:
            sys.exit("ERROR: gt_id not found / repeated gt_id found")
        query_test_id = query_id_from_db_id(db_seg_id=test_id,
                                            query_info=query_segments_info,
                                            db_info=db_segments_info,
                                            stretch_factor=stretch_factor)
        for si, sl in enumerate(test_seq_len):
            if stretch_factor == 1:
                assert test_id <= len(query)
            q = query[query_test_id:(query_test_id + sl), :] # shape(q) = (length, dim)
            # segment-level top k search for each segment
            _, I = index.search(q, k_probe) # _: distance, I: result IDs matrix

            # storing in python dictionaries since python>3.7 preserves insertion order in dicts.
            track_ids = {}
            for row in range(I.shape[0]):
                for col in range(I.shape[1]):
                    track_id = get_ref_from_db_id(I[row,col], db_segments_info, dummy_db_shape)
                    if track_id != "dummy":
                        if track_id not in track_ids.keys():
                            track_ids[track_id] = {row: I[row,col]}
                        else:
                            if row not in track_ids[track_id].keys():
                                # pick the best candidate for each reference from faiss output
                                track_ids[track_id][row] = I[row,col]
            factors = {}
            for k,v in track_ids.items():
                sf, _ = estimate_stretching_factor(np.array(list(v.keys())), np.array(list(v.values())))
                if sf < 0.5 or sf > 2:
                    """Outliers"""
                    sf = 1
                factors[k] = sf
            # get candidates sequences
            factors["dummy"] = 1
            candidates = []
            for row in range(I.shape[0]):
                for col in range(I.shape[1]):
                    db_id = I[row,col]
                    track_id =  get_ref_from_db_id(db_id, db_segments_info, dummy_db_shape)
                    factor = factors[track_id]
                    start = max(0, np.round(db_id-row/factor).astype(int))
                    if start > dummy_db_shape[0] + db_shape[0] - sl//factor - 1:
                        start = dummy_db_shape[0] + db_shape[0] - sl//factor - 1
                    ids = np.array(range(sl))
                    cid_seq = np.round(start + ids/factor).astype(int)
                    candidates.append(cid_seq)

            # offset compensation to get the start IDs of candidate sequences
            # for offset in range(len(I)):
            #     I[offset, :] -= offset
            # # unique candidates
            # candidates = np.unique(I[np.where(I >= 0)])   # ignore id < 0
            """ Sequence match score """
            _scores = np.zeros(len(candidates))
            for ci, cid_seq in enumerate(candidates):
                _scores[ci] = np.mean(
                    np.diag(
                        np.dot(q, fake_recon_index[cid_seq, :].T)
                        # np.dot(q, fake_recon_index[cid:cid + sl, :].T)
                        )
                    )

            """ Evaluate """
            candidates = np.array(candidates)
            pred_ids = candidates[np.argsort(-_scores)[:10]]
            # pred_id = candidates[np.argmax(_scores)] <-- only top1-hit
            if stretch_factor != 1:
                if pred_ids[0][0] in gt_segids: # pred_ids[0][0] compare the first id of the pred_ids sequence
                    top1_song[ti, si] = int(1)
                if len(set(pred_ids[:3][0])&(set(gt_segids))) > 0:
                    top3_song[ti, si] = int(1)
                if len(set(pred_ids[:10][0])&(set(gt_segids))) > 0:
                    top10_song[ti, si] = int(1)
            else:
                top1_exact[ti, si] = int(gt_id == pred_ids[0][0])
                top1_near[ti, si] = int(pred_ids[0][0] in [gt_id - 1, gt_id, gt_id + 1])
                top1_song[ti, si] = int(pred_ids[0][0] in gt_segids)
                if len(set(pred_ids[:3][0]) & set(gt_segids)) >= 1:
                    top3_song[ti, si] = 1
                if len(set(pred_ids[:10][0]) & set(gt_segids)) >= 1:
                    top10_song[ti, si] = 1

                top3_exact[ti, si] = int(gt_id in pred_ids[:3][0])
                top10_exact[ti, si] = int(gt_id in pred_ids[:10][0])


        if (ti != 0) & ((ti % display_interval) == 0):
            avg_search_time = (time.time() - start_time) / display_interval \
                / len(test_seq_len)
            top1_exact_rate = 100. * np.mean(top1_exact[:ti + 1, :], axis=0)
            top1_near_rate = 100. * np.mean(top1_near[:ti + 1, :], axis=0)
            top3_exact_rate = 100. * np.mean(top3_exact[:ti + 1, :], axis=0)
            top10_exact_rate = 100. * np.mean(top10_exact[:ti + 1, :], axis=0)
            top1_song_rate = 100 * np.mean(top1_song[:ti + 1, :], axis=0)
            top3_song_rate = 100 * np.mean(top3_song[:ti + 1, :], axis=0)
            top10_song_rate = 100 * np.mean(top10_song[:ti + 1, :], axis=0)

            pt.update_counter(ti, n_test, avg_search_time * 1000.)
            pt.update_table((top1_exact_rate,
                             top1_near_rate,
                             top3_exact_rate,
                             top10_exact_rate,
                             top1_song_rate,
                             top3_song_rate,
                             top10_song_rate))
            start_time = time.time() # reset stopwatch

    # Summary
    top1_exact_rate = 100. * np.mean(top1_exact, axis=0)
    top1_near_rate = 100. * np.mean(top1_near, axis=0)
    top3_exact_rate = 100. * np.mean(top3_exact, axis=0)
    top10_exact_rate = 100. * np.mean(top10_exact, axis=0)
    top1_song_rate = 100 * np.mean(top1_song, axis=0)
    top3_song_rate = 100 * np.mean(top3_song, axis=0)
    top10_song_rate = 100 * np.mean(top10_song, axis=0)

    pt.update_counter(ti, n_test, avg_search_time * 1000.)
    pt.update_table((top1_exact_rate, top1_near_rate, top3_exact_rate, top10_exact_rate,
                     top1_song_rate, top3_song_rate, top10_song_rate))
    pt.close_table() # close table and print summary
    del fake_recon_index, query, db
    np.save(f'{emb_dir}/raw_score.npy',
            np.concatenate((top1_exact, top1_near, top3_exact, top10_exact,
                            top1_song, top3_song, top10_song),axis=1)
            )
    np.save(f'{emb_dir}/test_ids.npy', test_ids)
    print(f'Saved test_ids and raw score to {emb_dir}.')

if __name__ == "__main__":
    curses.wrapper(eval_faiss())
