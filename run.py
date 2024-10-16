# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""run.py"""

import os
import sys
import pathlib
import click
import yaml
import logging

import numpy as np
import pandas as pd
import faiss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def load_config(config_filepath, display=True):
    if display:
        if os.path.exists(config_filepath):
            print(f"cli: Configuration from {config_filepath}")
        else:
            sys.exit(f"cli: ERROR! Configuration file {config_filepath} is missing!!")

    with open(config_filepath, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def update_config(cfg, key1: str, key2: str, val):
    cfg[key1][key2] = val
    return cfg


def print_config(cfg):
    os.system("")
    print("\033[36m" + yaml.dump(cfg, indent=4, width=120, sort_keys=False) + "\033[0m")
    return


def set_gpu(max_memory: float = 0.05) -> str:
    import tensorflow as tf
    import GPUtil

    print("TF version: {}".format(tf.__version__))
    gpus_available = GPUtil.getAvailable(limit=4, maxMemory=max_memory)
    # setting gpu for tensorflow
    try:
        gpu_index = gpus_available[0]
    except IndexError:
        raise ValueError("No GPU available!!")
    print("Using the first available GPU: {}".format(gpu_index))
    print("Physical devices: {}".format(tf.config.list_physical_devices("GPU")))
    tf.config.set_visible_devices(
        tf.config.list_physical_devices("GPU")[gpu_index], "GPU"
    )
    return


@click.group()
def cli():
    """
    train-> generate-> evaluate.

    How to use each command: \b\n
        python run.py COMMAND --help

    """
    return


""" Train """


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    default="default",
    type=click.STRING,
    help="Name of model configuration located in './config/.'",
)
@click.option("--max_epoch", default=None, type=click.INT, help="Max epoch.")
def train(config, max_epoch):
    """Train a neural audio fingerprinter.

    ex) python run.py train -c CONFIG_FILE --max_epoch=100

    NOTE: If './LOG_ROOT_DIR/checkpoint/CHECKPOINT_NAME already exists, the training will resume from the latest checkpoint in the directory.
    """

    from model.utils.config_gpu_memory_lim import allow_gpu_memory_growth
    from model.trainer import set_seed, trainer

    cfg = load_config(config)
    if max_epoch:
        update_config(cfg, "TRAIN", "MAX_EPOCH", max_epoch)
    print_config(cfg)
    if cfg["TRAIN"]["SEED"]:
        set_seed(cfg["TRAIN"]["SEED"])  # set SEED
    else:
        set_seed()
    # allow_gpu_memory_growth()
    checkpoint_name = cfg["CHECKPOINT_NAME"]
    trainer(cfg, checkpoint_name)


# Search and evalutation
@cli.command()
@click.argument("checkpoint_name", required=True)
@click.argument("checkpoint_index", required=True)
@click.option(
    "--emb_dir",
    default=None,
    required=False,
    type=click.STRING,
    help="Embeddings directory",
)
@click.option(
    "--config",
    "-c",
    default="default",
    required=False,
    type=click.STRING,
    help="Name of the model configuration file located in 'config/'."
    + " Default is 'default'.",
)
@click.option(
    "--index_type",
    "-i",
    default="ivfpq",
    type=click.STRING,
    help="Index type must be one of {'L2', 'IVF', 'IVFPQ', "
    + "'IVFPQ-RR', 'IVFPQ-ONDISK', HNSW'}",
)
@click.option(
    "--test_seq_len",
    default="1 3 5 9 11 19",
    type=click.STRING,
    help="A set of different number of segments to test. "
    + "Numbers are separated by spaces. Default is '1 3 5 9 11 19',"
    + " which corresponds to '1s, 2s, 3s, 5s, 6s, 10s'.",
)
@click.option(
    "--test_ids",
    "-t",
    default="icassp",
    type=click.STRING,
    help="One of {'all', 'icassp', 'path/file.npy', (int)}. If 'all', "
    + "test all IDs from the test. If 'icassp', use the 2,000 "
    + "sequence starting point IDs of 'eval/test_ids_icassp.npy' "
    + "located in ./eval directory. You can also specify the 1-D array "
    "file's location. Any numeric input N (int) > 0 will perform "
    "search test at random position (ID) N times. Default is 'icassp'.",
)
@click.option(
    "--nogpu", default=False, is_flag=True, help="Use this flag to use CPU only."
)
@click.option(
    "--stretch_factor",
    "-sf",
    default=1,
    type=click.FLOAT,
    help="Specify stretching factor of the queries.",
)
def evaluate(
    checkpoint_name,
    checkpoint_index,
    emb_dir,
    config,
    index_type,
    test_seq_len,
    test_ids,
    nogpu,
    stretch_factor,
):
    """Search and evalutation.

    ex) python run.py evaluate CHECKPOINT_NAME CHECKPOINT_INDEX

    With options: \b\n

    ex) python run.py evaluate CHECKPOINT_NAME CHEKPOINT_INDEX -i ivfpq -t 3000 --nogpu

    • Currently, the 'evaluate' command does not reference any information other
    than the output log directory from the config file.
    """
    from eval.eval_faiss import eval_faiss

    cfg = load_config(config)
    if emb_dir == None:
        emb_dir = os.path.join(
            cfg["DIR"]["OUTPUT_ROOT_DIR"], checkpoint_name, str(checkpoint_index)
        )

    if nogpu:
        if stretch_factor != 1:
            eval_faiss(
                [
                    emb_dir,
                    "--index_type",
                    index_type,
                    "--test_seq_len",
                    test_seq_len,
                    "--test_ids",
                    test_ids,
                    "--nogpu",
                    "-sf",
                    stretch_factor,
                ]
            )
        else:
            eval_faiss(
                [
                    emb_dir,
                    "--index_type",
                    index_type,
                    "--test_seq_len",
                    test_seq_len,
                    "--test_ids",
                    test_ids,
                    "--nogpu",
                ]
            )
    else:
        if stretch_factor != 1:
            eval_faiss(
                [
                    emb_dir,
                    "--index_type",
                    index_type,
                    "--test_seq_len",
                    test_seq_len,
                    "--test_ids",
                    test_ids,
                    "-sf",
                    stretch_factor,
                ]
            )
        else:
            eval_faiss(
                [
                    emb_dir,
                    "--index_type",
                    index_type,
                    "--test_seq_len",
                    test_seq_len,
                    "--test_ids",
                    test_ids,
                ]
            )


# Generate fingerprint (after training)
@cli.command()
@click.argument("checkpoint_name", required=True)
@click.argument("checkpoint_index", required=False)
@click.option(
    "--config",
    "-c",
    default="default",
    required=False,
    type=click.STRING,
    help="Path of the model configuration file located in 'config/'."
    + " Default is 'default'",
)
@click.option(
    "--source",
    "-s",
    default=None,
    type=click.STRING,
    required=False,
    help="Sources. It can be a .lst file or a path pointing to a "
    "custom source root directory. The source must be 16-bit "
    "8 Khz mono WAV. This is only useful when constructing a database"
    " without synthesizing queries.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.STRING,
    required=False,
    help="Root directory where the generated embeddings (uncompressed)"
    + " will be stored. Default is OUTPUT_ROOT_DIR/CHECKPOINT_NAME "
    + "defined in config.",
)
@click.option(
    "--query_dir",
    default=None,
    type=click.STRING,
    required=False,
    help="The queries directory. This is used for time stretching "
    + "experiments where we have multiple queries for the same db.",
)
@click.option(
    "--skip_dummy",
    default=False,
    is_flag=True,
    help="Exclude dummy-DB from the default source.",
)
def generate(
    checkpoint_name, checkpoint_index, config, source, output, skip_dummy, query_dir
):
    """Generate fingerprints from a saved checkpoint.

    ex) python run.py generate CHECKPOINT_NAME

    With custom config: \b\n
        python run.py generate CHECKPOINT_NAME -c CONFIG_NAME

    • If CHECKPOINT_INDEX is not specified, the latest checkpoint in the OUTPUT_ROOT_DIR will be loaded.
    • The default value for the fingerprinting source is [TEST_DUMMY_DB] and [TEST_QUERY_DB] specified in config file.
    """
    from model.utils.config_gpu_memory_lim import allow_gpu_memory_growth
    from model.generate import generate_fingerprint
    # import tensorflow as tf

    cfg = load_config(config)
    allow_gpu_memory_growth()
    if source:
        if os.path.isdir(source):
            source_type = "dir"
        elif os.path.isfile(source) and source.split(".")[-1] == "lst":
            source_type = "list"
        elif os.path.isfile(source) and source.split(".")[-1] != "lst":
            source_type = "file"
        else:
            print("ERROR: Unknown source")
            sys.exit()
    else:
        source_type = None
    generate_fingerprint(
        cfg=cfg,
        checkpoint_name=checkpoint_name,
        checkpoint_index=checkpoint_index,
        source=source,
        output_root_dir=output,
        skip_dummy=skip_dummy,
        source_type=source_type,
        query_dir=query_dir,
    )


# Create index
@cli.command()
@click.argument("path_data", required=True)
@click.argument("path_shape", required=True)
@click.argument("index_path", required=True)
@click.option(
    "--config",
    "-c",
    default="default",
    required=False,
    type=click.STRING,
    help="Path of the model configuration file located in 'config/'."
    + " Default is 'default'",
)
def index(path_data, path_shape, index_path, config):
    """Create FAISS index from fingerprint memap file.

    ex) python run.py index PATH_DATA PATH_SHAPE INDEX_PATH -c CONFIG
    args:
        PATH_DATA: Path of the fp embeddings memap file.
        PATH_SHAPE: Path of the fp embeddings shape npy file.
        INDEX_PATH: Path where the index will be stored.
        CONFIG: Config file path
    """
    from model.utils.config_gpu_memory_lim import allow_gpu_memory_growth
    from model.index_creator import create_index

    cfg = load_config(config)
    allow_gpu_memory_growth()
    create_index(path_data, path_shape, index_path, cfg)


# match
@cli.command()
@click.argument("query_bname", required=True)
@click.argument("query_fp_path", required=True)
@click.argument("refs_fp_path", required=True)
@click.argument("index_path", required=True)
@click.option(
    "--config",
    "-c",
    default="default",
    required=False,
    type=click.STRING,
    help="Path of the model configuration file located in 'config/' ."
    + " Default is 'default'",
)
@click.option(
    "--extension",
    "-e",
    default=".mp3",
    required=False,
    type=click.STRING,
    help="Extension of the original audios.",
)
def match(query_bname, query_fp_path, refs_fp_path, index_path, config, extension):
    """Match query fingerprint with FAISS index and compare with GT.

    ex) python run.py match QUERY_BNAME QUERY_FP_PATH REFS_FP_PATH INDEX_PATH -c CONFIG -e EXTENSION
    args:
        QUERY_BNAME: Audio file basename (without dir)
        QUERY_FP_PATH: Path of the query fp embedding memap file.
        REFS_FP_PATH: Path of the references fp embeddings memap file.
        INDEX_PATH: Path where the index is stored.
        CONFIG: Config file path
        EXTENSION: Refs extension
    """
    from model.utils.config_gpu_memory_lim import allow_gpu_memory_growth
    from model.matcher import Matcher
    from eval.eval_faiss import load_memmap_data

    cfg = load_config(config, display=False)
    allow_gpu_memory_growth()

    refs_segments_path = os.path.join(
        os.path.dirname(refs_fp_path), "refs_segments.csv"
    )
    references_segments = pd.read_csv(refs_segments_path)
    index = faiss.read_index(index_path, 2)  # 2 == readonly
    query, _ = load_memmap_data(
        os.path.dirname(query_fp_path),
        query_bname,
        append_extra_length=None,
        shape_only=False,
        display=False,
    )
    references_fp, _ = load_memmap_data(
        os.path.dirname(refs_fp_path),
        "custom_source",
        append_extra_length=None,
        shape_only=False,
        display=False,
    )
    matcher = Matcher(cfg, index, references_segments)
    formatted_matches = matcher.match(query, references_fp)
    print(
        '"Query","Query begin time","Query end time","Reference",'
        '"Reference begin time","Reference end time","Confidence"'
    )
    print('"","","","","","",""')
    for qmatch in formatted_matches:
        qmatch.update({"query": query_bname[:-4]})  # trim added .wav extension
        print(
            '"{}","{}","{}","{}","{}","{}","{}"'.format(
                qmatch["query"],
                qmatch["query_start"],
                qmatch["query_end"],
                qmatch["ref"] + extension,
                qmatch["ref_start"],
                qmatch["ref_end"],
                qmatch["score"],
            )
        )


if __name__ == "__main__":
    set_gpu()
    logging.basicConfig()
    cli()

########## IF TIME STRETCHING ##########
# Train
# python run.py train -c config/neuralfp.yaml

# Generate
# python run.py generate neuralfp 100 \
#     -c config/neuralfp.yaml \
#     -o logs/emb/neuralfp/tempo-0_900/100 \
#     --query_dir /datasets/neural-audio-fp-dataset/music/test-query-db-500-30s/query_stretch/tempo-0_900

# Evaluation
# python run.py evaluate neuralfp 100 \
#     -c config/neuralfp.yaml \
#     --emb_dir logs/emb/neuralfp/tempo-0_900/100 \
#     --stretch_factor 0.9
