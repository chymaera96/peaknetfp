#!/usr/bin/env python3
import argparse
import os
import yaml

import numpy as np
import tensorflow as tf

# Local imports
from model.dataset import Dataset
from model.generate import build_fp, load_checkpoint, test_step


def generate_dir_fingerprints(cfg, checkpoint_name, checkpoint_index, source_dir, output_dir, out_name="dummy_db"):
    # 1. Build model
    melspec_layer, fingerprinter = build_fp(cfg)
    checkpoint_root_dir = os.path.join(cfg['DIR']['LOG_ROOT_DIR'], 'checkpoint')
    load_checkpoint(checkpoint_root_dir, checkpoint_name, checkpoint_index, fingerprinter)

    # 2. Build dataset from directory of wav files
    dataset = Dataset(cfg)
    ds = dataset.get_custom_db_ds(source=source_dir, source_type="dir")

    # 3. Prepare memmap for embeddings
    n_items = ds.n_samples
    emb_sz = cfg['MODEL']['EMB_SZ']
    os.makedirs(output_dir, exist_ok=True)
    arr = np.memmap(os.path.join(output_dir, f"{out_name}.mm"),
                    dtype="float32", mode="w+", shape=(n_items, emb_sz))

    # 4. Run through batches
    i, bsz = 0, cfg['BSZ']['TS_BATCH_SZ']
    enq = tf.keras.utils.OrderedEnqueuer(ds, use_multiprocessing=True, shuffle=False)
    enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
              max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])

    while i < len(enq.sequence):
        X, _ = next(enq.get())
        emb = test_step(X, melspec_layer, fingerprinter, cfg)
        arr[i*bsz:(i+1)*bsz, :] = emb.numpy()
        i += 1
    enq.stop()

    arr.flush()
    np.save(os.path.join(output_dir, f"{out_name}_shape.npy"), (n_items, emb_sz))
    print(f"Stored {n_items} embeddings in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate fingerprints from directory of wav files")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint_name", type=str, required=True,
                        help="Checkpoint name (subfolder under LOG_ROOT_DIR/checkpoint/)")
    parser.add_argument("--checkpoint_index", type=int, default=0,
                        help="Checkpoint index (0 = latest)")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Directory containing wav files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated fingerprints")
    parser.add_argument("--out_name", type=str, default="dummy_db",
                        help="Base name for output files (default: dummy_db)")
    args = parser.parse_args()

    # Load config YAML into dict
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    generate_dir_fingerprints(cfg,
                              args.checkpoint_name,
                              args.checkpoint_index,
                              args.source_dir,
                              args.output_dir,
                              args.out_name)


if __name__ == "__main__":
    main()
