
import os

if __name__ == "__main__":
    query_dir_root = "/datasets/neural-audio-fp-dataset/music/test-query-db-500-30s/query_stretch/"
    exp_name = "peaknetfp"
    ckpt_index = "100"
    emb_dir_root = "logs/emb/"
    query_dirs = sorted(os.listdir(query_dir_root))
    with open(f"evaluate_{exp_name}.sh", "w") as f:
        for i, q_dir in enumerate(query_dirs):
            stretch_factor = float(q_dir.split("-")[-1].replace("_","."))
            query_dir = os.path.join(query_dir_root, q_dir)
            output_dir = os.path.join(emb_dir_root, exp_name, q_dir, ckpt_index)
            cmd = (
                f"python run.py evaluate {exp_name} {ckpt_index} "
                f"-c config/{exp_name}.yaml --emb_dir {output_dir} --stretch_factor {stretch_factor}"
            )
            f.write(cmd + "\n")

# python run.py evaluate peaknetfp 100 \
#                       -c config/peaknetfp.yaml
#                       --emb_dir logs/emb/peaknetfp/tempo-0_500/100
#                       --stretch_factor 0.5