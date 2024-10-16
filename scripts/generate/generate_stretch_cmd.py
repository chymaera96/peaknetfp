import os

if __name__ == "__main__":
    query_dir_root = "/datasets/neural-audio-fp-dataset/music/test-query-db-500-30s/query_stretch/"
    output_dir_root = "/src/peaknetfp"
    exp_name = "peaknetfp"
    ckpt = exp_name
    ckpt_index = "100"
    emb_dir_root = "logs/emb/"
    query_dirs = sorted(os.listdir(query_dir_root))
    with open(f"generate_{exp_name}.sh", "w") as f:
        for i, q_dir in enumerate(query_dirs):
            query_dir = os.path.join(query_dir_root, q_dir)
            output_dir = os.path.join(emb_dir_root, exp_name, q_dir, ckpt_index)
            cmd = (
                f"python run.py generate {ckpt} {ckpt_index} "
                f"-c config/{exp_name}.yaml -o {output_dir} --query_dir {query_dir}"
            )
            if i==0:
                f.write(cmd + "\n")
            else:
                f.write(cmd + " --skip_dummy" + " \n")