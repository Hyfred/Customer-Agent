
import datasets
from sandbox_fusion import run_code, RunCodeRequest, set_endpoint
from tqdm import tqdm
import json

set_endpoint("http://localhost:8070")


parquet_file = '/home/hongyee/hyenv/neo_rlvr_training/dataset/public_playground_rlvr'
dataframe = datasets.load_dataset(parquet_file)["train"]

seen_qid = set()

def dedup_by_qid(example):
    qid = example["extra_info"]["qid"]
    if qid in seen_qid:
        return False
    seen_qid.add(qid)
    return True

subset = dataframe.filter(dedup_by_qid)

for row in tqdm(subset, desc="Saving trajectories"):
    qid = row["extra_info"]["qid"]
    trajectory = row["extra_info"]["trajectory"]
    traj_str = json.dumps(trajectory, ensure_ascii=False)

    save_traj_code = f"""from data_loader import save_trajectory_file
save_trajectory_file({qid}, {traj_str})
print(1)
"""
    

    print(f"Processing qid={qid}...")
    try:
        res = run_code(
            RunCodeRequest(code=save_traj_code, language='python'),
            max_attempts=3,
            client_timeout=30
        )
        # print("Result:", res)
        # print(res.stderr)
    except Exception as e:
        print("Error:", e)