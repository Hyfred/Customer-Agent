
from sandbox_fusion import run_code, RunCodeRequest, set_endpoint
from tqdm import tqdm
import json

set_endpoint("http://localhost:8070")

test_code = '''from data_loader import save_trajectory_file,load_trajectory
a,b = load_trajectory('2767')
print(b)'''

res = run_code(
    RunCodeRequest(code=test_code, language='python'),
    max_attempts=3,
    client_timeout=30
)

print(res)