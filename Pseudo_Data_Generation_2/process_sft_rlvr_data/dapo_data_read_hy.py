
import datasets
from datasets import Dataset, DatasetDict, load_from_disk

parquet_file = '/path/to/data/neo_nlvr_training/data/public_playground_rlvr'
dataframe = datasets.load_dataset(parquet_file)["train"]
data_source = "/".join(parquet_file.split("/")[-2:])
n=dataframe.num_rows
print(n)
subset = dataframe.select(range(n // 4))
print(subset.num_rows)

loaded_ds = load_from_disk(parquet_file)
train = loaded_ds["train"]

# local_dataset_path = '/path/to/data/neo_nlvr_training/data/neo_playground_rlvr'
# dataset = datasets.load_dataset(local_dataset_path, "default")
# train_dataset = dataset

print(1)