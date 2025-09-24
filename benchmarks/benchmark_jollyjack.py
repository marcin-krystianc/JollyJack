
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import humanize
import random
import time
import os

n_files = 5
row_groups = 1
n_columns = 7_000
n_columns_to_read = 1_000
chunk_size = 32_000
n_rows = row_groups * chunk_size

parquet_path = "/tmp/my.parquet"

def worker_arrow_row_group(path):

    pr = pq.ParquetReader()
    pr.open(path)

    table = pr.read_row_groups(row_groups = random.sample(range(row_groups), 1), column_indices = random.sample(range(n_columns), n_columns_to_read))

def genrate_data(n_rows, n_columns, path):

    table = pa.Table.from_pydict({f"col_{j}": np.random.uniform(size = n_rows) for j in range(n_columns) })
    pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)
    parquet_size = os.stat(path).st_size
    print(f"Parquet path={path}, size={humanize.naturalsize(parquet_size)}")

def measure_reading(n_runs):

    tt = []
    for _ in range(n_runs):
        t = time.time()

        for f in range(n_files):
            worker_arrow_row_group(f"{parquet_path}{f}")

        tt.append(time.time() - t)

    return tt

print (f"pyarrow.version = {pa.__version__}")
print (f"pyarrow.file = {pa.__file__}")

for f in range(n_files):
    genrate_data(n_rows, n_columns, path = f"{parquet_path}{f}")

print(f"Reading duration:{measure_reading(10)}")
