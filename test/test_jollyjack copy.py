import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import time
import os

row_groups = 1
n_columns = 7
chunk_size = 64
n_rows = row_groups * chunk_size
work_items = 2

parquet_path = "/tmp/my.parquet"

def get_table(n_rows, n_columns, data_type = pa.float32()):
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_rows, n_columns).astype(np.float32)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(data[:, i]).cast(data_type, safe = False) for i in range(n_columns)]
    schema = pa.schema([(f'column_{i}', data_type) for i in range(n_columns)])
    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, schema=schema)

def worker_arrow_row_group():
    pr = pq.ParquetReader()
    pr.open(parquet_path, pre_buffer=True)
    pr.read_row_groups(range(row_groups), use_threads=False)

def genrate_data(n_rows, n_columns, path, compression, dtype):

    table = get_table(n_rows, n_columns, dtype)

    t = time.time()
    print(f"writing parquet file:{path}, columns={n_columns}, row_groups={row_groups}, rows={n_rows}, compression={compression}, dtype={dtype}")
    
    pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, compression=compression, store_schema=False)
    parquet_size = os.stat(path).st_size

    dt = time.time() - t
    print(f"finished writing parquet file in {dt:.2f} seconds")

def measure_reading(worker):

    tt = []
    # measure multiple times and take the fastest run
    for _ in range(0, 11):
        t = time.time()
        worker()
        tt.append(time.time() - t)

    return min(tt)

for dtype in [pa.float16()]:
    
    print(f".")
    genrate_data(n_rows, n_columns, path = parquet_path, compression = None, dtype = dtype)
    print(f"`ParquetReader.read_row_groups`, dtype:{dtype}, duration:{measure_reading(worker_arrow_row_group):.2f} seconds")