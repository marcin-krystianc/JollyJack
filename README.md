# JollyJack

## Features

- Reading parquet files directly into numpy arrays and torch tensors (fp16, fp32, fp64)
- Faster and requiring less memory than vanilla PyArrow
- Compatibility with [PalletJack](https://github.com/marcin-krystianc/PalletJack)

## Known limitations

- Data cannot contain null values

## Required

- pyarrow  ~= 22.0.0
 
JollyJack operates on top of pyarrow, making it an essential requirement for both building and using JollyJack. While our source package is compatible with recent versions of pyarrow, the binary distribution package specifically requires the latest major version of pyarrow.

##  Installation

```
pip install jollyjack
```

## How to use:

### Generating a sample parquet file:
```
import jollyjack as jj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

from pyarrow import fs

chunk_size = 3
n_row_groups = 2
n_columns = 5
n_rows = n_row_groups * chunk_size
path = "my.parquet"

data = np.random.rand(n_rows, n_columns).astype(np.float32)
pa_arrays = [pa.array(data[:, i]) for i in range(n_columns)]
schema = pa.schema([(f'column_{i}', pa.float32()) for i in range(n_columns)])
table =  pa.Table.from_arrays(pa_arrays, schema=schema)
pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)
```

### Generating a numpy array to read into:
```
# Create an array of zeros
np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')
```

### Reading entire file into numpy array:
```
pr = pq.ParquetReader()
pr.open(path)

row_begin = 0
row_end = 0

for rg in range(pr.metadata.num_row_groups):
    row_begin = row_end
    row_end = row_begin + pr.metadata.row_group(rg).num_rows

    # To define which subset of the numpy array we want read into,
    # we need to create a view which shares underlying memory with the target numpy array
    subset_view = np_array[row_begin:row_end, :] 
    jj.read_into_numpy (source = path
                        , metadata = pr.metadata
                        , np_array = subset_view
                        , row_group_indices = [rg]
                        , column_indices = range(pr.metadata.num_columns))

# Alternatively
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy (source = f
                        , metadata = None
                        , np_array = np_array
                        , row_group_indices = range(pr.metadata.num_row_groups)
                        , column_indices = range(pr.metadata.num_columns))
```

### Reading columns in reversed order:
```
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy (source = f
                        , metadata = None
                        , np_array = np_array
                        , row_group_indices = range(pr.metadata.num_row_groups)
                        , column_indices = {i:pr.metadata.num_columns - i - 1 for i in range(pr.metadata.num_columns)})
```

### Reading column 3 into multiple destination columns
```
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy (source = f
                        , metadata = None
                        , np_array = np_array
                        , row_group_indices = range(pr.metadata.num_row_groups)
                        , column_indices = ((3, 0), (3, 1)))
```

### Sparse reading
```
np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy (source = f
                        , metadata = None
                        , np_array = np_array
                        , row_group_indices = [0]
                        , row_ranges = [slice(0, 1), slice(4, 6)]
                        , column_indices = range(pr.metadata.num_columns)
						)
print(np_array)
```

### Using cache options
```
np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')
cache_options = pa.CacheOptions(hole_size_limit = 1024, range_size_limit = 2048, lazy = True)
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy (source = f
                        , metadata = None
                        , np_array = np_array
                        , row_group_indices = [0]
                        , row_ranges = [slice(0, 1), slice(4, 6)]
                        , column_indices = range(pr.metadata.num_columns)
                        , cache_options = cache_options,
                        , pre_buffer = True
						)
print(np_array)
```

### Generating a torch tensor to read into:
```
import torch
# Create a tesnsor and transpose it to get Fortran-style order
tensor = torch.zeros(n_columns, n_rows, dtype = torch.float32).transpose(0, 1)
```

### Reading entire file into the tensor:
```
pr = pq.ParquetReader()
pr.open(path)

jj.read_into_torch (source = path
                    , metadata = pr.metadata
                    , tensor = tensor
                    , row_group_indices = range(pr.metadata.num_row_groups)
                    , column_indices = range(pr.metadata.num_columns)
                    , pre_buffer = True
                    , use_threads = True)

print(tensor)
```

## Benchmarks: CPU-bound

| n_threads | use_threads | pre_buffer | dtype     | PyArrow   | JollyJack |
|-----------|-------------|------------|-----------|-----------|-----------|
| 1         | False       | False      | float     | **2.15s** | **0.50s** |
| 1         | True        | False      | float     | **1.55s** | **0.30s** |
| 1         | False       | True       | float     | **2.47s** | **0.97s** |
| 1         | True        | True       | float     | **1.78s** | **1.09s** |
| 1         | False       | False      | halffloat | **2.31s** | **0.52s** |
| 1         | True        | False      | halffloat | **1.69s** | **0.32s** |
| 1         | False       | True       | halffloat | **2.65s** | **0.98s** |
| 1         | True        | True       | halffloat | **1.96s** | **0.84s** |


## Benchmarks: I/O-bound

| n_threads | use_threads | pre_buffer | dtype     | PyArrow   | JollyJack |
|-----------|-------------|------------|-----------|-----------|-----------|
| 1         | False       | False      | float     | **8.91s** | **7.38s** |
| 1         | True        | False      | float     | **8.20s** | **5.41s** |
| 1         | False       | True       | float     | **6.14s** | **5.54s** |
| 1         | True        | True       | float     | **5.70s** | **5.51s** |
| 1         | False       | False      | halffloat | **9.02s** | **5.96s** |
| 1         | True        | False      | halffloat | **6.77s** | **2.98s** |
| 1         | False       | True       | halffloat | **3.74s** | **2.79s** |
| 1         | True        | True       | halffloat | **3.02s** | **2.73s** |

## Benchmarks: I/O-bound

| n_threads | use_threads | pre_buffer | dtype     | PyArrow   | JollyJack | io_uring  |
|-----------|-------------|------------|-----------|-----------|-----------|-----------|
| 1         | False       | False      | float     | **8.91s** | **7.38s** | **5.45s** |
| 1         | True        | False      | float     | **8.20s** | **5.41s** | **5.45s** |
| 1         | False       | True       | float     | **6.14s** | **5.54s** | **5.39s** |
| 1         | True        | True       | float     | **5.70s** | **5.51s** | **5.39s** |
| 1         | False       | False      | halffloat | **9.02s** | **5.96s** | **2.96s** |
| 1         | True        | False      | halffloat | **6.77s** | **2.98s** | **3.00s** |
| 1         | False       | True       | halffloat | **3.74s** | **2.79s** | **4.34s** |
| 1         | True        | True       | halffloat | **3.02s** | **2.73s** | **4.38s** |
