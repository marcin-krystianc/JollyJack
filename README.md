# JollyJack

## Features

- Reading parquet files directly into numpy arrays and torch tensors (fp16, fp32, fp64)
- Compatibility with PalletJack

## Known limitations

- Data cannot contain null values
- Only a local file system is supported

## Required:

- pyarrow  ~= 17.0
 
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
                        , column_indices = range(pr.metadata.num_columns)
                        , pre_buffer = True)

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

row_begin = 0
row_end = 0

for rg in range(pr.metadata.num_row_groups):
    row_begin = row_end
    row_end = row_begin + pr.metadata.row_group(rg).num_rows

    # To define which subset of the numpy array we want read into,
    # we need to create a view which shares underlying memory with the target numpy array
    subset_view = np_array[row_begin:row_end, :] 
    jj.read_into_torch (source = path
                        , metadata = pr.metadata
                        , tensor = tensor
                        , row_group_indices = [rg]
                        , column_indices = range(pr.metadata.num_columns)
                        , pre_buffer = True)


print(np_array)
```
