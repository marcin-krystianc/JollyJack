import unittest
import tempfile

import palletjack as pj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import itertools as it
import pyarrow.fs as fs
import os

chunk_size = 10
n_row_groups = 5
n_columns = 7
n_rows = n_row_groups * chunk_size
current_dir = os.path.dirname(os.path.realpath(__file__))

def get_table():
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_rows, n_columns)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(int(x) for x in data[:, i]) for i in range(n_columns)]

    # Optionally, create column names
    column_names = [f'column_{i}' for i in range(n_columns)]

    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, names=column_names)

class TestPalletJack(unittest.TestCase):

    def test_read_column_chunk(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table()

            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            pr = pq.ParquetReader()
            pr.open(path)
            data_org = pr.read_all(use_threads=False)
            # Create an array of zeros with shape (10, 10)
            np_array = np.zeros((n_rows, n_columns), dtype=float, order='F')
            print (dir(pj))

            pj.read_column_chunk(metadata = pr.metadata, parquet_path = path, np_array = np_array, row_idx = 0, column_idx = 1)
            np_array[0, 2] = 2.24        
            v = np_array[0, 1]           
            print (v)        
            print (np_array)

if __name__ == '__main__':
    unittest.main()
