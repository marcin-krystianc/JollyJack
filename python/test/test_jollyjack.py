import unittest
import tempfile

import jollyjack as jj
import palletjack as pj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import os

chunk_size = 10_000
n_row_groups = 3
n_columns = 10000
n_rows = n_row_groups * chunk_size
current_dir = os.path.dirname(os.path.realpath(__file__))

def get_table(n_rows, n_columns, data_type = pa.float32()):
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_rows, n_columns).astype(np.float32)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(data[:, i]) for i in range(n_columns)]
    schema = pa.schema([(f'column_{i}', data_type) for i in range(n_columns)])
    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, schema=schema)

class TestJollyJack(unittest.TestCase):
   
    def test_read_with_palletjack(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows, n_columns)
            pq.write_table(table, path, row_group_size=chunk_size, compression=None, use_dictionary=False, write_statistics=False, store_schema=True, write_page_index=False)
            
            index_path = path + '.index'
            pj.generate_metadata_index(path, index_path)
            
            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all()
            # Create an array of zeros
            np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')

            print("\nEmpty array:")
            print(np_array)
            row_begin = 0
            row_end = 0
            
            for rg in range(pr.metadata.num_row_groups):
                for cp in [[1, 2, 200], list(range(1000, 2000))]:
                    metadata = pj.read_metadata(index_path, row_groups=[1, 0], column_indices=cp)

                    row_begin = rg * chunk_size
                    row_end = (rg  + 1) * chunk_size
                    subset_view = np_array[row_begin:row_end, 0:(len(cp))] 
                    jj.read_into_numpy_f32(metadata = metadata
                                        , parquet_path = path
                                        , np_array = subset_view
                                        , row_group_idx = 0
                                        , column_indices = range(len(cp))
                                        , pre_buffer=True)

            self.assertTrue(np.array_equal(np_array, expected_data))

if __name__ == '__main__':
    unittest.main()
