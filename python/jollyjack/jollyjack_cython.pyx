# distutils: include_dirs = .

import cython
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
cimport numpy as cnp

from cython.operator cimport dereference as deref
from cython.cimports.jollyjack import cjollyjack

from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool

from libc.stdint cimport uint32_t
from pyarrow._parquet cimport *

cpdef void read_into_numpy_f32(parquet_path, FileMetaData metadata, cnp.ndarray np_array, row_group_idx, column_indices, pre_buffer=False):
    cdef string encoded_path = parquet_path.encode('utf8') if parquet_path is not None else "".encode('utf8')
    cdef vector[int] crow_group_indices = [row_group_idx]
    cdef vector[int] ccolumn_indices = column_indices
    cdef uint32_t cstride0_size = np_array.strides[0]
    cdef uint32_t cstride1_size = np_array.strides[1]
    cdef void* cdata = np_array.data
    cdef bool cpre_buffer = pre_buffer

    # Ensure the input is a 2D array
    assert np_array.ndim == 2, f"Unexpected np_array.ndim, {np_array.ndim} != 2"

    # Ensure the row and column indices are within the array bounds
    assert ccolumn_indices.size() == np_array.shape[1], f"Requested to read {ccolumn_indices.size()} columns , but the number of columns in numpy array is {np_array.shape[1]}"
    assert np_array.strides[0] < np_array.strides[1], f"Expected array in a Fortran order"

    # TODO SIZE ?
    with nogil:
        cjollyjack.ReadData(encoded_path.c_str(), metadata.sp_metadata
            , np_array.data, 1
            , cstride0_size, cstride1_size
            , crow_group_indices
            , ccolumn_indices
            , cpre_buffer)
        return

    return
