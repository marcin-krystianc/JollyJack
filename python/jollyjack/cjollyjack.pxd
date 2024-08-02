from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp cimport bool

from libc.stdint cimport uint32_t
from pyarrow._parquet cimport *

cdef extern from "jollyjack.h":
    cdef void ReadData(const char *parquet_path
        , shared_ptr[CFileMetaData] file_metadata
        , void* data, size_t buffer_size
        , size_t stride0_size, size_t stride1_sizexs
        , const vector[int] row_groups
        , const vector[int] column_indices
        , bool pre_buffer) except + nogil
