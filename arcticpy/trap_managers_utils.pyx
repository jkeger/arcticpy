import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport exp


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef void roll_2d_vertical(np.float64_t[:, :] arr) nogil:
    """ Equivalent to np.roll(arr, 1, axis=0) """
    # Unfortunately we cannot use np.zeros without the gil, so good old 
    # fashioned C is used instead
    cdef np.float64_t * arr_last_row = <np.float64_t *> malloc(
        sizeof(np.float64_t) * arr.shape[1]
    )
    cdef np.int64_t i, j

    # Copy the last row
    for j in range(0, arr.shape[1], 1):
        arr_last_row[j] = arr[arr.shape[0] - 1, j]

    # Move all the rows up an index
    for i in range(1, arr.shape[0], 1):
        for j in range(0, arr.shape[1], 1):
            arr[arr.shape[0] - i, j] = arr[arr.shape[0] - i - 1, j]

    # Put the copied last row into the 0 index
    for j in range(0, arr.shape[1], 1):
        arr[0, j] = arr_last_row[j]

    free(arr_last_row)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.int64_t cy_value_in_cumsum(np.float64_t value, np.float64_t[:] arr):
    """ Return whether or not a value is in the cumulative sum of an array. """
    cdef np.float64_t total = arr[0]
    cdef np.int64_t i
    
    with nogil:
        for i in range(1, arr.shape[0], 1):
            if value == total:
                return 1
            total += arr[i]

        if value == total:
            return 1
        else:
            return 0


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.float64_t cy_n_trapped_electrons_from_watermarks(
    np.float64_t[:, :] watermarks, np.float64_t[:] n_traps_per_pixel
):
    """ See n_trapped_electrons_from_watermarks() """
    cdef np.float64_t total = 0.0
    cdef np.int64_t i, j
    with nogil:
        for i in range(0, watermarks.shape[0], 1):
            for j in range(1, watermarks.shape[1], 1):
                total += watermarks[i, 0] * watermarks[i, j] * n_traps_per_pixel[j - 1]

    return total


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.int64_t cy_watermark_index_above_cloud_from_cloud_fractional_volume(
    np.float64_t cloud_fractional_volume, 
    np.float64_t[:, :] watermarks, 
    np.int64_t max_watermark_index,
):
    """ See watermark_index_above_cloud_from_cloud_fractional_volume() """
    
    cdef np.float64_t total = 0.0
    cdef np.int64_t i
    cdef np.int64_t index = watermarks.shape[0]
    
    with nogil:
        for i in range(watermarks.shape[0]):
            total += watermarks[i, 0]
            if cloud_fractional_volume <= total:
                index = min(index, i)

        if total < cloud_fractional_volume:
            return max_watermark_index + 1
        else:
            return index


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef void cy_update_watermark_volumes_for_cloud_below_highest(
    np.float64_t[:, :] watermarks,
    np.float64_t cloud_fractional_volume,
    np.int64_t watermark_index_above_cloud,
):
    """ See update_watermark_volumes_for_cloud_below_highest() """

    # The volume and cumulative volume of the watermark around the cloud volume
    cdef np.float64_t watermark_fractional_volume = watermarks[watermark_index_above_cloud, 0]
    cdef np.float64_t cumulative_watermark_fractional_volume = 0.0
    cdef np.float64_t old_fractional_volume = 0.0
    cdef np.int64_t i, j

    with nogil:
        for i in range(0, watermark_index_above_cloud + 1, 1):
            cumulative_watermark_fractional_volume += watermarks[i, 0]

        # Move one new empty watermark to the start of the list
        roll_2d_vertical(watermarks)

        # Re-set the relevant watermarks near the start of the list
        if watermark_index_above_cloud == 0:
            for j in range(0, watermarks.shape[1], 1):
                watermarks[0, j] = watermarks[1, j]
        else:
            for i in range(0, watermark_index_above_cloud + 1, 1):
                for j in range(0, watermarks.shape[1], 1):
                    watermarks[i, j] = watermarks[i + 1, j]

        # Update the new split watermarks' volumes
        old_fractional_volume = watermark_fractional_volume
        watermarks[watermark_index_above_cloud, 0] = cloud_fractional_volume - (
            cumulative_watermark_fractional_volume - watermark_fractional_volume
        )
        watermarks[watermark_index_above_cloud + 1, 0] = (
            old_fractional_volume - watermarks[watermark_index_above_cloud, 0]
        )


# @cython.boundscheck(False)
# @cython.nonecheck(False)
# @cython.wraparound(False)
# cpdef void cy_collapse_redundant_watermarks(
#     np.float64_t[:, :] watermarks,
#     np.float64_t[:, :] watermarks_copy,
#     # watermarks,
#     # watermarks_copy,
#     filled_watermark_value,
#     do_copy,
# ):
#     """ See collapse_redundant_watermarks() """
#     # Number of trap species
#     num_traps = watermarks.shape[1] - 1
# 
#     # Find the first watermark that is not completely filled for all traps
#     watermark_index_not_filled = min(
#         [
#             np.argmax(watermarks[:, 1 + i_trap] != filled_watermark_value)
#             for i_trap in range(num_traps)
#         ]
#     )
# 
#     # Skip if none or only one are completely filled
#     if watermark_index_not_filled <= 1:
#         return
# 
#     # Total fractional volume of filled watermarks
#     fractional_volume_filled = np.sum(watermarks[:watermark_index_not_filled, 0])
# 
#     # Combined fill values
#     if do_copy:
#         # Multiple trap species
#         if 1 < num_traps:
#             axis = 1
#         else:
#             axis = None
#         copy_fill_values = np.sum(
#             watermarks_copy[:watermark_index_not_filled, 0]
#             * watermarks_copy[:watermark_index_not_filled, 1:].T,
#             axis=axis,
#         ) / np.sum(watermarks_copy[:watermark_index_not_filled, 0])
# 
#     # Remove the no-longer-needed overwritten watermarks
#     watermarks[:watermark_index_not_filled, :] = 0
#     if do_copy:
#         watermarks_copy[:watermark_index_not_filled, :] = 0
# 
#     # Move the no-longer-needed watermarks to the end of the list
#     watermarks = np.roll(watermarks, 1 - watermark_index_not_filled, axis=0)
#     if do_copy:
#         watermarks_copy = np.roll(
#             watermarks_copy, 1 - watermark_index_not_filled, axis=0
#         )
# 
#     # Edit the new first watermark
#     watermarks[0, 0] = fractional_volume_filled
#     watermarks[0, 1:] = filled_watermark_value
#     if do_copy:
#         watermarks_copy[0, 0] = fractional_volume_filled
#         watermarks_copy[0, 1:] = copy_fill_values
