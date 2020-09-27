import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef void roll_2d_vertical(np.float64_t[:, :] arr) nogil:
    # Unfortunately we cannot use np.zeros without the gil, so good old fashioned
    # C is used instead
    cdef np.float64_t * arr_lr = <np.float64_t *> malloc(sizeof(np.float64_t) * arr.shape[1])
    cdef np.int64_t i, j

    # Copy the last row
    for j in range(0, arr.shape[1], 1):
        arr_lr[j] = arr[arr.shape[0]-1, j]

    # Move all the rows up an index
    for i in range(1, arr.shape[0], 1):
        for j in range(0, arr.shape[1], 1):
            arr[arr.shape[0]-i, j] = arr[arr.shape[0]-i-1, j]

    # Put the copied last row into the 0 index
    for j in range(0, arr.shape[1], 1):
        arr[0, j] = arr_lr[j]

    free(arr_lr)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def cy_n_trapped_electrons_from_watermarks(np.float64_t[:, :] watermarks, np.float64_t[:] n_traps_per_pixel):
    """ Sum the total number of electrons currently held in traps.
    Parameters
    ----------
        watermarks : np.ndarray The watermarks.

    See initial_watermarks_from_n_pixels_and_total_traps().
    """
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
def cy_update_watermark_volumes_for_cloud_below_highest(
        np.float64_t[:, :] watermarks,
        np.float64_t cloud_fractional_volume,
        np.int64_t watermark_index_above_cloud):

    # The volume and cumulative volume of the watermark around the cloud volume
    cdef np.float64_t watermark_fractional_volume = watermarks[watermark_index_above_cloud, 0]
    cdef np.float64_t cumulative_watermark_fractional_volume = 0.0
    cdef np.float64_t old_fractional_volume = 0.0
    cdef np.int64_t i, j

    with nogil:
        for i in range(0, watermark_index_above_cloud + 1, 1):
            cumulative_watermark_fractional_volume += watermarks[i, 0]

        # Move one new empty watermark to the start of the list
        #watermarks = np.roll(watermarks, 1, axis=0)
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



@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.int64_t cy_watermark_index_above_cloud_from_cloud_fractional_volume(
        np.float64_t cloud_fractional_volume, np.float64_t[:, :] watermarks, np.int64_t max_watermark_index
):
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
