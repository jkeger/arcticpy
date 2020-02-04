import os
import shutil

import numpy as np
import pytest
from astropy.io import fits

import arctic as ac


@pytest.fixture(name="hdr_path")
def test_header_info():
    hdr_path = "{}/test_files/util/header_info/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(hdr_path):
        shutil.rmtree(hdr_path)

    os.mkdir(hdr_path)

    return hdr_path


class TestFitsHeader:
    def test__parallel_and_serial__sets_up_header_info_consistent_with_previous_vis_pf(
        self, hdr_path
    ):

        parallel_clocker = ac.Clocker(sequence=1, iterations=1)

        serial_clocker = ac.Clocker(sequence=1, iterations=2)

        hdu = fits.PrimaryHDU(np.ones((1, 1)), fits.Header())
        hdu.header = ac.util.update_fits_header_info(
            ext_header=hdu.header,
            parallel_clocker=parallel_clocker,
            serial_clocker=serial_clocker,
        )

        hdu.writeto(hdr_path + "hdr.fits")
        hdu = fits.open(hdr_path + "hdr.fits")
        ext_header = hdu[0].header

        assert ext_header["cte_pite"] == 1
        assert ext_header["cte_site"] == 2
