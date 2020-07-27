import numpy as np


def israel_arctan_function(tau, rho, a, d_a, d_p, d_w, g_a, g_p, g_w):
    """ ### """
    result = (rho) * (
        a
        + (d_a * np.arctan((np.log(tau) - d_p) / d_w))
        + (g_a * np.exp(-0.5 * ((np.log(tau) - g_p) / g_w) ** 2))
    )
    if isinstance(result, np.ndarray):
        result = sum(result)
    return result


def delta_ellipticity(tau, rho=1):
    """ ### """
    a = 0.05333
    d_a = 0.03357
    d_p = 1.628
    d_w = 0.2951
    g_a = 0.09901
    g_p = 0.4553
    g_w = 0.4132
    return israel_arctan_function(tau, rho, a, d_a, d_p, d_w, g_a, g_p, g_w)


def set_min_max(value, min, max):
    """ Fix a value between a minimum and maximum. """
    if value < min:
        return min
    elif max < value:
        return max
    else:
        return value


def update_fits_header_info(
    ext_header,
    parallel_clocker=None,
    serial_clocker=None,
    parallel_traps=None,
    serial_traps=None,
    parallel_ccd_volume=None,
    serial_ccd_volume=None,
):
    """Update a fits header to include the parallel CTI settings.

    Params
    -----------
    ext_header : astropy.io.hdulist
        The opened header of the astropy fits header.
    """

    if parallel_clocker is not None:
        ext_header.set(
            "cte_pite",
            parallel_clocker.iterations,
            "Iterations Used In Correction (Parallel)",
        )

    if serial_clocker is not None:
        ext_header.set(
            "cte_site",
            serial_clocker.iterations,
            "Iterations Used In Correction (Serial)",
        )

        def add_trap(name, traps):
            for i, trap in traps:
                ext_header.set(
                    "cte_pt{}d".format(i),
                    trap.trap_density,
                    "Trap trap {} density ({})".format(i, name),
                )
                ext_header.set(
                    "cte_pt{}t".format(i),
                    trap.trap_lifetime,
                    "Trap trap {} lifetime ({})".format(i, name),
                )

        if parallel_traps is not None:
            add_trap(name="Parallel", traps=parallel_traps)

        if serial_traps is not None:
            add_trap(name="Serial", traps=serial_traps)

        if serial_ccd_volume is not None:
            ext_header.set(
                "cte_swln",
                serial_ccd_volume.well_notch_depth,
                "CCD Well notch depth (Serial)",
            )
            ext_header.set(
                "cte_swlp",
                serial_ccd_volume.well_fill_beta,
                "CCD Well filling power (Serial)",
            )

        if parallel_ccd_volume is not None:
            ext_header.set(
                "cte_pwln",
                parallel_ccd_volume.well_notch_depth,
                "CCD Well notch depth (Parallel)",
            )
            ext_header.set(
                "cte_pwlp",
                parallel_ccd_volume.well_fill_beta,
                "CCD Well filling power (Parallel)",
            )

        return ext_header

    return ext_header
