import warnings
import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from astropy import log
from astropy.table import Table
from ..crossspectrum import AveragedCrossspectrum, normalize_crossspectrum
from ..powerspectrum import AveragedPowerspectrum

from ..gti import cross_two_gtis, bin_intervals_from_gtis


__all__ = ["calculate_FAD_correction", "get_periodograms_from_FAD_results"]


def _get_fourier_intv(lc, start_ind, end_ind):
    """Calculate the Fourier transform of a light curve chunk.

    Parameters
    ----------
    lc : a :class:`Lightcurve` object
        Input light curve

    start_ind : int
        Start index of the light curve chunk

    end_ind : int
        End index of the light curve chunk

    Returns
    -------
    freq : array of floats
        Frequencies of the Fourier transform

    fft : array of complex numbers
        The Fourier transform

    nph : int
        Number of photons in the interval of the light curve

    nbins : int
        Number of bins in the light curve segment

    meancounts : float
        The mean counts/bin in the light curve
    """
    time = lc.time[start_ind:end_ind]
    counts = lc.counts[start_ind:end_ind]

    fourier = scipy.fft.fft(counts)

    freq = scipy.fft.fftfreq(len(time), lc.dt)
    good = freq > 0

    nbins = time.size
    return freq[good], fourier[good], np.sum(counts), nbins


def calculate_FAD_correction(lc1, lc2, segment_size, norm="none", gti=None,
                             plot=False, ax=None, smoothing_alg='gauss',
                             smoothing_length=None, verbose=False,
                             tolerance=0.05, strict=False, all_leahy=False,
                             output_file=None, return_objects=False):
    """Calculate Frequency Amplitude Difference-corrected (cross)power spectra.

    Reference: Bachetti \& Huppenkothen, 2018, ApJ, 853L, 21

    The two input light curve must be strictly simultaneous, and recorded by
    two independent detectors with similar responses, so that the count rates
    are similar and dead time is independent.
    The method does not apply to different energy channels of the same
    instrument, or to the signal observed by two instruments with very
    different responses. See the paper for caveats.

    Parameters
    ----------
    lc1: class:`stingray.ligthtcurve.Lightcurve`
        Light curve from channel 1
    lc2: class:`stingray.ligthtcurve.Lightcurve`
        Light curve from channel 2. Must be strictly simultaneous to ``lc1``
        and have the same binning time. Also, it must be strictly independent,
        e.g. from a different detector. There must be no dead time cross-talk
        between the two light curves.
    segment_size: float
        The final Fourier products are averaged over many segments of the
        input light curves. This is the length of each segment being averaged.
        Note that the light curve must be long enough to have at least 30
        segments, as the result gets better as one averages more and more
        segments.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, default ``none``
        The normalization of the (real part of the) cross spectrum.


    Other parameters
    ----------------
    plot : bool, default False
        Plot diagnostics: check if the smoothed Fourier difference scatter is
        a good approximation of the data scatter.
    ax : :class:`matplotlib.axes.axes` object
        If not None and ``plot`` is True, use this axis object to produce
         the diagnostic plot. Otherwise, create a new figure.
    smoothing_alg : {'gauss', ...}
        Smoothing algorithm. For now, the only smoothing algorithm allowed is
        ``gauss``, which applies a Gaussian Filter from `scipy`.
    smoothing_length : int, default ``segment_size * 3``
        Number of bins to smooth in gaussian window smoothing
    verbose: bool, default False
        Print out information on the outcome of the algorithm (recommended)
    tolerance : float, default 0.05
        Accepted relative error on the FAD-corrected Fourier amplitude, to be
        used as success diagnostics.
        Should be
        ```
        stdtheor = 2 / np.sqrt(n)
        std = (average_corrected_fourier_diff / n).std()
        np.abs((std - stdtheor) / stdtheor) < tolerance
        ```
    strict : bool, default False
        Decide what to do if the condition on tolerance is not met. If True,
        raise a ``RuntimeError``. If False, just throw a warning.
    all_leahy : **deprecated** bool, default False
        Save all spectra in Leahy normalization. Otherwise, leave unnormalized.
    output_file : str, default None
        Name of an output file (any extension automatically recognized by
        Astropy is fine)

    Returns
    -------
    results : class:`astropy.table.Table` object or ``dict`` or ``str``
        The content of ``results`` depends on whether ``return_objects`` is
        True or False.
        If ``return_objects==False``,
        ``results`` is a `Table` with the following columns:

        + pds1: the corrected PDS of ``lc1``
        + pds2: the corrected PDS of ``lc2``
        + cs: the corrected cospectrum
        + ptot: the corrected PDS of lc1 + lc2

        If ``return_objects`` is True, ``results`` is a ``dict``, with keys
        named like the columns
        listed above but with `AveragePowerspectrum` or
        `AverageCrossspectrum` objects instead of arrays.

    """
    if smoothing_length is None:
        smoothing_length = segment_size * 3

    if gti is None:
        gti = cross_two_gtis(lc1.gti, lc2.gti)

    if all_leahy:
        warnings.warn("`all_leahy` is deprecated. Use `norm` instead! "  +
                      " Setting `norm`=`leahy`.", DeprecationWarning)
        norm="leahy"

    lc1.gti = gti
    lc2.gti = gti
    lc1.apply_gtis()
    lc2.apply_gtis()
    summed_lc = lc1 + lc2
    start_inds, end_inds = \
        bin_intervals_from_gtis(gti, segment_size, lc1.time,
                                dt=lc1.dt)
    freq = 0
    # These will be the final averaged periodograms. Initializing with a single
    # scalar 0, but the final products will be arrays.
    pds1 = 0
    pds2 = 0
    ptot = 0
    cs = 0
    n = 0
    nph1_tot = nph2_tot = nph_tot = 0
    average_diff = average_diff_uncorr = 0

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

    for start_ind, end_ind in zip(start_inds, end_inds):
        freq, f1, nph1, nbins1 = _get_fourier_intv(lc1, start_ind,
                                                        end_ind)
        f1_leahy = f1 * np.sqrt(2 / nph1)
        freq, f2, nph2, nbins2 = _get_fourier_intv(lc2, start_ind,
                                                        end_ind)
        f2_leahy = f2 * np.sqrt(2 / nph2)
        freq, ftot, nphtot, nbinstot = \
            _get_fourier_intv(summed_lc, start_ind, end_ind)
        ftot_leahy = ftot * np.sqrt(2 / nphtot)

        nph1_tot += nph1
        nph2_tot += nph2
        nph_tot += nphtot

        fourier_diff = f1_leahy - f2_leahy

        if smoothing_alg == 'gauss':
            smooth_real = gaussian_filter1d(fourier_diff.real ** 2,
                                            smoothing_length)
        else:
            raise ValueError("Unknown smoothing algorithm: {}".format(
                smoothing_alg))

        if plot:
            ax.scatter(freq, fourier_diff.real, s=1)

        p1 = (f1 * f1.conj()).real
        p1 = p1 / smooth_real * 2
        p2 = (f2 * f2.conj()).real
        p2 = p2 / smooth_real * 2
        pt = (ftot * ftot.conj()).real
        pt = pt / smooth_real * 2

        c = (f2 * f1.conj()).real
        c = c / smooth_real * 2


        power1 = normalize_crossspectrum(p1, segment_size, nbins1, nph1,
                                         nph1, norm=norm)
        power2 = normalize_crossspectrum(p2, segment_size, nbins2, nph2,
                                         nph2, norm=norm)
        power_tot = normalize_crossspectrum(pt, segment_size, nbinstot, nphtot,
                                            nphtot, norm=norm)
        cs_power = normalize_crossspectrum(c, segment_size, nbins1, nph1,
                                           nph2, norm=norm)

        if n == 0 and plot:
            ax.plot(freq, smooth_real, zorder=10, lw=3)
            ax.plot(freq, f1_leahy.real, zorder=5, lw=1)
            ax.plot(freq, f2_leahy.real, zorder=5, lw=1)

        ptot += power_tot
        pds1 += power1
        pds2 += power2
        cs += cs_power
        average_diff += fourier_diff / smooth_real ** 0.5 * np.sqrt(2)
        average_diff_uncorr += fourier_diff
        n += 1

    std = (average_diff / n).std()
    stdtheor = 2 / np.sqrt(n)
    stduncorr = (average_diff_uncorr / n).std()
    is_compliant = np.abs((std - stdtheor) / stdtheor) < tolerance
    verbose_string = \
    '''
    -------- FAD correction ----------
    I smoothed over {smoothing_length} power spectral bins
    {n} intervals averaged.
    The uncorrected standard deviation of the Fourier
    differences is {stduncorr} (dead-time affected!)
    The final standard deviation of the FAD-corrected
    Fourier differences is {std}. For the results to be
    acceptable, this should be close to {stdtheor}
    to within {tolerance} %.
    In this case, the results ARE {compl}complying.
    {additional}
    ----------------------------------
    '''.format(smoothing_length=smoothing_length,
               n=n,
               stduncorr=stduncorr,
               std=std,
               stdtheor=stdtheor,
               tolerance=tolerance * 100,
               compl='NOT ' if not is_compliant else '',
               additional='Maybe something is not right.' if not is_compliant else '')

    if verbose and is_compliant:
        log.info(verbose_string)
    elif not is_compliant:
        warnings.warn(verbose_string)

    if strict and not is_compliant:
        raise RuntimeError('Results are not compliant, and `strict` mode '
                           'selected. Exiting.')

    results = Table()

    print("n: " + str(n))

    results['freq'] = freq
    results['pds1'] = pds1 / n
    results['pds2'] = pds2 / n
    results['cs'] = cs / n
    results['ptot'] = ptot / n
    results['fad'] = average_diff / n
    results.meta['fad_delta'] = (std - stdtheor) / stdtheor
    results.meta['is_compliant'] = is_compliant
    results.meta['n'] = n
    results.meta['nph1'] = nph1_tot
    results.meta['nph2'] = nph2_tot
    results.meta['nph'] = nph_tot
    results.meta['norm'] = 'leahy' if all_leahy else 'none'
    results.meta['smoothing_length'] = smoothing_length
    results.meta['df'] = np.mean(np.diff(freq))

    if output_file is not None:
        results.write(output_file, overwrite=True)

    if return_objects:
        result_table = results
        results = {}
        results['pds1'] = \
            get_periodograms_from_FAD_results(result_table, kind='pds1')
        results['pds2'] = \
            get_periodograms_from_FAD_results(result_table, kind='pds2')
        results['cs'] = \
            get_periodograms_from_FAD_results(result_table, kind='cs')
        results['ptot'] = \
            get_periodograms_from_FAD_results(result_table, kind='ptot')
        results['fad'] = result_table['fad']

    return results


def get_periodograms_from_FAD_results(FAD_results, kind='ptot'):
    """Get Stingray periodograms from FAD results.

    Parameters
    ----------
    FAD_results : :class:`astropy.table.Table` object or `str`
        Results from `calculate_FAD_correction`, either as a Table or an output
        file name
    kind : :class:`str`, one of ['ptot', 'pds1', 'pds2', 'cs']
        Kind of periodogram to get (E.g., 'ptot' -> PDS from the sum of the two
        light curves, 'cs' -> cospectrum, etc.)

    Returns
    -------
    results : `AveragedCrossspectrum` or `Averagedpowerspectrum` object
        The periodogram.
    """
    if isinstance(FAD_results, str):
        FAD_results = Table.read(FAD_results)

    if kind.startswith('p') and kind in FAD_results.colnames:
        powersp = AveragedPowerspectrum()
        powersp.nphot = FAD_results.meta['nph']
        if '1' in kind:
            powersp.nphots = FAD_results.meta['nph1']
        elif '2' in kind:
            powersp.nphots = FAD_results.meta['nph2']
    elif kind == 'cs':
        powersp = AveragedCrossspectrum(power_type='real')
        powersp.nphots1 = FAD_results.meta['nph1']
        powersp.nphots2 = FAD_results.meta['nph2']
    else:
        raise ValueError("Unknown periodogram type")

    powersp.freq = FAD_results['freq']
    powersp.power = FAD_results[kind]
    powersp.power_err = np.zeros_like(powersp.power)
    powersp.m = FAD_results.meta['n']
    powersp.df = FAD_results.meta['df']
    powersp.n = len(powersp.freq)
    powersp.norm = FAD_results.meta['norm']

    return powersp
