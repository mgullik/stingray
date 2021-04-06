import numpy as np
import numpy.random as ra
import scipy.interpolate as sci
from stingray import Lightcurve
from stingray.gti import gti_border_bins


def simulate_times(lc, use_spline=False):
    """
    Simulate an event list, by using the inverse CDF method:

    + Assume that the light curve is a probability density (must be positive
      definite)

    + Calculate the CDF from the cumulative sum, and normalize it from 0 to 1

    + Extract N random probability values from 0 to 1

    + Find the CDF values corresponding to these N values

    + Find the times corresponding to these N CDF values

    Parameters
    ----------
    lc: `Lightcurve` object

    Other Parameters
    ----------------
    use_spline : bool
        Approximate the light curve with a spline to avoid binning effects

    Returns
    -------
    times : array-like
        Simulated photon arrival times

    Examples
    --------
    >>> t = [0.5, 1.5, 3.5]
    >>> c = [100] * 3
    >>> lc = Lightcurve(t, c, gti=[[0, 2], [3, 4]], dt=1.)
    >>> times = simulate_times(lc, use_spline=True)
    >>> np.all(np.diff(times) > 0)  # Output array is sorted
    True
    >>> np.all(times >= 0.)  # All times inside GTIs
    True
    >>> np.all(times <= 4.)
    True
    >>> np.any(times > 3.)
    True
    >>> np.any(times < 2.)
    True
    >>> np.any((times > 2.) & (times < 3.))  # No times outside GTIs
    False
    >>> lc.counts[0] = -3.
    >>> simulate_times(lc)  # Test with one negative value in the lc
    Traceback (most recent call last):
        ...
    ValueError: simulate_times can only work with...
    """
    return simulate_times_from_count_array(
        lc.time, lc.counts, lc.gti, lc.dt, use_spline=use_spline)


def simulate_times_from_count_array(time, counts, gti, dt, use_spline=False):
    """Simulate an event list, by using the inverse CDF method.

    + Assume that the light curve is a probability density (must be positive
      definite)

    + Calculate the CDF from the cumulative sum, and normalize it from 0 to 1

    + Extract N random probability values from 0 to 1

    + Find the CDF values corresponding to these N values

    + Find the times corresponding to these N CDF values

    Parameters
    ----------
    time: array-like
    counts: array-like
    gti: [[gti00, gti01], ..., [gtin0, gtin1]]
    dt: float

    Other Parameters
    ----------------
    use_spline : bool
        Approximate the light curve with a spline to avoid binning effects

    Returns
    -------
    times : array-like
        Simulated photon arrival times

    Examples
    --------
    >>> t = [0.5, 1.5, 2.5, 3.5, 5.5]
    >>> c = [100] * 5
    >>> gti = [[0, 4], [5, 6]]
    >>> times = simulate_times_from_count_array(t, c, gti, 1, use_spline=True)
    >>> np.all(np.diff(times) > 0)  # Output array is sorted
    True
    >>> np.all(times >= 0.)  # All times inside GTIs
    True
    >>> np.all(times <= 6.)
    True
    >>> np.any(times > 5.)
    True
    >>> np.any(times < 4.)
    True
    >>> np.any((times > 4.) & (times < 5.))  # No times outside GTIs
    False
    >>> c[0] = -3.
    >>> simulate_times_from_count_array(t, c, gti, 1)  # Test with one negative value in the lc
    Traceback (most recent call last):
        ...
    ValueError: simulate_times can only work with...
    """
    time = np.asarray(time)
    counts = np.asarray(counts)
    gti = np.asarray(gti)
    kind = "linear"
    if use_spline and time.size > 2:
        kind = "cubic"

    if np.any(counts < 0):
        raise ValueError(
            "simulate_times can only work with positive-definite light curves"
        )

    if len(gti) > 1:  # Work GTI by GTI, to avoid the spillover of events
        all_events = []
        start_bins, stop_bins = gti_border_bins(gti, time, dt=dt)
        for i, (start, stop) in enumerate(zip(start_bins, stop_bins)):
            new_events = simulate_times_from_count_array(
                time[start:stop],
                counts[start:stop],
                [gti[i]],
                dt,
                use_spline=use_spline)
            all_events.append(new_events)
        return np.concatenate(all_events)

    if len(counts) == 1:  # Corner case: a single light curve bin
        dt = dt
        t0 = time[0] - dt / 2
        t1 = time[0] + dt / 2
        N = counts[0]
        return np.sort(np.random.uniform(t0, t1, N))

    tmin = gti[0, 0]
    tmax = gti[-1, 1]
    duration = (tmax - tmin)
    phase_bins = np.copy(time)
    phase_bins -= tmin
    phase_bins /= duration
    dph = dt / duration

    counts = np.concatenate(([0], counts))
    phase_bins = np.concatenate(
        ([0], phase_bins + dph / 2))
    n_events_predict = np.random.poisson(np.sum(counts))

    cdf = np.cumsum(counts, dtype=float)
    cdf -= cdf[0]
    cdf /= cdf[-1]

    inv_cdf_func = sci.interp1d(
        cdf,
        phase_bins,
        kind=kind)

    cdf_vals = np.sort(np.random.uniform(0, 1, n_events_predict))
    times = inv_cdf_func(cdf_vals)
    times *= duration
    times += tmin

    return times
