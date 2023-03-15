import os
import csv
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import frgtools.plotting as frgplt


def load_lambda(fpath):
    """
    Loads an entire folder of output files from Lambda 1050
    specrophotometer in the basement of SME (NE-MRC)

    input:
                    fpath: filepath to directory in which uvvis data is stored

    output:
                    dictionary of uvvis data. keys are the sample names, values are:
                                    wavelength: wavelengths scanned (nm)
                                    signal: values measured, units depend on tool setting
                                    type: signal type. again, depends on tool setting
                                    filepath: path to file
    """

    def readLambdaCSV(fpath):
        wl = []
        signal = []
        with open(fpath, "r") as d:
            d_reader = csv.reader(d, delimiter=",")
            header = d_reader.__next__()  # get header line

            if "%R" in header[1]:
                signalType = "Reflectance"
            elif "%T" in header[1]:
                signalType = "Transmittance"
            elif "A" in header[1]:
                signalType = "Absorbance"
            else:
                signalType = "Unknown"

            for row in d_reader:
                wl.append(float(row[0]))
                signal.append(float(row[1]))

        return np.array(wl), np.array(signal), signalType

    readMe = ".Sample."
    ignoreMe = ".sp"
    rawFids = os.listdir(fpath)

    data = {}

    for f in rawFids:
        if readMe in f:
            if f.endswith(ignoreMe):
                continue
            path = os.path.join(fpath, f)
            name = os.path.basename(f).split(readMe)[0]
            if ".Cycle" in f:
                cycle = os.path.basename(f).split(".Cycle")[-1].split(".Raw")[0]
                name += "_{}".format(cycle)
            wl, signal, signalType = readLambdaCSV(path)

            data[name] = {
                "wavelength": wl,
                "signal": signal,
                "type": signalType,
                "filepath": path,
            }

    return data


def calc_absorbance(r, t):
    """
    given reflectance and transmission measurements, approximates absorbance.

    Note that this is the most basic approach, different samples may
    require different treatment to obtain accurate absorbance values.

    A bit more explanation in the following reference. Basically this is an approximation
    that is valid for samples where absorbance*pathlength > 2, so it should hold for
    our typical bandgap estimation task where films are strongly absorbing.

                    Look, D. C. & Leach, J. H. On the accurate determination of absorption coefficient
            from reflectance and transmittance measurements: Application to Fe-doped GaN. Journal
            of Vacuum Science & Technology B, Nanotechnology and Microelectronics: Materials,
            Processing, Measurement%, and Phenomena 34, 04J105 (2016).

    If a substrate r,t is measured, you can approximately remove the substrate
    absorbance, since absorbance is additive, in the following way:

                    A_substrate = calc_absorbance(r_substrate, t_substrate)
                    A_measured = calc_absorbance(r, t)
                    A = A_measured - A_substrate

    inputs:
                    r, t: arrays of reflectance, transmittance data
    returns:
                    A: absorbance (AU, natural log)
    """
    r = np.asarray(r)
    t = np.asarray(t)

    if r.max() > 1:
        r /= 100  # convert percents to fractional values
    if t.max() > 1:
        t /= 100

    # return -np.log10(t / ((1 - r) ** 2))  # absorbance = -log_10(I/I0)
    return -np.log10(t /(1 - r))

def beers(a, pathlength, concentration=1):
    """
    Uses Beer-Lambert's law to convert absorbance values to
    absorption coefficients (alpha)

    A = alpha * pathlength * concentration

    inputs:
                    a: absorbance
                    pathlength: optical path length (cm). often this will be film thickness, cuvette width, etc.
                    concentration: relevant for solutions. defaults to one, aka no effect. (g/cm3)

    returns:
                    alpha: absorption coefficient (cm^-1)
    """
    a = np.asarray(a)

    return a / (pathlength * concentration)


def kubelka_munk(r):
    """
    Kubelka-Munk transform to analyze diffuse reflectance data.

    Note that this approach assumes that the data is capturing DIFFUSE reflectance,
    ie from measurement of a powder sample or scattering particulates embedded in a
    non-absorbing medium.

    Also note that the calculated value is the ratio alpha/S, where alpha = absorption
    coefficient and S = the scattering coefficient, which varies with particle size and
    packing. This value is proportional to alpha, and is often used in its place, but be
    aware of the difference - if S is significantly large, or worse variable across
    the wavelengths measured, analysis using this value in place of alpha will be impacted.

    See Paper 3 shared by A. El-Denglawey at the following link:
      https://www.researchgate.net/post/Which-form-of-Kubelka-Munk-function-should-I-used-to-calculate-band-gap-of-powder-sample
    """
    r = np.asarray(r)

    if (r > 1).any():
        r_ = r / 100
    else:
        r_ = r
    return (1 - r_) ** 2 / (2 * r_)


def tauc(
    wl,
    a,
    bandgap_type,
    wlmin=None,
    wlmax=None,
    fit_width=None,
    fit_threshold=0.1,
    plot=False,
    verbose=False,
):
    """
    Performs Tauc plotting analysis to determine optical bandgap from absorbance data
    Plots data in tauc units, then performs linear fits in a moving window to find the
    best linear region. this best fit line is extrapolated to the x-axis, which corresponds
    to the bandgap.

        https://doi.org/10.1016/0025-5408(68)90023-8

    inputs

                    wl: array of wavelengths (nm)
                    a: absorption coefficient. Absorbance can also be used - while the plot will be stretched in y, a scalar factor here doesnt affect the bandgap approximation
                    thickness: sample thickness (cm)
                    bandgap_type: ['direct', 'indirect']. determines coefficient on tauc value

                    wlmin: minimum wavelength (nm) to include in plot
                    wlmax: maximum wavelenght (nm) to include in plot
                    fit_width: width of linear fit window, in units of wl, a vector indices
                    fit_threshold: window values must be above this fraction of maximum tauc value.
                                                                                    prevents fitting region before the absorption onset
                    plot: boolean flag to generate plot of fit
                    verbose: boolean flag to (True) generate detailed output or (False) just output Eg

    output (verbose = False)
                    bandgap: optical band gap (eV)

    output (verbose = True)
                    dictionary with values:
                                    bandgap: optical band gap (eV)
                                    r2: r-squared value of linear fit
                                    bandgap_min: minimum bandgap within 95% confidence interval
                                    bandgap_max: maximum bandgap within 95% confidence interval
    """
    wl = np.array(wl)
    if wlmin is None:
        wlmin = wl.min()
    if wlmax is None:
        wlmax = wl.max()
    wlmask = np.where((wl >= wlmin) & (wl <= wlmax))

    wl = wl[wlmask]
    a = np.array(a)[wlmask]

    if fit_width is None:
        fit_width = len(wl) // 20  # default to 5% of data width

    fit_pad = fit_width // 2

    if str.lower(bandgap_type) == "direct":
        n = 0.5
    elif str.lower(bandgap_type) == "indirect":
        n = 2
    else:
        raise ValueError(
            'argument "bandgap_type" must be provided as either "direct" or "indirect"'
        )

    c = 3e8  # speed of light, m/s
    h = 4.13567e-15  # planck's constant, eV
    nu = c / (wl * 1e-9)  # convert nm to hz
    ev = 1240 / wl  # convert nm to ev

    taucvalue = (a * h * nu) ** (1 / n)
    taucvalue_threshold = taucvalue.max() * fit_threshold
    best_slope = None
    best_intercept = None
    best_r2 = 0

    for idx in range(fit_pad, len(wl) - fit_pad):
        if taucvalue[idx] >= taucvalue_threshold:
            fit_window = slice(idx - fit_pad, idx + fit_pad)
            slope, intercept, rval, _, stderr = linregress(
                ev[fit_window], taucvalue[fit_window]
            )
            r2 = rval**2
            if r2 > best_r2 and slope > 0:
                best_r2 = r2
                best_slope = slope
                best_intercept = intercept

    Eg = -best_intercept / best_slope  # x intercept

    if plot:
        fig, ax = plt.subplots()
        ax.plot(ev, taucvalue, "k")
        ylim0 = ax.get_ylim()
        ax.plot(
            ev, ev * best_slope + best_intercept, color=plt.cm.tab10(3), linestyle=":"
        )
        ax.set_ylim(*ylim0)
        ax.set_xlabel("Photon Energy (eV)")
        if n == 0.5:
            ax.set_ylabel(r"$({\alpha}h{\nu})^2$")
        else:
            ax.set_ylabel(r"$({\alpha}h{\nu})^{1/2}$")
        plt.show()

    if not verbose:
        return Eg
    else:
        ### calculate 95% CI of Eg
        mx = ev.mean()
        sx2 = ((ev - mx) ** 2).sum()
        sd_intercept = stderr * np.sqrt(1.0 / len(ev) + mx * mx / sx2)
        sd_slope = stderr * np.sqrt(1.0 / sx2)

        Eg_min = -(best_intercept - 1.96 * sd_intercept) / (
            best_slope + 1.96 * sd_slope
        )
        Eg_max = -(best_intercept + 1.96 * sd_intercept) / (
            best_slope - 1.96 * sd_slope
        )

        output = {
            "bandgap": Eg,
            "r2": best_r2,
            "bandgap_min": Eg_min,
            "bandgap_max": Eg_max,
            "taucvalue": taucvalue,
            "best_slope": best_slope,
            "best_intercept": best_intercept,
            "ev": ev,
        }
        return output


def urbach(
    wl: np.array, a: np.array, wlmin: float, wlmax: float, plot: bool = False
) -> float:
    """Given wavelength and absorbance values, calculates the urbach energy

                https://en.wikipedia.org/wiki/Urbach_tail
    Args:
            wl (np.array): wavelengths (nm)
            a (np.array): absorbance values
            wlmin (float): minimum wavelength to fit from
            wlmax (float): maximum wavelength to fit to
            plot (bool, optional): If true, plots the fit used to return urbach energy. Defaults to False.

    Returns:
            float: Urbach energy (eV)
    """
    ev = 1240 / wl
    evmin = 1240 / wlmax
    evmax = 1240 / wlmin

    tail_slice = slice(np.argmin(np.abs(ev - evmin)), np.argmin(np.abs(ev - evmax)))
    x, y = ev[tail_slice], np.log(a)[tail_slice]
    slope, intercept, _, _, _ = linregress(x, y)
    Eu = 1 / slope

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].plot(ev, np.log(a))
        frgplt.vline([evmin, evmax], ax[0])
        ax[0].set_xlabel("Photon Energy (eV)")
        ax[0].set_ylabel(r"$\alpha$ (AU)")
        ax[1].plot(x, y, "ko")
        ylim0 = ax[1].get_ylim()
        ax[1].plot(x, x * slope + intercept, "r:")
        ax[1].set_ylim(ylim0)

        frgplt.cornertext(text=f"E_u: {Eu:.3f} eV", location="upper left")
        ax[1].set_xlabel("Photon Energy (eV)")
        ax[1].set_ylabel(r"$ln(\alpha)$")

        plt.tight_layout()

    return Eu
