import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
import math
import frgtools.misc as frgm
from natsort import natsorted
import time
import pandas as pd
import glob
import seaborn as sns
from scipy.interpolate import interp1d
from pvlib.ivtools.sde import fit_sandia_simple as fit_jv


def load_tracer(fpath):
    """
    Loads JV scans taken by ReRa Tracer software. fpath must target a text file exported by right-clicking
    scans in Tracer and selecting export.
    """

    def skipLines(f, numLines):
        for i in range(numLines):
            f.readline()

    def parseLine(line, leadingTabs, trailingTabs):
        contents = []
        counter = 0
        totalPer = leadingTabs + trailingTabs
        lineparts = line.split("\t")
        for part in lineparts:
            if counter == leadingTabs:
                if part != "\n":
                    contents.append(part)
            counter = counter + 1
            if counter > totalPer:
                counter = 0

        return contents

    data = {}
    with open(fpath, "r") as f:
        data["ID"] = parseLine(f.readline(), 1, 1)
        data["Device"] = parseLine(f.readline(), 1, 1)
        data["Curve"] = parseLine(f.readline(), 1, 1)
        for idx, each in enumerate(data["Curve"]):
            if "Ill" in each:
                data["Curve"][idx] = "Illuminated"
            else:
                data["Curve"][idx] = "Dark"
        data["Area"] = [float(x) for x in parseLine(f.readline(), 1, 1)]
        skipLines(f, 2)
        data["Date"] = parseLine(f.readline(), 1, 1)
        data["Time"] = parseLine(f.readline(), 1, 1)
        skipLines(f, 4)

        data["V"] = [[] for x in range(len(data["Curve"]))]
        data["I"] = [[] for x in range(len(data["Curve"]))]
        data["P"] = [[] for x in range(len(data["Curve"]))]

        for line in f:
            raw = parseLine(line, 0, 0)
            for i in range(len(data["Curve"])):
                try:
                    data["V"][i].append(float(raw[i * 3 + 0]))
                    data["I"][i].append(float(raw[i * 3 + 1]))
                    data["P"][i].append(float(raw[i * 3 + 2]))
                except:
                    pass

    return data


def load_FRG(fpath):
    """
    Loads JV data as exported by Grace's FRG MATLAB software in late 2019
    """

    def readFRGFile(fpath):
        data = {}

        with open(fpath, "r") as f:
            ReadingHeader = True  # check when header has been fully parsed
            BothDirections = False  # sweep forward + reverse

            while ReadingHeader:
                line = f.readline()
                lineparts = line.split(":")

                if "Data" in lineparts[0]:
                    ReadingHeader = False
                    f.readline()
                    f.readline()
                else:
                    try:
                        data[lineparts[0]] = float(lineparts[1])
                    except:
                        data[lineparts[0]] = lineparts[1][1:].replace("\n", "")

            vforward = []
            iforward = []
            timeforward = []
            if data["sweepDir"] == "Forward + Reverse":
                BothDirections = True
                vreverse = []
                ireverse = []
                timereverse = []

            for line in f:
                lineparts = f.readline().split("\t")
                if len(lineparts) == 1:
                    break
                vforward.append(lineparts[0])
                iforward.append(lineparts[1])
                timeforward.append(lineparts[2])
                if BothDirections:
                    vreverse.append(lineparts[0])
                    ireverse.append(lineparts[1])
                    timereverse.append(lineparts[2])

            data["V"] = np.array(vforward).astype(float)
            data["I"] = np.array(iforward).astype(float)
            data["J"] = data["I"] / data["area_cm2"]
            data["delay"] = np.array(timeforward).astype(float)

            if BothDirections:
                data2 = data.copy()
                data2["sampleName"] = data["sampleName"] + "_Reverse"
                data["sampleName"] = data["sampleName"] + "_Forward"
                data2["V"] = np.array(vreverse).astype(float)
                data2["I"] = np.array(ireverse).astype(float)
                data2["J"] = data2["I"] / data2["area_cm2"]
                data2["delay"] = np.array(timereverse).astype(float)
                output = [data, data2]
            else:
                output = data

        return output

    fids = [os.path.join(fpath, x) for x in os.listdir(fpath)]

    alldata = {}
    for f in fids:
        output = readFRGFile(f)
        if type(output) == list:
            for each in output:
                alldata[each["sampleName"]] = each
        else:
            alldata[output["sampleName"]] = output
    return alldata


def fit_dark(v, i, area, plot=False, init_guess={}, bounds={}, maxfev=5000):
    """
    Takes inputs of voltage (V), measured current (A), and cell area (cm2)

    Fitting by 2-diode model provides parameters:
    Diode saturation currents: Jo1, (Jo2 if 2-diode model) (A/cm2)
    Series resistance: Rs (ohms cm2)
    Shunt resistance: Rsh (ohms)
    """

    j = [i_ / area for i_ in i]
    v = np.asarray(v)
    j = np.asarray(j)

    n = [idx for idx, v_ in enumerate(v) if v_ >= 0]
    vfit = v.copy()
    jfit = j.copy()
    # vfit = [v[n_] for n_ in n]
    # jfit = [np.log(i[n_]/area) for n_ in n]
    # jfit = [i[n_]/area for n_ in n]
    vfit = np.asarray(vfit)
    jfit = np.asarray(jfit)
    x = np.vstack((vfit, jfit))

    xplot = np.vstack((v, j))

    init_guess_ = [1e-12, 1e-12, 2, 1e3]  # jo1, jo2, rs, rsh
    for key, val in init_guess.items():
        for idx, choices in enumerate(
            [["jo1", "j01"], ["jo2", "j02"], ["rs", "rseries"], ["rsh", "rshunt"]]
        ):
            if str.lower(key) in choices:
                init_guess_[idx] = val
                break

    bounds_ = [[0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]]
    for key, vals in bounds.items():
        for idx, choices in enumerate(
            [["jo1", "j01"], ["jo2", "j02"], ["rs", "rseries"], ["rsh", "rshunt"]]
        ):
            if str.lower(key) in choices:
                bounds_[0][idx] = vals[0]
                bounds_[1][idx] = vals[1]
                break

    best_vals, covar = curve_fit(
        _Dark2Diode,
        x,
        x[1, :],
        p0=init_guess_,
        bounds=bounds_,
        maxfev=maxfev,
        method="dogbox",
    )

    results = {
        "jo1": best_vals[0],
        "jo2": best_vals[1],
        "rs": best_vals[2],
        "rsh": best_vals[3],
        "covar": covar,
        "jfit": _Dark2Diode(
            xplot, best_vals[0], best_vals[1], best_vals[2], best_vals[3], exp=False
        ),
    }

    if plot:
        fig, ax = plt.subplots()
        ax.plot(v, np.log(j * 1000), label="Measured")
        ax.plot(v, np.log(results["jfit"] * 1000), linestyle="--", label="Fit")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("log(Current Density) (mA/cm2)")
        plt.show()

    return results


def fit_light(
    v, i, area, diodes=2, plot=False, init_guess={}, bounds={}, maxfev=5000, type=None
):
    """
    Takes inputs of voltage (V), measured current (A), and cell area (cm2)

    Fits an illuminated JV curve to find at least the basic JV parameters:
    Open-circuit voltage: Voc (V)
    Short-circuit current: Jsc (mA/cm2)
    Max power point voltage: Vmpp (V)

    Fitting by 2-diode (default) or 1-diode model as specified by diodes argument provides additional parameters:
    Diode saturation currents: Jo1, (Jo2 if 2-diode model) (A/cm2)
    Series resistance: Rs (ohms cm2)
    Shunt resistance: Rsh (ohms)
    Photogenerated current: Jl (A/cm2)
    """

    j = [i_ / area for i_ in i]  # convert A to mA

    if max(j) > 0.05:
        print(
            "Current seems too high (max = {0} mA/cm2). Please double check that your area (cm2) and measured current (A) are correct.".format(
                max(j * 1000)
            )
        )

    v = np.asarray(v)
    j = np.asarray(j)

    j = j[v >= 0]
    v = v[v >= 0]
    v = v[j >= 0]
    j = j[j >= 0]

    p = np.multiply(v, j)
    x = np.vstack((v, j))

    jsc = j[np.argmin(np.abs(v))]

    if diodes == 2:
        init_guess_ = [1e-12, 1e-12, 2, 1e3, jsc]  # jo1, jo2, rs, rsh, jl
        for key, val in init_guess.items():
            for idx, choices in enumerate(
                [
                    ["jo1", "j01"],
                    ["jo2", "j02"],
                    ["rs", "rseries"],
                    ["rsh", "rshunt"],
                    ["jl", "jill", "jilluminated"],
                ]
            ):
                if str.lower(key) in choices:
                    init_guess_[idx] = val
                    break

        bounds_ = [[0, 0, 0, 0, jsc * 0.9], [np.inf, np.inf, np.inf, np.inf, jsc * 1.1]]
        for key, vals in bounds.items():
            for idx, choices in enumerate(
                [
                    ["jo1", "j01"],
                    ["jo2", "j02"],
                    ["rs", "rseries"],
                    ["rsh", "rshunt"],
                    ["jl", "jill", "jilluminated"],
                ]
            ):
                if str.lower(key) in choices:
                    bounds_[0][idx] = vals[0]
                    bounds_[1][idx] = vals[1]
                    break

        best_vals, covar = curve_fit(
            _Light2Diode,
            x,
            x[1, :],
            p0=init_guess_,
            maxfev=maxfev,
            bounds=bounds_,
            method="trf",
        )

        results = {
            "jo1": best_vals[0],
            "jo2": best_vals[1],
            "rs": best_vals[2],
            "rsh": best_vals[3],
            "jl": best_vals[4],
            "covar": covar,
        }
        results["jfit"] = _Light2Diode(
            x,
            results["jo1"],
            results["jo2"],
            results["rs"],
            results["rsh"],
            results["jl"],
        )

    elif diodes == 1:
        init_guess_ = [1e-12, 2, 1e3, jsc]  # jo1, rs, rsh, jl
        for key, val in init_guess.items():
            for idx, choices in enumerate(
                [
                    ["jo1", "j01"],
                    ["rs", "rseries"],
                    ["rsh", "rshunt"],
                    ["jl", "jill", "jilluminated"],
                ]
            ):
                if str.lower(key) in choices:
                    init_guess_[idx] = val
                    break

        bounds_ = [[0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]]
        for key, vals in bounds.items():
            for idx, choices in enumerate(
                [
                    ["jo1", "j01"],
                    ["rs", "rseries"],
                    ["rsh", "rshunt"],
                    ["jl", "jill", "jilluminated"],
                ]
            ):
                if str.lower(key) in choices:
                    bounds_[0][idx] = vals[0]
                    bounds_[1][idx] = vals[1]
                    break

        print(init_guess_)
        print(bounds_)
        best_vals, covar = curve_fit(
            _Light1Diode, x, x[1, :], p0=init_guess_, maxfev=maxfev, bounds=bounds_
        )

        results = {
            "jo1": best_vals[0],
            "rs": best_vals[1],
            "rsh": best_vals[2],
            "jl": best_vals[3],
            "covar": covar,
        }
        results["jfit"] = _Light1Diode(
            x, results["jo1"], results["rs"], results["rsh"], results["jl"]
        )
    else:
        print(
            "Error: Invalid number of diodes requested for fitting. Diode must equal 1 or 2. User provided {0}. Diode equation not fit.".format(
                diodes
            )
        )
        results = {}

    if plot and len(results) > 0:
        fig, ax = plt.subplots()
        ax.plot(v, j * 1000, label="Measured")
        xlim0 = ax.get_xlim()
        ylim0 = ax.get_ylim()
        ax.plot(v, results["jfit"] * 1000, linestyle="--", label="Fit")
        ax.set_xlim(xlim0)
        ax.set_ylim(ylim0)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current Density (mA/cm2)")
        plt.show()

    results["voc"] = v[np.argmin(np.abs(j))]
    results["jsc"] = j[np.argmin(np.abs(v))]
    results["vmpp"] = v[np.argmax(p)]
    results["pce"] = p.max() * 10
    results["ff"] = p.max() / (results["voc"] * results["jsc"])

    return results


### Diode Models


def _Dark2Diode(x, jo1, jo2, rs, rsh, exp=False):
    v = x[0]
    if exp:
        j_meas = np.exp(x[1])
    else:
        j_meas = x[1]

    # constants
    q = 1.60217662e-19  # coulombs
    k = 1.380649e-23  # J/K
    T = 298.15  # assume room temperature

    # calculation
    d1 = jo1 * np.exp((q * (v - (j_meas * rs))) / (k * T))
    d2 = jo2 * np.exp((q * (v - (j_meas * rs))) / (2 * k * T))
    j = d1 + d2 + (v - (j_meas * rs)) / rsh

    if exp:
        return np.log(j)
    else:
        return j


def _Light2Diode(x, jo1, jo2, rs, rsh, jl):
    v = x[0]
    j_meas = x[1]

    # constants
    q = 1.60217662e-19  # coulombs
    k = 1.380649e-23  # J/K
    T = 298.15  # assume room temperature

    # calculation
    d1 = jo1 * np.exp((q * (v + (j_meas * rs))) / (k * T))
    d2 = jo2 * np.exp((q * (v + (j_meas * rs))) / (2 * k * T))
    j = jl - d1 - d2 - (v + j_meas * rs) / rsh

    return j


def _Light1Diode(x, jo1, rs, rsh, jl):
    v = x[0]
    j_meas = x[1]

    # constants
    q = 1.60217662e-19  # coulombs
    k = 1.380649e-23  # J/K
    T = 298.15  # assume room temperature

    # calculation
    d1 = jo1 * np.exp((q * (v + (j_meas * rs))) / (k * T))
    j = jl - d1 - (v + j_meas * rs) / rsh

    return j


def interpolate_func(v, i, pct_change_allowed=0.1):
    """Get interpolation function.

    Args:
        pct_change_allowed (float): smoothing parameter - only allow points with less than this amount of percent
            change between adjacent points
        spline_kwargs (dict): keywords for spline (see scipy.interpolate.splrep)
    """
    pct_change = np.abs(np.diff(i)) / i[:-1]
    pct_change = np.insert(pct_change, 0, 0)
    v = v[pct_change < pct_change_allowed]
    i = i[pct_change < pct_change_allowed]
    tck = interpolate.splrep(v, i, s=0)
    return tck


def smooth(v, i, nmpts=3000):
    # sorting IV curve by voltage (needed for interpolation)
    x = v  # not smooth
    y = i  # not smooth

    xy = np.column_stack((x, y))
    xy = xy[xy[:, 0].argsort()]
    x = xy[:, 0]
    y = xy[:, 1]

    v_smooth = np.linspace(x[0], x[-1], nmpts)

    tck = interpolate_func(x, y, pct_change_allowed=2)  # , spline_kwargs={'s': 0.025})

    i_smooth = interpolate.splev(v_smooth, tck, der=0)

    return v_smooth, i_smooth


def smooth_light_fit(v, i, area=0.07):
    # 	Deniz Cakan 20220822
    # 	Extracts the solar cell device performance parameters from the IV curve

    v, i = smooth(v, i)
    # maximum power point
    pmax = np.max((v * i))
    mpp_idx = np.argmax((v * i))
    vpmax = v[mpp_idx]
    ipmax = i[mpp_idx]
    j = i / area / 0.001

    voc = np.interp(0, i, v)
    isc = np.interp(0, v, -i)
    jsc = np.interp(0, v, -j)

    p = -np.multiply(v, j)
    pce = np.max(p)
    ff = (pce) / (jsc * voc) * 100

    pce = voc * jsc * ff / 100
    params = {
        "p": p,
        "pce": pce,
        "vmpp": vpmax,
        "voc": voc,
        "isc": isc,
        "jsc": jsc,
        "ff": ff,
    }
    return params


def calculate_jv_parameters(v=np.array, j=np.array):
    """
    From Sean P. Dunfield's code
    Takes in voltage, current, and power vectors, calculates scalars and returns a dictionary of scalars

    Args:
            all_v (list[np.ndarray]): list of voltage vectors
            all_j (list[np.ndarray]): list of current vectors
            all_p (list[np.ndarray]): list of power vectors
            direction (str): direction -- either FWD or REV

    Returns:
            dict: dictionary of parameter values over time
    """
    v = v
    derivative_v_step = 0.1
    p = v * j

    # Try to calculate scalars
    try:

        # Calculate Jsc and Rsh using J(v=0) to J(v = v_dir)
        wherevis0 = np.nanargmin(np.abs(v))
        wherevis0_1 = np.nanargmin(np.abs(v - derivative_v_step))
        j1 = j[wherevis0]
        j2 = j[wherevis0_1]
        v1 = v[wherevis0]
        v2 = v[wherevis0_1]
        m = (j2 - j1) / (v2 - v1)
        b = j1 - m * v1
        if m != 0:
            rsh = float(abs(1 / m))
            jsc = float(b)
        else:
            rsh = np.inf
            jsc = float(b)

        # Calculate Voc and Rs from J(J=0) to derivative_v_step V before
        v_iter = max(math.ceil(derivative_v_step / (v[2] - v[1])), 1)
        wherejis0 = np.nanargmin(np.abs(j))
        wherejis0_1 = wherejis0 - int(v_iter)
        j1 = j[wherejis0]
        j2 = j[wherejis0_1]
        v1 = v[wherejis0]
        v2 = v[wherejis0_1]
        m = (j2 - j1) / (v2 - v1)
        b = j1 - m * v1
        rs = float(abs(1 / m))
        voc = float(-b / m)

        # Calculate Pmp, Vmp, Jmp
        pmp = np.nanmax(p)
        pmaxloc = np.nanargmax(p)
        vmp = v[pmaxloc]
        jmp = j[pmaxloc]

        # Calculate Rch using Vmpp-(derivative_v_step/2) V to vmpp+(derivative_v_step/2) V
        j1 = j[pmaxloc - math.floor(v_iter / 2)]
        j2 = j[pmaxloc + math.floor(v_iter / 2)]
        v1 = v[pmaxloc - math.floor(v_iter / 2)]
        v2 = v[pmaxloc + math.floor(v_iter / 2)]
        if j1 != j2 and v1 != v2:
            m = (j2 - j1) / ((v2 - v1))
            rch = float(abs(1 / m))
        else:
            rch = np.nan

        # Calculate FF and PCE if its not going to throw an error, else flag FF as NaN
        if pmp > 0 and voc > 0 and jsc > 0:
            ff = 100 * pmp / (voc * jsc)
            pce = ff * jsc * voc / 100
        else:
            ff = np.nan

    # If we run into any issues, just make values for time NaN
    except:
        # pass
        jsc = np.nan
        rsh = np.nan
        voc = np.nan
        rs = np.nan
        vmp = np.nan
        jmp = np.nan
        pmp = np.nan
        rch = np.nan
        ff = np.nan
        pce = np.nan

    # Create dictionary to hold data
    returndict = {
        "pce": pce,
        "jsc": jsc,
        "voc": voc,
        "ff": ff,
        "rsh": rsh * 1e2,
        "rs": rs * 1e2,
        "rch": rch,
        "jmp": jmp,
        "vmp": vmp,
        "pmp": pmp,
    }

    return returndict


def calc_i_factor(data):

    # AREA = 0.07 #cm^2, mask area to turn values areal

    for n in range(len(data)):
        try:
            i_factor_raw_v = data["voltage_measured"][n]
            i_factor_raw_j = -data["current_measured"][n] / 0.07 / 0.001

            v_interp = interp1d(i_factor_raw_v, i_factor_raw_j)
            j_interp = interp1d(i_factor_raw_j, i_factor_raw_v)
            i_factor_jsc = v_interp(0)
            i_factor_voc = j_interp(0)
            i_factor_v = i_factor_raw_v[i_factor_raw_v > 0]
            i_factor_v = i_factor_v[i_factor_v < i_factor_voc]

            i_factor_v = np.append(i_factor_v, i_factor_voc)
            i_factor_v = np.append(i_factor_v, 0)
            i_factor_v = np.sort(i_factor_v)

            i_factor_j = i_factor_raw_j[i_factor_raw_v > 0]
            i_factor_j = i_factor_j[i_factor_j > 0]
            if i_factor_j[0] < i_factor_j[-1]:
                i_factor_j = i_factor_j[::-1]

            i_factor_j = np.insert(i_factor_j, len(i_factor_j), 0)
            i_factor_j = np.insert(i_factor_j, 0, i_factor_jsc)

            j_ill, j_0, r_s, r_sh, vth_n = fit_jv(i_factor_v, i_factor_j)

            i_factor = vth_n / 0.026

            data["i_factor"][n] = i_factor
        except:
            data["i_factor"][n] = np.nan
    return data


def jv_metrics_pkl(
    rootdir=str,
    batch=str,
    area=0.07,
    pce_cutoff=3,
    ff_cutoff=None,
    voc_cutoff=None,
    export_raw=False,
):
    # sample name must be recorded in the following format: 's{sample_number}_{rescan_number}'
    rootdir = rootdir
    dark = os.path.join(rootdir, "dark")
    light = os.path.join(rootdir, "light")
    reference = os.path.join(rootdir, "ref")

    # fids = []
    # for f in frgm.listdir(light, display=False):
    #     if "control" in f and "_rgb" not in f:
    #         continue
    #     fids.append(f)
    fids = natsorted(glob.glob(light + "/*.csv"))
    # print(len(fids))

    data = {}
    internal = {}
    internal["name"] = []
    internal["direction"] = []
    internal["repeat"] = []
    internal["pixel"] = []

    data["p"] = []
    data["current_measured"] = []

    data["voltage_measured"] = []
    data["voltage_setpoint"] = []

    area = 0.07
    sim_correction_factor = 1.00

    for n in range(len(fids)):
        internal["name"].append(os.path.basename(fids[n])[:-4].split("_")[0])

        if os.path.basename(fids[n])[:-4].split("_")[1] == "":
            pixel = "0"
            internal["pixel"].append(pixel)
        if os.path.basename(fids[n])[:-4].split("_")[1] != "":
            pixel = os.path.basename(fids[n])[:-4].split("_")[1]
            internal["pixel"].append(pixel)

        if os.path.basename(fids[n])[:-4].split("_")[2] == "":
            repeat = "0"
            internal["repeat"].append(repeat)
        if os.path.basename(fids[n])[:-4].split("_")[2] != "":
            repeat = os.path.basename(fids[n])[:-4].split("_")[2]
            internal["repeat"].append(repeat)

        internal["direction"].append(os.path.basename(fids[n])[:-4].split("_")[3])

        df_single = pd.read_csv(fids[n], header=0)
        voltage_setpoint = np.asarray(df_single["Voltage (V)"])
        current_measured = np.asarray(df_single["Current (A)"]) * sim_correction_factor
        measured_voltage = np.asarray(df_single["Measured Voltage (V)"])

        data["voltage_measured"].append(measured_voltage)
        data["current_measured"].append(current_measured)
        #     data['current_density'].append(current_density)
        data["voltage_setpoint"].append(voltage_setpoint)
        data["p"].append(-measured_voltage * current_measured)

    df = pd.DataFrame(data)

    df["voc"] = np.nan
    df["jsc"] = np.nan
    df["pce"] = np.nan
    df["ff"] = np.nan
    df["rsh"] = np.nan
    df["rs"] = np.nan
    df["rch"] = np.nan
    df["i_factor"] = np.nan

    df["current_measured_flipped"] = np.array

    df["current_measured_flipped"] = -1 * df["current_measured"]

    df.insert(0, "name", internal["name"])
    df.insert(1, "pixel", internal["pixel"])
    df.insert(2, "repeat", internal["repeat"])
    df.insert(3, "direction", internal["direction"])

    df["area"] = np.nan

    # rch doesnt work with reverse curves currently
    df["voltage_measured_temp"] = df["voltage_measured"]
    df["current_measured_temp"] = df["current_measured"]
    for i in range(len(df)):
        if df["direction"][i] == "rev":
            df["voltage_measured"][i] = df["voltage_measured"][i][::-1]
            df["current_measured"][i] = df["current_measured"][i][::-1]

    for n in range(len(df)):

        df["area"][n] = area
        try:
            df["voc"][n] = np.round(
                calculate_jv_parameters(
                    df["voltage_measured"][n], -df["current_measured"][n] / area * 1e3
                )["voc"]
                * 1000,
                1,
            )
            df["jsc"][n] = np.round(
                calculate_jv_parameters(
                    df["voltage_measured"][n], -df["current_measured"][n] / area * 1e3
                )["jsc"],
                2,
            )
            df["pce"][n] = np.round(
                calculate_jv_parameters(
                    df["voltage_measured"][n], -df["current_measured"][n] / area * 1e3
                )["pce"],
                2,
            )
            df["ff"][n] = np.round(
                calculate_jv_parameters(
                    df["voltage_measured"][n], -df["current_measured"][n] / area * 1e3
                )["ff"],
                2,
            )
            df["rsh"][n] = np.round(
                calculate_jv_parameters(
                    df["voltage_measured"][n],
                    -df["current_measured"][n] / area * 1e3,
                )["rsh"],
                3,
            )
            df["rs"][n] = np.round(
                calculate_jv_parameters(
                    df["voltage_measured"][n], -df["current_measured"][n] / area * 1e3
                )["rs"],
                4,
            )
            df["rch"][n] = np.round(
                calculate_jv_parameters(
                    df["voltage_measured"][n],
                    -df["current_measured"][n],
                )["rch"],
                4,
            )

        except:
            pass
    df["voltage_measured"] = df["voltage_measured_temp"]
    df["current_measured"] = df["current_measured_temp"]

    df = df.sort_values(by=["pce"], ascending=False)
    # df = df.dropna()
    if pce_cutoff != None:
        df = df[~(df["pce"] <= pce_cutoff)]
    if ff_cutoff != None:
        df = df[~(df["ff"] <= ff_cutoff)]
    if voc_cutoff != None:
        df = df[~(df["voc"] <= voc_cutoff)]

    # df = df[~(df['pce'] <= 3)]
    df = df[~(df["ff"] >= 95)]
    # df = df[~(df['ff'] <= 70)]

    # df = df[~(df['voc'] >= 1.3*1e3)]
    # df = df[~(df['voc'] <= 0.7*1e3)]

    df = df.reset_index(drop=True)

    df["PASCAL_ID"] = ""
    df["PASCAL_ID"] = df["name"].astype(int)

    df = calc_i_factor(df)

    df_filter = df

    Filter_1 = ""

    df_filter1 = df_filter[df_filter.repeat.str.contains(Filter_1)]
    df_filter3 = df_filter1.reset_index(drop=True)

    df_export = df_filter3[
        [
            "PASCAL_ID",
            "pixel",
            "direction",
            "pce",
            "ff",
            "voc",
            "jsc",
            "rsh",
            "rs",
            "rch",
            "i_factor",
        ]
    ]

    TodaysDate = time.strftime("%Y%m%d")
    fp = "JV_pkl"
    if not os.path.exists(fp):
        os.mkdir(fp)
    os.chdir(fp)
    df_export.to_pickle(f"{TodaysDate}_{batch}_JV.pkl")
    os.chdir("..")

    if export_raw == True:
        df_export_raw = df_filter3[
            [
                "PASCAL_ID",
                "pixel",
                "direction",
                "voltage_measured",
                "current_measured",
                "pce",
                "ff",
                "voc",
                "jsc",
                "rsh",
                "rs",
                "rch",
                "i_factor",
            ]
        ]

        TodaysDate = time.strftime("%Y%m%d")
        fp = "JV_pkl"
        if not os.path.exists(fp):
            os.mkdir(fp)
        os.chdir(fp)
        df_export_raw.to_pickle(f"{TodaysDate}_{batch}_RAW_JV.pkl")
        os.chdir("..")
    return df_filter3


def boxplot_jv(
    data,
    xvar=None,
    pce_lim=None,
    ff_lim=None,
    voc_lim=None,
    jsc_lim=None,
    rsh_lim=None,
    rs_lim=None,
    rch_lim=None,
    i_factor_lim=None,
):
    """
    takes a dataframe and plots a boxplot of the JV parameters
    one of the columns must be the direction of the scan
    so 2 rows per device (fwd and rev)


    """
    data = data
    data["all"] = "all"
    if xvar == None:
        xvar = "all"

    horiz = 4
    vert = 2
    embiggen = 3
    q = 0

    y_var_list = ["pce", "rch", "ff", "rsh", "voc", "rs", "jsc", "i_factor"]
    fig, ax = plt.subplots(
        vert,
        horiz,
        figsize=(horiz * embiggen, vert * embiggen),
        constrained_layout=True,
    )

    for k in range(horiz):
        for n in range(vert):
            y_var = y_var_list[q]

            ax[n, k] = sns.boxplot(
                x=xvar,
                y=y_var,
                data=data,
                hue=data["direction"],
                showfliers=False,
                ax=ax[n, k],
            )
            ax[n, k].get_legend().remove()

            ax[n, k] = sns.stripplot(
                x=xvar,
                y=y_var,
                data=data,
                hue=data["direction"],
                size=3,
                linewidth=0.2,
                ax=ax[n, k],
                dodge=True,
            )
            ax[n, k].get_legend().remove()

            if xvar == "all":
                ax[n, k].set_xlabel("")

            if y_var == "pce":
                y_axis_label = "Power Conversion Effiency %"
                ax[n, k].set_ylabel(y_axis_label)
                if pce_lim:
                    ax[n, k].set(ylim=(pce_lim[0], pce_lim[1]))
                    
            if y_var == "jsc":
                y_axis_label = "J$_{SC}$ mA/cm$^2$"
                ax[n, k].set_ylabel(y_axis_label)
                if jsc_lim:
                    ax[n, k].set(ylim=(jsc_lim[0], jsc_lim[1]))

            if y_var == "voc":
                y_axis_label = "V$_{OC}$ mV"
                ax[n, k].set_ylabel(y_axis_label)
                if voc_lim:
                    ax[n, k].set(ylim=(voc_lim[0], voc_lim[1]))

            if y_var == "ff":
                y_axis_label = "Fill Factor %"
                ax[n, k].set_ylabel(y_axis_label)
                if ff_lim:
                    ax[n, k].set(ylim=(ff_lim[0], ff_lim[1]))

            if y_var == "rsh":
                y_axis_label = "Shunt Resistance Ωcm$^2$"
                ax[n, k].set_ylabel(y_axis_label)
                if rsh_lim:
                    ax[n, k].set(ylim=(rsh_lim[0], rsh_lim[1]))

            if y_var == "rs":
                y_axis_label = "Series Resistance Ωcm$^2$"
                ax[n, k].set_ylabel(y_axis_label)
                if rs_lim:
                    ax[n, k].set(ylim=(rs_lim[0], rs_lim[1]))

            if y_var == "rch":
                y_axis_label = "Characteristic Resistance Ωcm$^2$"
                ax[n, k].set_ylabel(y_axis_label)
                if rch_lim:
                    ax[n, k].set(ylim=(rch_lim[0], rch_lim[1]))

            if y_var == "i_factor":
                y_axis_label = "Ideality Factor"

                ax[n, k].set_ylabel(y_axis_label)
                if i_factor_lim:
                    ax[n, k].set(ylim=(i_factor_lim[0], i_factor_lim[1]))

            q += 1
    plt.show()
