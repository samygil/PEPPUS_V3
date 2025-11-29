#!/usr/bin/env python

########################################################################
# PETRUS/SRC/Corrections.py:
# This is the Corrections Module of PEPPUS tool
#
#  Project:        PEPPUS
#  File:           Corrections.py
#
#   Author: GNSS Academy
#   Copyright GNSS Academy
#
# -----------------------------------------------------------------
# Date       | Author             | Action
# -----------------------------------------------------------------
#
########################################################################

# Import External and Internal functions and Libraries
#----------------------------------------------------------------------
import sys, os
# Add path to find all modules
Common = os.path.dirname(os.path.dirname(
    os.path.abspath(sys.argv[0]))) + '/COMMON'
sys.path.insert(0, Common)
from collections import OrderedDict, defaultdict
from COMMON import GnssConstants as Const
from COMMON.Misc import findSun, crossProd
# from COMMON.Tropo import computeTropoMpp, computeZtd, computeSigmaTropo
from COMMON.Tropo import computeGeoidHeight
from InputOutput import RcvrIdx, ObsIdx, SatPosIdx, SatClkIdx, SatApoIdx
import numpy as np
from bisect import bisect_left, bisect_right
from math import exp, sqrt


def runCorrectMeas(Conf, Rcvr, ObsInfo, PreproObsInfo, 
SatPosInfo, SatClkInfo, SatApoInfo, SatComPos_1, Sod_1, EpochsPerSat):

    # Purpose: correct GNSS preprocessed measurements and compute the first
    #          pseudo range residuals

    #          More in detail, this function handles the following:
    #          tasks:

    #             *  Compute the Satellite Antenna Phase Center position at the transmission time and corrected from the Sagnac
    #                effect interpolating the SP3 file positions
    #             *  Compute the Satellite Clock Bias interpolating the biases coming from the RINEX CLK file and
    #                applying the Relativistic Correction (DTR)
    #             *  Estimate the Slant Troposphere delay (STD) using MOPS model (ZTD) and its mapping function. 
    #             *  Correct the Pre-processed measurements from Geometrical Range, Satellite clock and Troposphere. 
    #             *  Build the Corrected Measurements and Measurement Residuals
    #             *  Build the Sigma UERE


    # Parameters
    # ==========
    # Conf: dict
    #         Configuration dictionary
    # Rcvr: list
    #         Receiver information: position, masking angle...
    # ObsInfo: list
    #         OBS info for current epoch
    #         ObsInfo[1][1] is the second field of the 
    #         second satellite
    # PreproObsInfo: dict
    #         Preprocessed observations for current epoch per sat
    #         PreproObsInfo["G01"]["C1"]
    # SatPosInfo: dict
    #         containing the SP3 file info
    # SatClkInfo: dict
    #         containing the RINEX CLK file info
    # SatApoInfo: dict
    #         containing the ANTEX file info
    # SatComPos_1: dict
    #         containing the previous satellite positions
    # Sod_1: dict
    #         containing the time stamp of previous satellite positions

    # Returns
    # =======
    # CorrInfo: dict
    #         Corrected measurements for current epoch per sat
    #         CorrInfo["G01"]["CorrectedPsr"]

    # --- Reindex SatClkInfo, SatPosInfo, SatApoInfo if needed ---
    def reindex_info(info_dict, prepro_obs):
        expanded = {}
        for sat in prepro_obs.keys():  # sat like "G27"
            sys = sat[0]   # "G"
            prn = int(sat[1:])  # "27" → 27
            if sys in info_dict and prn in info_dict[sys]:
                expanded[sat] = info_dict[sys][prn]
            else:
                print(f"Warning: {sat} not found in CLK info_dict")
        return expanded
    
    def convert_satpos_format(SatPosInfo_raw):
        SatPosInfo = {}
        for sat, odict in SatPosInfo_raw.items():
            epochs = list(odict.keys())
            positions = [list(pos) for pos in odict.values()]
            SatPosInfo[sat] = {
                "Epochs": epochs,
                "Positions": positions
            }
        return SatPosInfo
    

    # Apply reindexing
    SatClkInfo = reindex_info(SatClkInfo, PreproObsInfo)

    SatPosInfo_raw = reindex_info(SatPosInfo, PreproObsInfo)
    SatPosInfo = convert_satpos_format(SatPosInfo_raw)

    SatApoInfo = reindex_info(SatApoInfo, PreproObsInfo)

    # Set default interpolation orders if not defined
    Conf.setdefault("ClkInterpOrder", 1)   # Default: linear interpolation for clock
    Conf.setdefault("PosInterpOrder", 10)  # Default: 10-point Lagrange for SP3 positions

    # Initialize output
    CorrInfo = OrderedDict({})

    # Get SoD
    Sod = int(float(ObsInfo[0][ObsIdx["SOD"]]))

    # Get DoY
    Doy = int(float(ObsInfo[0][ObsIdx["DOY"]]))

    # Get Year
    Year = int(float(ObsInfo[0][ObsIdx["YEAR"]]))

    # Find Sun position
    SunPos = findSun(Year, Doy, Sod)

    # Get receiver reference position IN METERS
    RcvrRefPosXyz = np.array(\
                            (\
                                Rcvr[RcvrIdx["XYZ"]][0],
                                Rcvr[RcvrIdx["XYZ"]][1],
                                Rcvr[RcvrIdx["XYZ"]][2],
                            )
                        )

    # Loop over satellites
    for SatLabel, SatPrepro in PreproObsInfo.items():
        
        # Initialize output info
        SatCorrInfo = {
            "Sod": 0.0,             # Second of day
            "Doy": 0,               # Day of year
            "Elevation": 0.0,       # Elevation
            "Azimuth": 0.0,         # Azimuth
            "Flag": 1,              # 0: Not Used 1: Used for PA 2: Used for NPA
            "SatX": 0.0,            # X-Component of the Satellite CoP Position 
                                    # at transmission time and corrected from Sagnac
            "SatY": 0.0,            # Y-Component of the Satellite CoP Position  
                                    # at transmission time and corrected from Sagnac
            "SatZ": 0.0,            # Z-Component of the Satellite CoP Position  
                                    # at transmission time and corrected from Sagnac
            "ApoX": 0.0,            # X-Component of the Satellite APO in ECEF
            "ApoY": 0.0,            # Y-Component of the Satellite APO in ECEF
            "ApoZ": 0.0,            # Z-Component of the Satellite APO in ECEF
            "SatClk": 0.0,          # Satellite Clock Bias
            "FlightTime": 0.0,      # Signal Flight Time
            "Dtr": 0.0,             # Relativistic correction
            "Std": 0.0,             # Slant Tropospheric Delay
            "CorrCode": 0.0,        # Code corrected from delays
            "CorrPhase": 0.0,       # Phase corrected from delays
            "GeomRange": 0.0,       # Geometrical Range (distance between Satellite 
                                    # Position and Receiver Reference Position)
            "CodeResidual": 0.0,    # Code Residual
            "PhaseResidual": 0.0,   # Phase Residual
            "RcvrClk": 0.0,         # Receiver Clock estimation
            "SigmaTropo": 0.0,      # Sigma of the Tropo Delay Error
            "SigmaAirborne": 0.0,   # Sigma Airborne Error
            "SigmaNoiseDiv": 0.0,   # Sigma of the receiver noise + divergence
            "SigmaMultipath": 0.0,  # Sigma of the receiver multipath
            "SigmaUere": 0.0,       # Sigma User Equivalent Range Error (Sigma of 
                                    # the total residual error associated to the 
                                    # satellite)
            "TropoMpp": 0.0,        # Tropospheric mapping function

        } # End of SatCorrInfo

        # Prepare outputs ----------------------------------------------
        SatCorrInfo["Sod"] = Sod # Get SoD
        SatCorrInfo["Doy"] = Doy # Get DoY
        SatCorrInfo["Elevation"] = SatPrepro["Elevation"] # Get Elevation
        SatCorrInfo["Azimuth"] = SatPrepro["Azimuth"] # Get Azimuth
        SatCorrInfo["Flag"] = 0  # default for not used
        # Get list of SODs for this satellite
        sods = sorted(EpochsPerSat.get(SatLabel, []))

        # Find previous and next epochs
        EpochBefore = max([s for s in sods if s < Sod], default=None)
        EpochAfter  = min([s for s in sods if s > Sod], default=None)

        #---------------------------------------------------------------

        # Only attempt corrections if status is OK
        if SatPrepro["Status"] == 1:
            can_process = True

            # 1) INTERPOLATING SAT CLOCK BIAS ---------------------------------
            try:
                clk_series = SatClkInfo[SatLabel]  # OrderedDict {SOD: bias}
                SatClkBias = interpolate_from_dict(clk_series, Sod)
            except Exception as e:
                can_process = False

            # 2) TRANSMISSION TIME, COM POSITION AND SAGNAC CORRECTION --------
            if can_process:
                try:
                    DeltaT = SatPrepro["C1"] / Const.SPEED_OF_LIGHT
                    TransmissionTime = Sod - DeltaT - (SatClkBias*1e-9)
                    SatComPos = computeSatComPos(SatLabel, TransmissionTime, SatPosInfo, Sod, Conf["PosInterpOrder"])
                    # Compute flight time (in seconds)
                    FlightTime = np.linalg.norm(SatComPos - RcvrRefPosXyz) / Const.SPEED_OF_LIGHT
                    # Apply Sagnac correction
                    SatComPos = applySagnac(SatComPos, FlightTime)
                    # Convert flight time to ms
                    SatCorrInfo["FlightTime"] = FlightTime * 1e3
                except Exception as e:
                    can_process = False

            # 3) ANTENNA PHASE OFFSET (APO) -------------------------------------
            if can_process:
                try:
                    Rcvr2SatCom = SatComPos - RcvrRefPosXyz # Vector from receiver to satellite CoM
                    SatApo, SatApoCorr = computeSatApo(Sod, Rcvr2SatCom, SunPos, SatLabel, SatComPos, SatApoInfo)
                    SatCopPos = SatComPos + SatApo 
                except Exception as e:
                    can_process = False

            # 4) RELATIVISTIC CORRECTION (DTR) -----------------------------------
            if (
                can_process and
                EpochBefore is not None and
                (Sod - EpochBefore <= 300)
            ):
                Dtr = computeDtr(SatComPos_1.get(SatLabel), SatComPos, Sod, Sod_1.get(SatLabel), SatPrepro["Elevation"], Conf)
            else:
                Dtr = Const.NAN

            # 5) TROPO DELAY AND SIGMA COMPUTATION --------------------------------
            if can_process and is_valid_dtr(Dtr): # only if Dtr is valid
                try:
                    TropoMpp = computeTropoMpp(SatPrepro["Elevation"])
                    sigma_tropo = computeSigmaTROPO(TropoMpp)
                    STD = computeSlantTropoDelay_MOPS(Rcvr[3], Rcvr[4], Rcvr[5], Doy, TropoMpp) 
                    snoisediv, sigma_multipath, sigma_air = computeSigma(SatPrepro["Elevation"], Conf)

                    sigma_uere = computeSigmaUERE(
                        Conf["SP3_ACC"],
                        Conf["CLK_ACC"],
                        sigma_tropo,
                        sigma_air,
                    )
                    # if sigma_uere <= 0.0:
                        # print(f"Skip {SatLabel} @SOD={Sod}: SigmaUERE<=0 (check cfg)")

                except Exception as e:
                    # print(f"Skip {SatLabel} @SOD={Sod}: Tropo/AIR/UERE fail ({e})")
                    can_process = False

            # 6) Corrected measurements and residuals -------------------------------------
            required_vars = ["SatClkBias", "SatCopPos", "Dtr",]
            missing = []
            for v in required_vars:
                try:
                    _ = eval(v)
                except NameError:
                    missing.append(v)

            # Satellite clock bias (in seconds) -> meters
            SatCorrInfo["SatClk"] = SatClkBias * Const.SPEED_OF_LIGHT # in meters
            SatCorrInfo["SatX"], SatCorrInfo["SatY"], SatCorrInfo["SatZ"] = SatCopPos
            SatCorrInfo["ApoX"], SatCorrInfo["ApoY"], SatCorrInfo["ApoZ"] = SatApo
                    
            if can_process and is_valid_dtr(Dtr) and len(missing) == 0:
                try:
                    # Compute geometrical range between Sat and Receiver reference position
                    SatCorrInfo["GeomRange"] = np.linalg.norm(SatCopPos - RcvrRefPosXyz)
                    SatCorrInfo["Dtr"] = Dtr * Const.SPEED_OF_LIGHT  # in meters
                    SatCorrInfo["SatClk"] += SatCorrInfo["Dtr"]
                    SatCorrInfo["Std"] = STD
                    SatCorrInfo["SigmaTropo"] = sigma_tropo
                    SatCorrInfo["SigmaAirborne"] = sigma_air
                    SatCorrInfo["SigmaNoiseDiv"] = snoisediv
                    SatCorrInfo["SigmaMultipath"] = sigma_multipath
                    SatCorrInfo["SigmaUere"] = sigma_uere
                    
                    # Code/Phase after Tropospheric correction (STD), Dtr and SatClkBias added
                    SatCorrInfo["CorrCode"] = SatPrepro["IF_C"] + SatCorrInfo["SatClk"] - STD 
                    SatCorrInfo["CorrPhase"] = SatPrepro["IF_L"] + SatCorrInfo["SatClk"] - STD 
                
                    # Compute 1st residuals: difference between corrected measurement and model

                    SatCorrInfo["CodeResidual"] = SatCorrInfo["CorrCode"] - SatCorrInfo["GeomRange"] 
                    SatCorrInfo["PhaseResidual"]  = SatCorrInfo["CorrPhase"] - SatCorrInfo["GeomRange"] 
                    SatCorrInfo["Flag"] = 1

                except Exception as e:
                    print(f"Skip {SatLabel} @SOD={Sod}: residuals fail ({e})")
            else:
                SatCorrInfo["Dtr"] = Const.NAN
                # print(f"Skip {SatLabel} @SOD={Sod}: missing vars for residuals: {missing}")

            
        # Always store the satellite info, even if skipped -------------------------------------
        CorrInfo[SatLabel] = SatCorrInfo 
        can_process = False

        try:
            SatComPos_1[SatLabel] = SatComPos # current SatComPos 
            Sod_1[SatLabel] = Sod # Sod is current seconds-of-day (float)
        except UnboundLocalError:
            pass  # Skip saving if SatComPos wasn't defined


    # RECEIVER CLOCK ESTIMATION  --------------------------------------------------------------
    # This block estimates the receiver clock bias as a weighted average
    # of the code residuals, using weights W = 1 / (SigmaUERE^2).
    # It also includes a simple outlier rejection step.

    num = 0.0
    den = 0.0

    res_list = []
    w_list = []
    labels = []

    # 1) Collect residuals and weights for all valid satellites
    for SatLabel, SatCorrInfo in CorrInfo.items():
        if SatCorrInfo["Flag"] != 1:
            continue

        # Optional: filter by elevation to avoid low-elevation satellites
        if SatCorrInfo["Elevation"] < 15.0:
            continue

        res = SatCorrInfo["CodeResidual"]   # CorrCode - GeomRange (before RcvrClk correction)
        sigma = SatCorrInfo["SigmaUere"]

        if sigma <= 0:
            continue

        res_list.append(res)
        w_list.append(1.0 / (sigma**2))
        labels.append(SatLabel)

    # 2) Outlier rejection (Median Absolute Deviation method)
    if len(res_list) > 0:
        res_arr = np.array(res_list)
        w_arr = np.array(w_list)

        median = np.median(res_arr)
        mad = np.median(np.abs(res_arr - median))
        if mad == 0:
            mad = 1.0  # avoid division by zero

        # Keep only satellites within 3*MAD of the median
        keep = np.abs(res_arr - median) <= (3.0 * mad)
        res_arr = res_arr[keep]
        w_arr = w_arr[keep]

        # 3) Weighted average to estimate receiver clock
        num = np.sum(res_arr * w_arr)
        den = np.sum(w_arr)
        RcvrClk = (num / den) if den > 0 else 0.0
    else:
        RcvrClk = 0.0

    # 4) Apply RcvrClk correction to all residuals
    for SatLabel, SatCorrInfo in CorrInfo.items():
        SatCorrInfo["RcvrClk"] = RcvrClk
        if SatCorrInfo["Flag"] == 1:
            # Subtract RcvrClk from residuals to center them around zero
            SatCorrInfo["CodeResidual"] -= RcvrClk
            SatCorrInfo["PhaseResidual"] -= RcvrClk
        # print(f"{SatLabel} ... {SatCorrInfo['CodeResidual']:.4f} ... {SatCorrInfo['RcvrClk']:.4f} ... {RcvrClk:.4f}")

    return CorrInfo


def is_valid_dtr(dtr):
    """
    Check if relativistic correction Dtr is valid.
    Typical values are within ±5e-8 s (~±15 m).
    """
    if not isinstance(dtr, float):
        return False
    if dtr == Const.NAN or np.isnan(dtr):
        return False
    # Reject extreme values (outside ±5e-8 s)
    if abs(dtr) > 5e-8:
        return False
    return True


# def computeSatComPos(SatLabel, TransmissionTime, SatPosInfo, Sod, PosInterpOrder):
#     """
#     Computes the satellite Center of Mass (CoM) position at the signal transmission time
#     using 10-point Lagrange interpolation over SP3 position data.

#     Parameters
#     ----------
#     SatLabel : str
#         Satellite identifier (e.g., 'G27').
#     TransmissionTime : float
#         Signal transmission time in seconds of day (SoD).
#     SatPosInfo : dict
#         Dictionary containing SP3 position data for each satellite.
#     Sod : float
#         Current epoch time in seconds of day (not used here but kept for interface consistency).
#     PosInterpOrder : int
#         Number of interpolation points (must be 10: 5 before + 5 after).

#     Returns
#     -------
#     np.array
#         Interpolated satellite position [X, Y, Z] in meters.
#     """
#     if PosInterpOrder != 10:
#         raise ValueError("Expected 10-point Lagrange interpolation (5 before + 5 after)")

#     # Step 1: Extract and sort available epochs and positions for the satellite
#     time_list = np.array(sorted(SatPosInfo[SatLabel]["Epochs"]))  # SP3 epochs in seconds of day
#     pos_list = np.array(SatPosInfo[SatLabel]["Positions"])        # Corresponding XYZ positions in kilometers

#     # Step 2: If TransmissionTime matches an SP3 epoch, return the exact position (no interpolation)
#     if TransmissionTime in time_list:
#         idx = np.where(time_list == TransmissionTime)[0][0]
#         return np.array(pos_list[idx]) * 1000  # Convert from km to m

#     # Step 3: Separate epochs before and after the transmission time
#     before_mask = time_list < TransmissionTime
#     after_mask = time_list > TransmissionTime

#     times_before = time_list[before_mask]
#     times_after = time_list[after_mask]
#     pos_before = pos_list[before_mask]
#     pos_after = pos_list[after_mask]

#     # Step 4: Select interpolation window
#     if len(times_before) >= 5 and len(times_after) >= 5:
#         # Standard case: 5 before + 5 after
#         times = np.concatenate([times_before[-5:], times_after[:5]])
#         positions = np.concatenate([pos_before[-5:], pos_after[:5]])
#     elif len(times_before) < 5 and len(times_after) >= PosInterpOrder:
#         # Early edge case: not enough before, use 10 future points
#         times = times_after[:PosInterpOrder]
#         positions = pos_after[:PosInterpOrder]
#     elif len(times_after) < 5 and len(times_before) >= PosInterpOrder:
#         # Late edge case: not enough after, use 10 past points
#         times = times_before[-PosInterpOrder:]
#         positions = pos_before[-PosInterpOrder:]
#     else:
#         raise ValueError("Not enough SP3 points for interpolation")

#     # Step 5: Define the Lagrange interpolation function
#     # def lagrange_interp(x, x_vals, y_vals):
#     #     result = 0.0
#     #     for i in range(len(x_vals)):
#     #         term = y_vals[i]
#     #         for j in range(len(x_vals)):
#     #             if i != j:
#     #                 term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
#     #         result += term
#     #     return result

#     def lagrange_interp(x, x_vec, y_vec):
#         if len(x_vec) != len(y_vec):
#             raise ValueError("x_vec and y_vec must have the same length.")

#         n = len(x_vec)
#         L = np.ones(n)

#         for i in range(n):
#             for j in range(n):
#                 if i != j:
#                     L[i] *= (x - x_vec[j]) / (x_vec[i] - x_vec[j])

#         return sum(y_vec[i] * L[i] for i in range(n))

#     # Step 6: Interpolate each coordinate axis independently
#     x = lagrange_interp(TransmissionTime, times, positions[:, 0])
#     y = lagrange_interp(TransmissionTime, times, positions[:, 1])
#     z = lagrange_interp(TransmissionTime, times, positions[:, 2])

#     # Step 7: Convert from kilometers to meters
#     return np.array([x, y, z]) * 1000




def computeSatComPos(SatLabel, TransmissionTime, SatPosInfo, Sod, PosInterpOrder):
    """
    Computes the satellite Center of Mass (CoM) position at the signal transmission time
    using 10-point Lagrange interpolation over SP3 position data.

    Parameters
    ----------
    SatLabel : str
        Satellite identifier (e.g., 'G27').
    TransmissionTime : float
        Signal transmission time in seconds of day (SoD).
    SatPosInfo : dict
        Dictionary containing SP3 position data for each satellite.
    Sod : float
        Current epoch time in seconds of day (not used here but kept for interface consistency).
    PosInterpOrder : int
        Number of interpolation points (must be 10: 5 before + 5 after).

    Returns
    -------
    np.array
        Interpolated satellite position [X, Y, Z] in meters.
    """
    if PosInterpOrder != 10:
        raise ValueError("Expected 10-point Lagrange interpolation (5 before + 5 after)")

    # Step 1: Extract and sort available epochs and positions for the satellite
    time_list = np.array(sorted(SatPosInfo[SatLabel]["Epochs"]))  # SP3 epochs in seconds of day
    pos_list = np.array(SatPosInfo[SatLabel]["Positions"])        # Corresponding XYZ positions in kilometers

    # Step 2: If TransmissionTime matches an SP3 epoch, return the exact position (no interpolation)
    if TransmissionTime in time_list:
        idx = np.where(time_list == TransmissionTime)[0][0]
        return np.array(pos_list[idx]) * 1000  # Convert from km to m

    # Step 3: Select interpolation window 
    HigherIdx = bisect_right(time_list, TransmissionTime)
    NumSp3Records = len(time_list)

    if HigherIdx > 4 and HigherIdx < (NumSp3Records - 5):
        FirstIdx = HigherIdx - 5
    elif HigherIdx <= 4:
        FirstIdx = 0
    else:
        FirstIdx = NumSp3Records - PosInterpOrder

    interp_times = time_list[FirstIdx:FirstIdx + PosInterpOrder]
    interp_positions = pos_list[FirstIdx:FirstIdx + PosInterpOrder]

    # Step 4: Define the Lagrange interpolation function
    def lagrange_interp(x, x_vec, y_vec):
        if len(x_vec) != len(y_vec):
            raise ValueError("x_vec and y_vec must have the same length.")
        n = len(x_vec)
        L = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    L[i] *= (x - x_vec[j]) / (x_vec[i] - x_vec[j])
        return sum(y_vec[i] * L[i] for i in range(n))

    # Step 5: Interpolate each coordinate axis independently
    x = lagrange_interp(TransmissionTime, interp_times, interp_positions[:, 0])
    y = lagrange_interp(TransmissionTime, interp_times, interp_positions[:, 1])
    z = lagrange_interp(TransmissionTime, interp_times, interp_positions[:, 2])

    # Step 6: Convert from kilometers to meters
    return np.array([x, y, z]) * 1000




# def applySagnac(SatComPos, FlightTime):

#     # Rotation vector
#     omega_vec = np.array([0.0, 0.0, Const.OMEGA_EARTH]) # OMEGA_EARTH = Earth's rotation rate in rad/s

#     # Cross product: omega x SatComPos
#     sagnac_shift = np.cross(omega_vec, SatComPos) * FlightTime

#     # Corrected position
#     SatComPosCorr = SatComPos - sagnac_shift

#     return SatComPosCorr


def applySagnac(SatComPos, FlightTime):
    # Earth rotation angle during signal flight
    theta = Const.OMEGA_EARTH * FlightTime  # rad

    c, s = np.cos(theta), np.sin(theta)

    # Rotation about Z-axis (ECEF)
    Rz = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Rotate the satellite position from transmit to receive frame
    return Rz @ SatComPos

# def computeSatApo(SatComPos, RcvrPos, SunPos, ApoBody):
#     """
#     Computes satellite Antenna Phase Offset (APO) in ECEF.
#     ApoBody: np.array([X, Y, Z]) in satellite body frame (meters)
#     """
#     # Get unit vectors i, j, k (satellite body frame)
#     z_axis = -SatComPos / np.linalg.norm(SatComPos) 
#     x_axis = SunPos - np.dot(SunPos, z_axis) * z_axis
#     x_axis /= np.linalg.norm(x_axis)

#     x_axis = SunPos / np.linalg.norm(SunPos)
#     y_axis = np.cross(z_axis, x_axis)
#     y_axis /= np.linalg.norm(y_axis)
#     x_axis = np.cross(y_axis, z_axis)


#     # Build rotation matrix from satellite body frame to ECEF
#     R = np.column_stack((x_axis, y_axis, z_axis))

#     # Transform to ECEF
#     ApoEcef = R @ ApoBody
#     return ApoEcef


def computeSatApo(Sod, Rcvr2SatCom, SunPos, SatLabel, SatComPos, SatApoInfo):

    def crossProd(a, b):
        return np.cross(a, b)
    
    SatApoPrn = SatApoInfo[SatLabel]

    SatApoL1 = SatApoPrn["L1"] / 1000
    SatApoL2 = SatApoPrn["L2"] / 1000

    # get z direction
    k = -SatComPos/np.linalg.norm(SatComPos)

    # find e
    Sat2Sun = (SunPos - SatComPos)
    e = Sat2Sun / np.linalg.norm(Sat2Sun)

    # get y direction parallel to solar panels
    j = crossProd(k,e)
    j = j / np.linalg.norm(j)

    # get x direction
    i = crossProd(j,k)
    i = i / np.linalg.norm(i)

    # compute rotation matrix
    R = np.vstack((i,j,k)).T

    # compute Apo in ECEF
    SatApoL1 = R @ SatApoL1
    SatApoL2 = R @ SatApoL2

    # compute iono free APO in ECEF
    SatApo = (SatApoL2 - (Const.GPS_GAMMA_L1L2 * SatApoL1)) / (1 - Const.GPS_GAMMA_L1L2)

    # compute APO correction
    SatApoCorr = np.dot(Rcvr2SatCom/np.linalg.norm(Rcvr2SatCom), SatApo)

    return SatApo, SatApoCorr

def computeDtr(SatComPos_1, SatComPos, Sod, Sod_1, Elevation, Conf):
    """
    Computes relativistic correction Dtr using velocity approximation.
    Returns Const.NAN if inputs are invalid or calculation fails.
    """
    try:
        if SatComPos is None or SatComPos_1 is None:
            return Const.NAN
        if len(SatComPos) != 3 or len(SatComPos_1) != 3:
            return Const.NAN

        dt = Sod - Sod_1
        if dt == 0:
            return Const.NAN

        vel = (SatComPos - SatComPos_1) / dt
        r_dot_v = np.dot(SatComPos, vel)

        dtr = -2.0 * r_dot_v / Const.SPEED_OF_LIGHT**2

        # Reject invalid or extreme values
        if np.isnan(dtr) or dtr == Const.NAN:
            return Const.NAN
        # Typical Dtr is within ±5e-8 s (~±15 m). Discard if outside.
        if abs(dtr) > 5e-8:
            return Const.NAN

        return dtr

    except Exception:
        return Const.NAN
    
def computeTropoMpp(Elevation):
    """
    Computes tropospheric mapping function from elevation angle.
    """
    # --- Mapping function (valid for elevation ≥ 4°)
    El_rad = np.radians(Elevation)
    return 1.001 / np.sqrt(0.002001 + np.sin(El_rad)**2)

def computeSigmaTROPO(TropoMpp):
    """
    Computes sigma of tropospheric delay error.
    """
    sigma_ztd = 0.12  # Zenith delay uncertainty [m]
    return sigma_ztd * TropoMpp


def computeSlantTropoDelay_MOPS(RcvrLon, RcvrLat, RcvrH, Doy, TropoMpp):
    """
    Computes slant tropospheric delay using MOPS model.
    Inputs:
        RcvrLon        : Longitude of receiver [degrees]
        RcvrLat        : Latitude of receiver [degrees]
        RcvrH          : Height of receiver [meters]
        Doy            : Day of year [1–365]
        Elevation      : Satellite elevation angle [degrees]
    Output:
        STD            : Slant Tropospheric Delay [meters]
    """

    # --- RTCA DO-229 tables (average and seasonal values)
    avg_table = {
        "P": [(15, 1013.25), (30, 1017.25), (45, 1015.75), (60, 1011.75), (75, 1013.00)],
        "T": [(15, 299.65), (30, 294.15), (45, 283.15), (60, 272.15), (75, 263.65)],
        "e": [(15, 26.31), (30, 21.79), (45, 11.66), (60, 6.78), (75, 4.11)],
        "beta": [(15, 6.30*1e-3), (30, 6.05*1e-3), (45, 5.58*1e-3), (60, 5.39*1e-3), (75, 4.53*1e-3)],
        "lambda": [(15, 2.77), (30, 3.15), (45, 2.57), (60, 1.81), (75, 1.55)],
    }
    seasonal_table = {
        "P": [(15, 0.00), (30, -3.75), (45, -2.25), (60, -1.75), (75, -0.5)],
        "T": [(15, 0.00), (30, 7.00), (45, 11.00), (60, 15.00), (75, 14.50)],
        "e": [(15, 0.00), (30, 8.85), (45, 7.24), (60, 5.36), (75, 3.39)],
        "beta": [(15, 0.00), (30, 0.25*1e-3), (45, 0.32*1e-3), (60, 0.81*1e-3), (75, 0.62*1e-3)],
        "lambda": [(15, 0.00), (30, 0.33), (45, 0.46), (60, 0.74), (75, 0.30)],
    }

    def interpolate(lat, table):
        
        lat = abs(lat)  
        # Linear interpolation by latitude
        for i in range(len(table) - 1):
            lat_i, val_i = table[i]
            lat_j, val_j = table[i + 1]
            if lat_i <= lat <= lat_j:
                return val_i + (val_j - val_i) * (lat - lat_i) / (lat_j - lat_i)
        return table[-1][1]  # fallback extrapolation
    
    
    if abs(RcvrLat) <= 15:
        P0         = avg_table["P"][0][1]
        T0         = avg_table["T"][0][1]
        e0         = avg_table["e"][0][1]
        beta0      = avg_table["beta"][0][1]
        lambda0    = avg_table["lambda"][0][1]
        deltaP     = seasonal_table["P"][0][1]
        deltaT     = seasonal_table["T"][0][1]
        deltae     = seasonal_table["e"][0][1]
        deltabeta  = seasonal_table["beta"][0][1]
        deltalambda= seasonal_table["lambda"][0][1]

    elif abs(RcvrLat) >= 75:
        P0         = avg_table["P"][-1][1]
        T0         = avg_table["T"][-1][1]
        e0         = avg_table["e"][-1][1]
        beta0      = avg_table["beta"][-1][1]
        lambda0    = avg_table["lambda"][-1][1]
        deltaP     = seasonal_table["P"][-1][1]
        deltaT     = seasonal_table["T"][-1][1]
        deltae     = seasonal_table["e"][-1][1]
        deltabeta  = seasonal_table["beta"][-1][1]
        deltalambda= seasonal_table["lambda"][-1][1]

    else:
        P0         = interpolate(RcvrLat, avg_table["P"])
        T0         = interpolate(RcvrLat, avg_table["T"])
        e0         = interpolate(RcvrLat, avg_table["e"])
        beta0      = interpolate(RcvrLat, avg_table["beta"])
        lambda0    = interpolate(RcvrLat, avg_table["lambda"])
        deltaP     = interpolate(RcvrLat, seasonal_table["P"])
        deltaT     = interpolate(RcvrLat, seasonal_table["T"])
        deltae     = interpolate(RcvrLat, seasonal_table["e"])
        deltabeta  = interpolate(RcvrLat, seasonal_table["beta"])
        deltalambda= interpolate(RcvrLat, seasonal_table["lambda"])

    # --- Seasonal correction
    D_min = 28 if RcvrLat >= 0 else 211
    cos_term = np.cos(2 * np.pi * (Doy - D_min) / 365.25)
    P = P0 - deltaP * cos_term
    T = T0 - deltaT * cos_term
    e = e0 - deltae * cos_term
    beta = beta0 - deltabeta * cos_term
    lambda0 = lambda0 - deltalambda * cos_term

    # --- Physical constants
    k1 = 77.604 # K/mbar
    k2 = 382000 # K²/mbar
    Rd = 287.054 # J/(kg⋅K)
    gm = 9.784 # m/s²
    g = 9.80665 # m/s²

    # # --- Zenith delays at zero-altitude
    z_dry = 1e-6 * k1 * Rd * P / gm
    z_wet = 1e-6 * (k2 * Rd * e) / ( (gm * (lambda0 + 1) - beta*Rd) * T)

    #  H = h (heigh of receiver in WGS) - N (from Tropo.py computeGeoidHeight(Lon, Lat))
    H = RcvrH - computeGeoidHeight(RcvrLon, RcvrLat)
    
    base = 1 - ((beta * H) / T)
    if base <= 0:
        return Const.NAN
    else:
        d_dry = z_dry * (base ** (g / (Rd * beta)))
        d_wet = z_wet * (base ** (( (lambda0 +1)*g  / (Rd * beta)) -1)  )

    # --- Final slant tropospheric delay
    STD = (d_dry + d_wet) * TropoMpp

    return STD

def computeSigma(Elevation, Conf, smoothing_time=30.0):
    """
    Compute sigma components used by PEPPUS/SBAS (noise, divergence, combined,
    multipath and airborne sigma) following MOPS guidance and the professor's
    note about applying SIGMA_AIR_DF.

    Inputs:
        Elevation      : satellite elevation angle (degrees)
        Conf           : configuration dict (must contain ELEV_NOISE_TH and
                         SIGMA_AIR_DF and AIR_ACC_DESIG optionally)
        smoothing_time : airborne smoothing filter nominal time constant (s).
                         Used to estimate a conservative sigma_div value.
    Returns:
        sigma_noise    : sigma of receiver noise (m)
        sigma_div      : sigma due to ionospheric divergence (m)
        snoisediv      : sqrt(sigma_noise^2 + sigma_div^2) (m)
        sigma_multipath: multipath sigma (m)
        sigma_air      : sqrt(snoisediv^2 + sigma_multipath^2) (m)
    Notes / choices made:
      - Uses MOPS steady-state combined-limits for GPS / designator A:
          max-signal limit = 0.15 m
          min-signal limit = 0.36 m
        (These are the numbers in MOPS Appendix A for GPS and the
         airborne accuracy rows you cited.)
      - Applies Conf["SIGMA_AIR_DF"] multiplicative factor to both
        the noise/div combined term and the multipath term (per .cfg file).
      - Estimates sigma_div from iono divergence rate (0.018 m/s) using
        a conservative dependence on smoothing_time:
            sigma_div = 0.018 * sqrt(smoothing_time)
        This choice is conservative but small for typical smoothing_time = 30
        (0.018*sqrt(30) ≈ 0.0985 m). If you have the exact filter response,
        replace this with the true steady-state residual.
      - If estimated sigma_div > snoisediv_limit*factor, clamp sigma_div to that limit
        and set sigma_noise = 0 (conservative).
      - Otherwise set sigma_noise = sqrt(snoisediv^2 - sigma_div^2).
    """

    # --- 1) multipath base model (DO-229 style)
    mp_base = 0.13 + 0.53 * np.exp(-Elevation / 10.0)   # meters

    # --- 2) choose MOPS combined steady-state limit depending on elevation (min/max signal)
    # For GPS, designator A limits:
    if Conf["AIR_ACC_DESIG"] == "A":
        if Elevation >= Conf["ELEV_NOISE_TH"]:
            snoisediv_limit = 0.15   # maximum signal level (m)
        else:
            snoisediv_limit = 0.36   # minimum signal level (m)
    else:
        # If AIR_ACC_DESIG == 'B'
        if Elevation >= Conf["ELEV_NOISE_TH"]:
            snoisediv_limit = 0.11   # designator B, max-level 
        else:
            snoisediv_limit = 0.15   # designator B, min-level 

    # apply airborne DF factor (prof's note: apply to both MP and Noise components)
    snoisediv_target = snoisediv_limit * Conf["SIGMA_AIR_DF"]
    sigma_multipath = mp_base * Conf["SIGMA_AIR_DF"]

    # --- 3) conservative estimate of sigma_div (steady-state residual due to iono divergence)
    # MOPS divergence rate:
    iono_rate = 0.018  # m/s (MOPS)

    # We pick a conservative mapping from smoothing_time to steady-state residual:
    #   sigma_div ≈ iono_rate * sqrt(smoothing_time)

    # Rationale: the divergence *rate* integrated over some effective filter response time
    # gives a residual; sqrt() is conservative but keeps sigma_div smaller than naive iono_rate * smoothing_time.

    if smoothing_time is None:
        smoothing_time = 30.0
    smoothing_time = float(smoothing_time)
    sigma_div_est = iono_rate * np.sqrt(max(1.0, smoothing_time))  # never < iono_rate according to MOPS guideline

    # --- 4) enforce combined limit
    if sigma_div_est >= snoisediv_target:
        # iono divergence alone already exceeds allowed combined limit: clamp
        sigma_div = snoisediv_target
        sigma_noise = 0.0
        snoisediv = snoisediv_target
    else:
        sigma_div = sigma_div_est
        # remaining budget goes to sigma_noise
        sigma_noise = np.sqrt(max(0.0, snoisediv_target**2 - sigma_div**2))
        snoisediv = np.sqrt(sigma_noise**2 + sigma_div**2)  # should be snoisediv_target (numerical)

    # --- 5) total airborne equipment sigma
    sigma_air = np.sqrt(snoisediv**2 + sigma_multipath**2)

    return snoisediv, sigma_multipath, sigma_air


def estimateRcvrClk(CodeResiduals, SigmaUEREs):
    """
    Estimates receiver clock bias using weighted average of code residuals.
    """
    weights = 1.0 / np.array(SigmaUEREs)**2
    weighted_sum = np.sum(weights * np.array(CodeResiduals))
    total_weight = np.sum(weights)
    return weighted_sum / total_weight

def computeSigmaUERE(sigma_sp3, sigma_clk, sigma_tropo, sigma_air):
    """
    Combines all sigma components to compute UERE.
    """
    sigma_clk *= Const.SPEED_OF_LIGHT * 1e-9
    sigma_sp3 *= 1e-2 
    return np.sqrt(sigma_sp3**2 + sigma_clk**2 + sigma_tropo**2 + sigma_air**2)

def interpolate_from_dict(clk_series, sod, default=0.0):
    xs = list(clk_series.keys())
    ys = list(clk_series.values())
    # print(f"interp xs={xs[:3]}, ys={ys[:3]}")  # validating the result
    if sod < xs[0] or sod > xs[-1]:
        return default
    return np.interp(sod, xs, ys) 

