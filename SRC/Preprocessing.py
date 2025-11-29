#!/usr/bin/env python

########################################################################
# PETRUS/SRC/Preprocessing.py:
# This is the Preprocessing Module of PEPPUS tool
#
#  Project:        PEPPUS
#  File:           Preprocessing.py
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
from collections import OrderedDict
from COMMON import GnssConstants as Const
from InputOutput import RcvrIdx, ObsIdx, REJECTION_CAUSE
from InputOutput import FLAG, VALUE, TH, CSNEPOCHS, CSNPOINTS, CSPDEGREE
import numpy as np
from COMMON.Iono import computeIonoMappingFunction

# Preprocessing internal functions
#-----------------------------------------------------------------------


def runPreProcMeas(Conf, Rcvr, ObsInfo, PrevPreproObsInfo):
    
    # Purpose: preprocess GNSS raw measurements from OBS file
    #          and generate PREPRO OBS file with the cleaned,
    #          smoothed measurements

    #          More in detail, this function handles:

    #          * Measurements cleaning and validation and exclusion due to different 
    #          criteria as follows:
    #             - Minimum Masking angle
    #             - Maximum Number of channels
    #             - Minimum Carrier-To-Noise Ratio (CN0)
    #             - Pseudo-Range Output of Range 
    #             - Maximum Pseudo-Range Step
    #             - Maximum Pseudo-Range Rate
    #             - Maximum Carrier Phase Increase
    #             - Maximum Carrier Phase Increase Rate
    #             - Data Gaps checks and handling 
    #             - Cycle Slips detection

    #          * Build iono-free combination

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
    # PrevPreproObsInfo: dict
    #         Preprocessed observations for previous epoch per sat
    #         PrevPreproObsInfo["G01"]["C1"]

    # Returns
    # =======
    # PreproObsInfo: dict
    #         Preprocessed observations for current epoch per sat
    #         PreproObsInfo["G01"]["C1"]

    # Initialize output
    PreproObsInfo = OrderedDict({})

    # Loop over satellites
    for SatObs in ObsInfo:
        # Initialize output info
        SatPreproObsInfo = {
            "Sod": 0.0,                         # Second of day
            "Doy": 0,                           # Day of year
            "Elevation": 0.0,                   # Elevation
            "Azimuth": 0.0,                     # Azimuth
            "C1": 0.0,                          # GPS L1C/A pseudorange
            "L1": 0.0,                          # GPS L1 carrier phase (in cycles)
            "L1Meters": 0.0,                    # GPS L1 carrier phase (in m)
            "S1": 0.0,                          # GPS L1C/A C/No
            "P2": 0.0,                          # GPS L2P pseudorange
            "L2": 0.0,                          # GPS L2 carrier phase
            "L2Meters": 0.0,                    # GPS L2 carrier phase (in m)
            "S2": 0.0,                          # GPS L2 C/No
            "IF_C": 0.0,                        # Iono-Free combination of codes
            "IF_L": 0.0,                        # Iono-Free combination of phases
            "Status": 1,                        # Measurement status
            "RejectionCause": 0,                # Cause of rejection flag
            "CodeRate": Const.NAN,              # Code Rate
            "CodeRateStep": Const.NAN,          # Code Rate Step
            "PhaseRate": Const.NAN,             # Phase Rate
            "PhaseRateStep": Const.NAN,         # Phase Rate Step
            "GF_L": Const.NAN,                  # Geometry-Free combination of phases in meters
            "VtecRate": Const.NAN,              # VTEC Rate
            "iAATR": Const.NAN,                 # Instantaneous AATR
            "Mpp": 0.0,                         # Iono Mapping
            
        } # End of SatPreproObsInfo

        # Get satellite label
        SatLabel = SatObs[ObsIdx["CONST"]] + "%02d" % int(SatObs[ObsIdx["PRN"]])

        # Prepare outputs
        # Get SoD
        SatPreproObsInfo["Sod"] = float(SatObs[ObsIdx["SOD"]])
        # Get DoY
        SatPreproObsInfo["Doy"] = int(SatObs[ObsIdx["DOY"]])
        # Get Elevation
        SatPreproObsInfo["Elevation"] = float(SatObs[ObsIdx["ELEV"]])
        # Get Azimuth
        SatPreproObsInfo["Azimuth"] = float(SatObs[ObsIdx["AZIM"]])
        # Get C1
        SatPreproObsInfo["C1"] = float(SatObs[ObsIdx["C1"]])
        # Get L1 in cycles and in m
        SatPreproObsInfo["L1"] = float(SatObs[ObsIdx["L1"]])
        SatPreproObsInfo["L1Meters"] = float(SatObs[ObsIdx["L1"]]) * Const.GPS_L1_WAVE
        # Get S1
        SatPreproObsInfo["S1"] = float(SatObs[ObsIdx["S1"]])
        # Get P2
        SatPreproObsInfo["P2"] = float(SatObs[ObsIdx["P2"]])
        # Get L2 in cycles and in m
        SatPreproObsInfo["L2"] = float(SatObs[ObsIdx["L2"]])
        SatPreproObsInfo["L2Meters"] = float(SatObs[ObsIdx["L2"]]) * Const.GPS_L2_WAVE
        # Get S2
        SatPreproObsInfo["S2"] = float(SatObs[ObsIdx["S2"]])

        # Prepare output for the satellite
        PreproObsInfo[SatLabel] = SatPreproObsInfo

    # Limit the satellites to the Number of Channels
    # ----------------------------------------------------------
    # Initialize Elevation cut due to number of channels limitation
    ChannelsElevation = 0.0

    # Get difference between number of satellites and number of channels
    NChannelsRejections = len(PreproObsInfo) - int(Conf["NCHANNELS_GPS"])

    # If some satellites shall be rejected
    if NChannelsRejections > 0:
        # Initialize Elevation list for number of channels limitation
        ElevationList = np.array([])

        # Loop over satellites to build elevation list
        for SatLabel, PreproObs in PreproObsInfo.items():
            ElevationList = np.concatenate((ElevationList, [PreproObs["Elevation"]]))

        # Sort elevation list
        ElevationList = sorted(ElevationList)

        # Get Elevation cut
        ChannelsElevation = ElevationList[NChannelsRejections]

    # Loop over satellites
    for SatLabel, PreproObs in PreproObsInfo.items():
        # Get epoch
        Epoch = PreproObs["Sod"]

        # Check data gaps
        # ----------------------------------------------------------
        # Compute gap between previous and current observation
        DeltaT = Epoch - PrevPreproObsInfo[SatLabel]["PrevEpoch"]

        # If there is a gap
        if DeltaT > Conf["MAX_GAP"][TH]:
            # If check gaps activated
            if Conf["MAX_GAP"][FLAG] == 1:
                # Reject the measurement and indicate the rejection cause
                # (if it is not a non-visibility period)
                if(PrevPreproObsInfo[SatLabel]["PrevRej"] != REJECTION_CAUSE["MASKANGLE"]):
                    PreproObs["RejectionCause"] = REJECTION_CAUSE["DATA_GAP"]

            # Reinitialize CS detection
            PrevPreproObsInfo[SatLabel]["GF_L_Prev"] = [0.0] * int(Conf["CYCLE_SLIPS"][CSNPOINTS])
            PrevPreproObsInfo[SatLabel]["GF_Epoch_Prev"] = [0.0] * int(Conf["CYCLE_SLIPS"][CSNPOINTS])
            PrevPreproObsInfo[SatLabel]["CycleSlipBuffIdx"] = 0
            PrevPreproObsInfo[SatLabel]["CycleSlipFlags"] = [0.0] * int(Conf["CYCLE_SLIPS"][CSNEPOCHS])
            PrevPreproObsInfo[SatLabel]["CycleSlipFlagIdx"] = 0

            # Reinitialize some variables
            PrevPreproObsInfo[SatLabel]["PrevCode"] = Const.NAN                                 # Previous Code IF
            PrevPreproObsInfo[SatLabel]["PrevPhase"] = Const.NAN                                # Previous Phase IF
            PrevPreproObsInfo[SatLabel]["PrevCodeRate"] = Const.NAN                             # Previous Code Rate IF
            PrevPreproObsInfo[SatLabel]["PrevPhaseRate"] = Const.NAN                            # Previous Phase Rate IF
            PrevPreproObsInfo[SatLabel]["PrevStec"] = Const.NAN                                   # Previous Geometry-Free Observable
            PrevPreproObsInfo[SatLabel]["PrevStecEpoch"] = Const.NAN                              # Previous Geometry-Free Observable epoch

            # Raise Reset Ambiguities flag
            PrevPreproObsInfo[SatLabel]["ResetAmb"] = 1

        # End of if DeltaT > Conf["MAX_GAP"][TH]:

        # Build Iono-free combination of codes and phases
        PreproObs["IF_C"] = (PreproObs["P2"] - Const.GPS_GAMMA_L1L2 * PreproObs["C1"]) / (1 - Const.GPS_GAMMA_L1L2)
        PreproObs["IF_L"] = (PreproObs["L2Meters"] - Const.GPS_GAMMA_L1L2 * PreproObs["L1Meters"]) / (1 - Const.GPS_GAMMA_L1L2)

        # Compute the Geometry-Free combination of phases
        PreproObs["GF_LCycles"] = PreproObs["L1"] - PreproObs["L2"]
        PreproObs["GF_L"] = PreproObs["L1Meters"] - PreproObs["L2Meters"]

        # If satellite shall be rejected due to number of channels limitation
        # --------------------------------------------------------------------------------------------------------------------
        if PreproObs["Elevation"] < ChannelsElevation:
            # Indicate the rejection cause
            PreproObs["RejectionCause"] = REJECTION_CAUSE["NCHANNELS_GPS"]
            PreproObs["Status"] = 0

        # If satellite shall be rejected due to mask angle
        # ----------------------------------------------------------
        if PreproObs["Elevation"] < Rcvr[RcvrIdx["MASK"]]:
            # Indicate the rejection cause
            PreproObs["RejectionCause"] = REJECTION_CAUSE["MASKANGLE"]
            PreproObs["Status"] = 0

        # If satellite shall be rejected due to C/N0 (only if activated in conf) - Both frequencies shall be checked
        # --------------------------------------------------------------------------------------------------------------------
        if (Conf["MIN_SNR"][FLAG] == 1):
            if(PreproObs["S1"] < float(Conf["MIN_SNR"][VALUE])):
                # Indicate the rejection cause
                PreproObs["RejectionCause"] = REJECTION_CAUSE["MIN_SNR_L1"]
                PreproObs["Status"] = 0

            if(PreproObs["S2"] < float(Conf["MIN_SNR"][VALUE])):
                # Indicate the rejection cause
                if(PreproObs["RejectionCause"] == REJECTION_CAUSE["MIN_SNR_L1"]):
                    PreproObs["RejectionCause"] = REJECTION_CAUSE["MIN_SNR_L1_L2"]
                    PreproObs["Status"] = 0

                else:
                    PreproObs["RejectionCause"] = REJECTION_CAUSE["MIN_SNR_L2"]
                    PreproObs["Status"] = 0

        # End of if (Conf["MIN_SNR"][FLAG] == 1) \

        # If satellite shall be rejected due to Pseudorange Out-of-range (only if activated in conf)
        # --------------------------------------------------------------------------------------------------------------------
        if (Conf["MAX_PSR_OUTRNG"][FLAG] == 1):
            if (PreproObs["C1"] > float(Conf["MAX_PSR_OUTRNG"][VALUE])):
                # Indicate the rejection cause
                PreproObs["RejectionCause"] = REJECTION_CAUSE["MAX_PSR_OUTRNG_L1"]
                PreproObs["Status"] = 0

            if (PreproObs["P2"] > float(Conf["MAX_PSR_OUTRNG"][VALUE])):
                # Indicate the rejection cause
                if PreproObs["RejectionCause"] == REJECTION_CAUSE["MAX_PSR_OUTRNG_L1"]:
                    PreproObs["RejectionCause"] = REJECTION_CAUSE["MAX_PSR_OUTRNG_L1_L2"]
                    PreproObs["Status"] = 0
                else:
                    PreproObs["RejectionCause"] = REJECTION_CAUSE["MAX_PSR_OUTRNG_L2"]
                    PreproObs["Status"] = 0

        # End of if (Conf["MAX_PSR_OUTRNG"][FLAG] == 1)

        # Cycle slips detection
        # fit a polynomial using previous GF measurements to compare the predicted value
        # with the observed one
        # --------------------------------------------------------------------------------------------------------------------
        # if check Cycle Slips activated
        if (Conf["CYCLE_SLIPS"][FLAG] == 1):
            
            # print("DBG CS:", Epoch, SatLabel, PrevPreproObsInfo[SatLabel]["CycleSlipBuffIdx"],
            # PrevPreproObsInfo[SatLabel]["GF_L_Prev"])

            # Get N
            N = PrevPreproObsInfo[SatLabel]["CycleSlipBuffIdx"]

            # If the buffer is full, we can detect the cycle slip with a polynom
            if N == (Conf["CYCLE_SLIPS"][CSNPOINTS]):
                # Adjust polynom to the samples in the buffer
                Polynom = np.polynomial.polynomial.polyfit(PrevPreproObsInfo[SatLabel]["GF_Epoch_Prev"],
                PrevPreproObsInfo[SatLabel]["GF_L_Prev"],
                int(Conf["CYCLE_SLIPS"][CSPDEGREE]))

                # Predict value evaluating the polynom
                TargetPred = np.polynomial.polynomial.polyval(PreproObs["Sod"],
                Polynom)

                # Compute Residual
                Residual = abs(PreproObs["GF_LCycles"] - TargetPred)

                # Compute CS flag
                CsFlag = Residual > Conf["CYCLE_SLIPS"][TH]

                # Update CS flag buffer
                PrevPreproObsInfo[SatLabel]["CycleSlipFlagIdx"] = \
                    (PrevPreproObsInfo[SatLabel]["CycleSlipFlagIdx"] + 1) % \
                        int(Conf["CYCLE_SLIPS"][CSNEPOCHS])
                PrevPreproObsInfo[SatLabel]["CycleSlipFlags"][PrevPreproObsInfo[SatLabel]["CycleSlipFlagIdx"]] = \
                    CsFlag

                # print("DBG: CSDETECT", Epoch, SatLabel, PreproObs["GF_LCycles"], TargetPred, Residual)

                # Check if threshold was exceeded CSNEPOCHS times
                if (np.sum(PrevPreproObsInfo[SatLabel]["CycleSlipFlags"]) == \
                    int(Conf["CYCLE_SLIPS"][CSNEPOCHS])):
                    # Rise flag and reinitialize arrays
                    PreproObs["RejectionCause"] = REJECTION_CAUSE["CYCLE_SLIP"]

                    # Raise Reset Ambiguities flag
                    PrevPreproObsInfo[SatLabel]["ResetAmb"] = 2

                    # Reinitialize CS detection
                    PrevPreproObsInfo[SatLabel]["GF_L_Prev"] = [0.0] * int(Conf["CYCLE_SLIPS"][CSNPOINTS])
                    PrevPreproObsInfo[SatLabel]["GF_Epoch_Prev"] = [0.0] * int(Conf["CYCLE_SLIPS"][CSNPOINTS])
                    PrevPreproObsInfo[SatLabel]["CycleSlipBuffIdx"] = 0
                    PrevPreproObsInfo[SatLabel]["CycleSlipFlags"] = [0.0] * int(Conf["CYCLE_SLIPS"][CSNEPOCHS])
                    PrevPreproObsInfo[SatLabel]["CycleSlipFlagIdx"] = 0

                    # Reinitialize some variables
                    PrevPreproObsInfo[SatLabel]["PrevCode"] = Const.NAN                                 # Previous Code IF
                    PrevPreproObsInfo[SatLabel]["PrevPhase"] = Const.NAN                                # Previous Phase IF
                    PrevPreproObsInfo[SatLabel]["PrevCodeRate"] = Const.NAN                             # Previous Code Rate IF
                    PrevPreproObsInfo[SatLabel]["PrevPhaseRate"] = Const.NAN                            # Previous Phase Rate IF
                    PrevPreproObsInfo[SatLabel]["PrevStec"] = Const.NAN                                   # Previous Geometry-Free Observable
                    PrevPreproObsInfo[SatLabel]["PrevStecEpoch"] = Const.NAN                              # Previous Geometry-Free Observable epoch

                # If threshold was exceeded less than CSNEPOCHS times,
                # don't update the buffer and set the measurement to invalid
                elif CsFlag == 1:
                    PreproObs["Status"] = 0

                # If threshold was not exceeded
                else:
                    # Leave space for the new sample
                    PrevPreproObsInfo[SatLabel]["GF_L_Prev"][:-1] = PrevPreproObsInfo[SatLabel]["GF_L_Prev"][1:]
                    PrevPreproObsInfo[SatLabel]["GF_Epoch_Prev"][:-1] = PrevPreproObsInfo[SatLabel]["GF_Epoch_Prev"][1:]

                    # Store new sample
                    PrevPreproObsInfo[SatLabel]["GF_L_Prev"][-1] = PreproObs["GF_LCycles"]
                    PrevPreproObsInfo[SatLabel]["GF_Epoch_Prev"][-1] = PreproObs["Sod"]

                # End of if (np.sum(PrevPreproObsInfo[SatLabel]["CycleSlipFlags"])

            # Buffer is not full, need to add new GF observable
            else:
                PrevPreproObsInfo[SatLabel]["GF_L_Prev"][N] = PreproObs["GF_LCycles"]
                PrevPreproObsInfo[SatLabel]["GF_Epoch_Prev"][N] = PreproObs["Sod"]
                PrevPreproObsInfo[SatLabel]["CycleSlipBuffIdx"] += 1

                PreproObs["Status"] = 0

            # End of if NumberMeasurements == Conf["CYCLE_SLIPS"][CSNEPOCHS]:

        # End of if (Conf["CYCLE_SLIPS"][FLAG] == 1)

        # If previous information is available
        if(PrevPreproObsInfo[SatLabel]["PrevPhase"] != Const.NAN):
            # Check Phase Rate (only if activated in conf)
            # --------------------------------------------------------------------------------------------------------------------
            # Compute Phase Rate in meters/second
            PreproObs["PhaseRate"] = (PreproObs["L1Meters"] - PrevPreproObsInfo[SatLabel]["PrevPhase"]) / DeltaT

            # Check Phase Rate
            if (Conf["MAX_PHASE_RATE"][FLAG] == 1) \
                    and (abs(PreproObs["PhaseRate"]) > Conf["MAX_PHASE_RATE"][VALUE]):
                        # Indicate the rejection cause
                        PreproObs["RejectionCause"] = REJECTION_CAUSE["MAX_PHASE_RATE"]
                        PreproObs["Status"] = 0

            # If there are enough samples
            if (PrevPreproObsInfo[SatLabel]["PrevPhaseRate"] != Const.NAN):
                # Check Phase Rate Step (only if activated in conf)
                # ----------------------------------------------------------
                # Compute Phase Rate Step in meters/second^2
                PreproObs["PhaseRateStep"] = (PreproObs["PhaseRate"] - PrevPreproObsInfo[SatLabel]["PrevPhaseRate"]) / DeltaT

                if (Conf["MAX_PHASE_RATE_STEP"][FLAG] == 1) \
                    and (abs(PreproObs["PhaseRateStep"]) > Conf["MAX_PHASE_RATE_STEP"][VALUE]):
                        # Indicate the rejection cause
                        PreproObs["RejectionCause"] = REJECTION_CAUSE["MAX_PHASE_RATE_STEP"]
                        PreproObs["Status"] = 0

            else: PreproObs["Status"] = 0

            # End of if (PrevPreproObsInfo[SatLabel]["PrevPhaseRate"] != Const.NAN):

        else: PreproObs["Status"] = 0

        # End of if(PrevPreproObsInfo[SatLabel]["PrevPhase"] != Const.NAN):

        # If previous information is available
        if(PrevPreproObsInfo[SatLabel]["PrevCode"] != Const.NAN):
            # Check Code Step (only if activated in conf)
            # --------------------------------------------------------------------------------------------------------------------
            # Compute Code Rate in meters/second
            PreproObs["CodeRate"] = (PreproObs["C1"] - PrevPreproObsInfo[SatLabel]["PrevCode"]) / DeltaT

            # Check Code Rate
            if (Conf["MAX_CODE_RATE"][FLAG] == 1) \
                    and (abs(PreproObs["CodeRate"]) > Conf["MAX_CODE_RATE"][VALUE]):
                        # Indicate the rejection cause
                        PreproObs["RejectionCause"] = REJECTION_CAUSE["MAX_CODE_RATE"]
                        PreproObs["Status"] = 0

            # If there are enough samples
            if (PrevPreproObsInfo[SatLabel]["PrevCodeRate"] != Const.NAN):
                # Compute Code Rate Step in meters/second^2
                PreproObs["CodeRateStep"] = (PreproObs["CodeRate"] - \
                    PrevPreproObsInfo[SatLabel]["PrevCodeRate"]) / DeltaT
                # Check Code Rate Step (only if activated in conf)
                # ----------------------------------------------------------
                if (Conf["MAX_CODE_RATE_STEP"][FLAG] == 1) \
                        and (abs(PreproObs["CodeRateStep"]) > Conf["MAX_CODE_RATE_STEP"][VALUE]):
                            # Indicate the rejection cause
                            PreproObs["RejectionCause"] = REJECTION_CAUSE["MAX_CODE_RATE_STEP"]
                            PreproObs["Status"] = 0

            else: PreproObs["Status"] = 0

            # End of if (PrevPreproObsInfo[SatLabel]["PrevCodeRate"] != Const.NAN):

        else: PreproObs["Status"] = 0

        # End of if(PrevPreproObsInfo[SatLabel]["PrevCode"] != Const.NAN)

        # Update previous values
        # ----------------------------------------------------------
        PrevPreproObsInfo[SatLabel]["PrevC1"] = PreproObs["C1"]
        PrevPreproObsInfo[SatLabel]["PrevP2"] = PreproObs["P2"]
        PrevPreproObsInfo[SatLabel]["PrevL1"] = PreproObs["L1"]
        PrevPreproObsInfo[SatLabel]["PrevL2"] = PreproObs["L2"]
        PrevPreproObsInfo[SatLabel]["PrevCode"] = PreproObs["C1"]
        PrevPreproObsInfo[SatLabel]["PrevPhase"] = PreproObs["L1Meters"]
        PrevPreproObsInfo[SatLabel]["PrevCodeRate"] = PreproObs["CodeRate"]
        PrevPreproObsInfo[SatLabel]["PrevPhaseRate"] = PreproObs["PhaseRate"]
        PrevPreproObsInfo[SatLabel]["PrevRej"] = PreproObs["RejectionCause"]
        PrevPreproObsInfo[SatLabel]["PrevEpoch"] = Epoch

    # End of for SatLabel, PreproObs in PreproObsInfo.items():

    # Loop over satellites
    for SatLabel, PreproObs in PreproObsInfo.items():
        # Compute Iono Mapping Function
        PreproObs["Mpp"] = computeIonoMappingFunction(PreproObs["Elevation"])

        # If the checks were successfully passed
        if PreproObs["Status"] == 1:
            # Build Geometry-Free combination of Phases
            # ----------------------------------------------------------
            # Divide Geometry-Free by 1-GAMMA to obtain STEC
            Stec = PreproObs["GF_L"] / (1 - Const.GPS_GAMMA_L1L2)

            # If valid Previous Geometry-Free Observable
            if PrevPreproObsInfo[SatLabel]["PrevStec"] != Const.NAN:
                # Compute the VTEC Rate
                # ----------------------------------------------------------
                
                # print("DBG STEC:", PreproObs["Sod"], SatLabel, Stec, PrevPreproObsInfo[SatLabel]["PrevStec"], PrevPreproObsInfo[SatLabel]["PrevStecEpoch"])
                
                # Estimate the STEC Gradient
                DeltaStec = \
                    (Stec - PrevPreproObsInfo[SatLabel]["PrevStec"]) / \
                    (PreproObs["Sod"] - PrevPreproObsInfo[SatLabel]["PrevStecEpoch"])

                # Compute VTEC Gradient
                DeltaVtec = DeltaStec / PreproObs["Mpp"]

                # Store DeltaVtec in mm/s
                PreproObs["VtecRate"] = DeltaVtec * 1000

                # Compute Instantaneous Along-Arc-TEC-Rate (AATR)
                # AATR is the delta VTEC weighted with the mapping function
                # ----------------------------------------------------------
                # Compute AATR
                PreproObs["iAATR"] = PreproObs["VtecRate"] / PreproObs["Mpp"]

            # Update previous Geometry-Free Observable
            PrevPreproObsInfo[SatLabel]["PrevStec"] = Stec
            PrevPreproObsInfo[SatLabel]["PrevStecEpoch"] = PreproObs["Sod"]

            # End of if PrevPreproObsInfo[SatLabel]["PrevStec"] != Const.NAN:

        # End of if PreproObs["Status"] == 1:

    # End of for SatLabel, PreproObs in PreproObsInfo.items()
    
    return PreproObsInfo

# End of function runPreProcMeas()

########################################################################
# END OF PREPROCESSING FUNCTIONS MODULE
########################################################################
