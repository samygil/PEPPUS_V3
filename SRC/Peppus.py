#!/usr/bin/env python

########################################################################
# Peppus.py:
# This is the Main Module of PEPPUS tool
#
#   Project:        PEPPUS
#   File:           Peppus.py
#
#   Author: GNSS Academy
#   Copyright GNSS Academy
#
# -----------------------------------------------------------------
# Date       | Author               | Action
# -----------------------------------------------------------------
#
# Usage:
#   Peppus.py $SCEN_PATH
########################################################################

import sys, os

# Update Path to reach COMMON
Common = os.path.dirname(
    os.path.abspath(sys.argv[0])) + '/COMMON'
sys.path.insert(0, Common)

# Import External and Internal functions and Libraries
#----------------------------------------------------------------------
from collections import OrderedDict, defaultdict
from yaml import dump
from COMMON import GnssConstants as Const
from InputOutput import readConf
from InputOutput import processConf
from InputOutput import readRcvr
from InputOutput import createOutputFile
from InputOutput import readObsEpoch
from InputOutput import generatePreproFile
from InputOutput import readSatPos
from InputOutput import readSatClk
from InputOutput import readSatApo
from InputOutput import generateCorrFile
from InputOutput import PreproHdr, CorrHdr
from InputOutput import CSNEPOCHS, CSNPOINTS
from InputOutput import RcvrIdx, ObsIdx
from InputOutput import PosHdr, generatePosFile
from Preprocessing import runPreProcMeas
from PreprocessingPlots import generatePreproPlots
from Corrections import runCorrectMeas
from CorrectionsPlots import generateCorrPlots
from COMMON.Dates import convertJulianDay2YearMonthDay
from COMMON.Dates import convertYearMonthDay2Doy
from Kpvt import initKpvtSolution, computeKpvtSolution

#----------------------------------------------------------------------
# INTERNAL FUNCTIONS
#----------------------------------------------------------------------

def displayUsage():
    sys.stderr.write("ERROR: Please provide path to SCENARIO as a unique argument\n")

#######################################################
# MAIN BODY
#######################################################

# Check Input/Output Arguments
if len(sys.argv) != 2:
    displayUsage()
    sys.exit()

# Extract the arguments
Scen = sys.argv[1]

# Select the Configuratiun file name
CfgFile = Scen + '/CFG/peppus.cfg'

# Read conf file
Conf = readConf(CfgFile)
# print(dump(Conf))

# Process Configuration Parameters
Conf = processConf(Conf)

# Select the RCVR Positions file name
RcvrFile = Scen + '/INP/RCVR/' + Conf["RCVR_FILE"]

# Read RCVR Positions file
RcvrInfo = readRcvr(RcvrFile)

# Print header
print( '------------------------------------')
print( '--> RUNNING PEPPUS:')
print( '------------------------------------')

# Loop over RCVRs
#-----------------------------------------------------------------------
for Rcvr in RcvrInfo.keys():
    # Display Message
    print( '\n***-----------------------------***')
    print( '*** Processing receiver: ' + Rcvr + '   ***')
    print( '***-----------------------------***')


# >>> T2.1: Initializing Kalman Filter State and Covariance
    print(f"DEBUG: Rcvr: {Rcvr}, Type: {type(RcvrInfo[Rcvr])}, Content: {RcvrInfo[Rcvr]}")
    Xk_1, Pk_1, SatAmbiguityInfo = initKpvtSolution(Conf, RcvrInfo[Rcvr], Conf["INI_DATE_JD"])
    
    # CORRECTION: Retrieve the initial year from Conf since 'Year' is not defined here yet.
    IniYear = Conf["INI_YEAR"]

    # If PVT output is enabled, create the output file
    if Conf["PVT_OUT"] == 1:
        # Define the full path and name to the output PVT file
        PvtFile = Scen + \
            '/OUT/PVT/' + "PVT_%s_Y%02d.dat" % \
                (Rcvr, IniYear % 100) # Use IniYear
        # Create output file
        fpvt = createOutputFile(PvtFile, PosHdr)

    # Loop over Julian Days in simulation
    #-----------------------------------------------------------------------
    for Jd in range(Conf["INI_DATE_JD"], Conf["END_DATE_JD"] + 1):
        # Compute Year, Month and Day in order to build input file name
        Year, Month, Day = convertJulianDay2YearMonthDay(Jd)

        # Compute the Day of Year (DoY)
        Doy = convertYearMonthDay2Doy(Year, Month, Day)

        # Display Message
        print( '\n*** Processing Day of Year: ' + str(Doy) + ' ... ***')

        # Define the full path and name to the OBS INFO file to read
        ObsFile = Scen + \
            '/INP/OBS/' + "OBS_%s_Y%02dD%03d.dat" % \
                (Rcvr, Year % 100, Doy)

        # Display Message
        print("INFO: Reading file: %s..." %
        ObsFile)

        # Define the full path and name to the SATPOS file to read and open the file
        SatPosFile = Scen + \
            '/OUT/SAT/' + "SATPOS_Y%02dD%03d.dat" % \
                (Year % 100, Doy)
        SatPosInfo = readSatPos(SatPosFile)

        # Define the full path and name to the SATCLK file to read and open the file
        SatClkFile = Scen + \
            '/OUT/SAT/' + "SATCLK_Y%02dD%03d_30s.dat" % \
                (Year % 100, Doy)
        SatClkInfo = readSatClk(SatClkFile)
        
        # Define the full path and name to the SATAPO file to read and open the file
        SatApoFile = Scen + \
            '/OUT/SAT/' + Conf["SATAPO_FILE"]
        SatApoInfo = readSatApo(SatApoFile)


        ## OUTPUTS
        # If Preprocessing outputs are activated
        if Conf["PREPRO_OUT"] == 1:
            # Define the full path and name to the output PREPRO OBS file
            PreproObsFile = Scen + \
                '/OUT/PPVE/' + "PREPRO_OBS_%s_Y%02dD%03d.dat" % \
                    (Rcvr, Year % 100, Doy)

            # Create output file
            fpreprobs = createOutputFile(PreproObsFile, PreproHdr)

        # If Corrected outputs are activated
        if Conf["PCOR_OUT"] == 1:
            # Define the full path and name to the output PCOR file
            CorrFile = Scen + \
                '/OUT/PCOR/' + "PCOR_%s_Y%02dD%03d.dat" % \
                    (Rcvr, Year % 100, Doy)

            # Create output file
            fcorr = createOutputFile(CorrFile, CorrHdr)

        # Initialize Variables
        EndOfFile = False
        ObsInfo = [None]
        PrevPreproObsInfo = {}
        for prn in range(1, Const.MAX_NUM_SATS_CONSTEL + 1):
            PrevPreproObsInfo["G%02d" % prn] = {
                "PrevEpoch": 86400,                                             # Previous SoD with measurements
                "PrevL1": 0.0,                                                  # Previous L1
                "PrevL2": 0.0,                                                  # Previous L2
                "PrevC1": 0.0,                                                  # Previous Smoothed C1
                "PrevP2": 0.0,                                                  # Previous Smoothed C2
                "PrevRej": 1,                                                   # Previous Rejection flag
                
                "CycleSlipBuffIdx": 0,                                          # Index of CS buffer
                "CycleSlipFlagIdx": 0,                                          # Index of CS flag array
                "GF_L_Prev": [0.0] * int(Conf["CYCLE_SLIPS"][CSNPOINTS]),       # Array with previous GF carrier phase observables
                "GF_Epoch_Prev": [0.0] * int(Conf["CYCLE_SLIPS"][CSNPOINTS]),   # Array with previous epochs
                "CycleSlipFlags": [0.0] * int(Conf["CYCLE_SLIPS"][CSNEPOCHS]),  # Array with last cycle slips flags
                
                "PrevCode": Const.NAN,                                          # Previous Code
                "PrevPhase": Const.NAN,                                         # Previous Phase
                "PrevCodeRate": Const.NAN,                                      # Previous Code Rate
                "PrevPhaseRate": Const.NAN,                                     # Previous Phase Rate
                "PrevStec": Const.NAN,                                          # Previous STEC
                "PrevStecEpoch": Const.NAN,                                     # Previous STEC epoch
                
                "ResetAmb": 1,                                                  # Reset Ambiguities flag
            } # End of SatPreproObsInfo

        SatComPos_1 = {}
        Sod_1 = {}
        EpochsPerSat = defaultdict(list)

        # Open OBS file
        with open(ObsFile, 'r') as fobs:
            # Read header line of OBS file
            fobs.readline()
            
            # LOOP over all Epochs of OBS file
            # ----------------------------------------------------------
            while not EndOfFile:

                # If ObsInfo is not empty
                if ObsInfo != []:

                    # Read Only One Epoch
                    ObsInfo = readObsEpoch(fobs)

                    # If ObsInfo is empty, exit loop
                    if ObsInfo == []:
                        break

                        ##########################
                    # Preprocess OBS measurements
                    # ----------------------------------------------------------
                    PreproObsInfo = runPreProcMeas(Conf, RcvrInfo[Rcvr], ObsInfo, PrevPreproObsInfo)

                    # If PREPRO outputs are requested
                    if Conf["PREPRO_OUT"] == 1:
                        # Generate output file
                        generatePreproFile(fpreprobs, PreproObsInfo)

                    # Get SoD
                    Sod = int(float(ObsInfo[0][ObsIdx["SOD"]]))

                    # The rest of te analyses are executed every configured sampling rate
                    if(Sod % Conf["SAMPLING_RATE"] == 0):

                        for sat, obs in PreproObsInfo.items():
                            EpochsPerSat[sat].append(Sod)

                        # Correct measurements and estimate the variances with SBAS information
                        # ---------------------------------------------------------
                        CorrInfo = runCorrectMeas(Conf, RcvrInfo[Rcvr], ObsInfo, PreproObsInfo, 
                        SatPosInfo, SatClkInfo, SatApoInfo, SatComPos_1, Sod_1, EpochsPerSat)
                        
                        # If PCOR outputs are requested
                        if Conf["PCOR_OUT"] == 1:
                            # Generate output file
                            generateCorrFile(fcorr, CorrInfo)

                        ###########################################################################
                        # KALMAN PVT COMPUTATION
   
                        # Fix 1: Converte o dicionário CorrInfo (keys: 'GXX') em uma lista de dicionários
                        CorrInfoList = list(CorrInfo.values())

                        # Fix 2: Ajusta o print de debug para usar a lista convertida
                        print(f"DEBUG PE: Antes do KPVT, CorrInfo size: {len(CorrInfoList)}")
                        if CorrInfoList:
                            print(f"DEBUG PE: Primeiro item do CorrInfo: {CorrInfoList[0]}")


                        PosInfo, Xk, Pk, SatAmbiguityInfo = computeKpvtSolution(Conf, RcvrInfo[Rcvr], CorrInfoList, Xk_1, Pk_1, Doy, SatAmbiguityInfo)



                        # Status and covariance for next epoch

                        Xk_1 = Xk
                        Pk_1 = Pk
                        SatAmbiguityInfo_1 = SatAmbiguityInfo
                        
                        # KALMAN PVT COMPUTATION
                        
                        # If PVT output are requested
                        if Conf["PVT_OUT"] == 1:
                            # Generate output file
                            generatePosFile(fpvt, PosInfo)
                        
                # End of if ObsInfo != []:

                else:
                    EndOfFile = True

                # End of if ObsInfo != []:

            # End of while not EndOfFile:

        # End of with open(ObsFile, 'r') as f:

        # If PREPRO outputs are requested
        if Conf["PREPRO_OUT"] == 1:
            # Close PREPRO output file
            fpreprobs.close()

            # Display Message
            print("INFO: Reading file: %s and generating PREPRO figures..." %
            PreproObsFile)

            # Generate Preprocessing plots
            generatePreproPlots(PreproObsFile)

        # If PCOR outputs are requested
        if Conf["PCOR_OUT"] == 1:
            # Close PCOR output file
            fcorr.close()
            
            # Display Message
            print("INFO: Reading file: %s and generating PCOR figures..." %
            CorrFile)
            
            # Generate PCOR plots
            generateCorrPlots(CorrFile, RcvrInfo[Rcvr])

    # End of Julian Day loop

    if Conf["PVT_OUT"] == 1:
        # Close PVT output file
        fpvt.close()

# End of RCVR loop

print( '--> END OF PEPPUS ANALYSIS')
print( '------------------------------------')

#######################################################
# End of Peppus.py
#######################################################