#!/usr/bin/env python

########################################################################
# Kpvt.py:
# This module implements the Kalman Filter based Position, Velocity, and 
# Time (PVT) computation for PEPPUS, using Precise Point Positioning (PPP) 
# methodology with float ambiguities estimation.
#
#   Project:        PEPPUS
#   File:           Kpvt.py
#
#   Author: GNSS Academy
#   Copyright GNSS Academy
########################################################################

import numpy as np
import math
from COMMON import GnssConstants as Const


# ----------------------------------------------------------------------
# INTERNAL FUNCTIONS
# ----------------------------------------------------------------------

def _computeHMatrix(NumStates, NumSats, ValidCorrInfo, RcvrPos):
    """
    Computes the measurement matrix (H matrix) for the Kalman Filter.
    
    The state vector (X) order is:
    [ Rx, Ry, Rz, dtr, Amb_GXX, Amb_GYY, ... ]
    
    The measurement vector (Z) is ordered by satellite.
    """
    
    # We assume 1 measurement (P_IF) per sat
    H = np.zeros((NumSats, NumStates)) 

    # Loop over satellites
    for i in range(NumSats):
        sat_corr = ValidCorrInfo[i]
        
        # Predicted Geometric Range based on current state estimate (RcvrPos is the predicted position)
        dx = sat_corr['SatX'] - RcvrPos[0]
        dy = sat_corr['SatY'] - RcvrPos[1]
        dz = sat_corr['SatZ'] - RcvrPos[2]
        GeomRange = math.sqrt(dx**2 + dy**2 + dz**2)

        # Direction Cosines (partial derivatives w.r.t Rx, Ry, Rz)
        e_x = -dx / GeomRange # -dx/rho
        e_y = -dy / GeomRange # -dy/rho
        e_z = -dz / GeomRange # -dz/rho

        # --- Position States (Rx, Ry, Rz) ---
        H[i, 0] = e_x
        H[i, 1] = e_y
        H[i, 2] = e_z
        
        # --- Receiver Clock State (dtr) ---
        H[i, 3] = 1.0 # Derivative w.r.t to the bias *in meters* (dtr*c) is 1
        
        # Note: Ambiguity states (i > 3) are omitted for this simplified P_IF code measurement.
        # The corresponding H matrix elements remain zero.

    return H

def _computeWMatrix(NumSats, ValidCorrInfo):
    """
    Computes the Measurement Noise Covariance Matrix (W or R matrix).
    W is a diagonal matrix containing the measurement variances (sigmas^2).
    """
    W = np.zeros((NumSats, NumSats))
    
    # Loop over satellites (i is the row/column index)
    for i in range(NumSats):
        # We assume 'SigmaUere' holds the total variance (sigma^2)
        # Using a fallback value (10.0m^2) if the key is missing
        sigma_uere = ValidCorrInfo[i].get('SigmaUere', 10.0) 
        W[i, i] = sigma_uere # Assuming input is already variance (sigma^2)
        
    return W

def _computeZVector(NumSats, ValidCorrInfo, RcvrPos, RcvrClkBias):
    """
    Computes the measurement innovation vector (Z).
    Z = Observed Range - Estimated Range (from current state Xk_1)
    """
    Z = np.zeros((NumSats, 1))

    # Loop over satellites
    for i in range(NumSats):
        sat_corr = ValidCorrInfo[i]

        # Observed Range (CorrCode is the corrected pseudorange, i.e., P_IF)
        ObservedRange = sat_corr.get('CorrCode', Const.NAN) 
        
        # Estimated Geometric Range based on predicted Receiver Position
        dx = sat_corr['SatX'] - RcvrPos[0]
        dy = sat_corr['SatY'] - RcvrPos[1]
        dz = sat_corr['SatZ'] - RcvrPos[2]
        EstimatedRange = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Estimate of Range (R_est) = GeomRange + RcvrClkBias (in meters)
        
        # Innovation (Z) = Observed Range - Estimated Range
        # Z = ObservedRange - (GeomRange + RcvrClkBias)
        Z[i, 0] = ObservedRange - EstimatedRange - RcvrClkBias

    return Z
    
# ----------------------------------------------------------------------
# EXPORTED FUNCTIONS
# ----------------------------------------------------------------------

def initKpvtSolution(Conf, RcvrInfo, Jd):
    """
    Initializes the Kalman Filter state vector (X) and covariance matrix (P).
    
    The state vector size (NumStates) is 4 + NumAmbiguities.
    - [0:2]: ECEF Receiver Position (Rx, Ry, Rz)
    - [3]: Receiver Clock Bias (dtr * C) in meters
    - [4:]: Float Ambiguities 
    """
    
    MAX_SATS = Const.MAX_NUM_SATS_CONSTEL
    NumStates = 4 + MAX_SATS # Max size: 4 fixed states + Max ambiguities

    # 1. Initialize State Vector (Xk_1)
    Xk_1 = np.zeros((NumStates, 1))
    
    # Initial Position (from RcvrInfo - assuming index 8 holds the ECEF coordinates)
    Xk_1[0, 0] = RcvrInfo[8][0] 
    Xk_1[1, 0] = RcvrInfo[8][1] 
    Xk_1[2, 0] = RcvrInfo[8][2] 
    
    
    # Initial Clock Bias (0)
    Xk_1[3, 0] = 0.0 # in meters
    
    # Ambiguities (0) - indices 4 to end
    
    # 2. Initialize Covariance Matrix (Pk_1)
    Pk_1 = np.zeros((NumStates, NumStates))
    
    # Initial Position Variance 
    INIT_POS_SIGMA = 300.0 # m
    for i in range(3):
        Pk_1[i, i] = INIT_POS_SIGMA ** 2 
        
    # Initial Clock Variance (in meters^2)
    INIT_CLK_SIGMA = 1000.0 # m
    Pk_1[3, 3] = INIT_CLK_SIGMA ** 2
    
    # Initial Ambiguity Variance (very large)
    INIT_AMB_SIGMA = 1000.0 # m
    for i in range(4, NumStates):
        Pk_1[i, i] = INIT_AMB_SIGMA ** 2

    # Map of Ambiguity indices (Sat ID -> index in X vector)
    SatAmbiguityInfo = {}
    
    # Note: SatAmbiguityInfo will be populated in the first epoch a satellite is tracked.
    
    return Xk_1, Pk_1, SatAmbiguityInfo


def computeKpvtSolution(Conf, RcvrInfo, CorrInfoList, Xk_1, Pk_1, Doy, SatAmbiguityInfo_1):
    """
    Performs the Kalman Filter computation (Prediction and Update) for one epoch.
    
    Inputs:
        Conf: Configuration parameters.
        RcvrInfo: Receiver static information (dictionary with X, Y, Z).
        CorrInfoList: List of corrected measurements (dictionaries), one per satellite.
        Xk_1: State vector from the previous epoch.
        Pk_1: Covariance matrix from the previous epoch.
        Doy: Day of Year.
        SatAmbiguityInfo_1: Map of Ambiguity indices from the previous epoch.
        
    Outputs:
        PosInfo: Output dictionary with computed PVT solution.
        Xk: Updated state vector.
        Pk: Updated covariance matrix.
        SatAmbiguityInfo: Updated map of Ambiguity indices.
    """
    
    # Deep copy to maintain the state before updates
    SatAmbiguityInfo = SatAmbiguityInfo_1.copy() 

    # --- 1. Filter Valid Measurements ---
    ValidCorrInfo = []
    MIN_NUM_SATS = 4 
    
    for sat_corr in CorrInfoList:
        # Check Flag (Flag=1 for valid, Flag=0 for rejected/no data)
        # Check if the core data (CorrCode and Sat Position) is available
        if (sat_corr.get('Flag', 0) > 0 and 
            sat_corr.get('CorrCode', Const.NAN) != 0.0 and 
            sat_corr.get('CorrCode', Const.NAN) != Const.NAN and
            sat_corr.get('SatX', 0.0) != 0.0): 
            
            ValidCorrInfo.append(sat_corr)
            
    NumSats = len(ValidCorrInfo)
    
    # --- 2. Check for Solution Availability ---
    Sod = ValidCorrInfo[0]['Sod'] if NumSats > 0 else CorrInfoList[0]['Sod']

    if NumSats < MIN_NUM_SATS:
        # Return no solution (Sol=0) and previous state
        
        PosInfo = {
            'Sod': Sod,
            'Doy': Doy,
            'Sol': 0, # Sol Flag = 0 (No solution)
            'X': Const.NAN, 
            'Y': Const.NAN, 
            'Z': Const.NAN,
            'dt': Const.NAN,
            'Lon': Const.NAN, # New: Longitude
            'Lat': Const.NAN, # New: Latitude
            'Alt': Const.NAN, # New: Altitude
            'Clk': Const.NAN, # New: Clock Bias (in seconds)
            'NumSatVis': NumSats,
            'NumSat': 0,      # New: NumSat
            'Hpe': Const.NAN, # New: HPE
            'Vpe': Const.NAN, # New: VPE
            'Epe': Const.NAN, # New: EPE
            'Npe': Const.NAN, # New: NPE
            'PDOP': Const.NAN,
        }
        
        # Return previous state for the next prediction step
        return PosInfo, Xk_1, Pk_1, SatAmbiguityInfo 

    # --- 3. Prediction Step ---
    
    Fk = np.eye(Xk_1.shape[0])
    Qk = np.zeros(Pk_1.shape)
    dt = Conf["SAMPLING_RATE"] 
    
    # Receiver Clock Noise (dtr) - Index 3 (in meters^2/s)
    CLK_Q_PARAM = Conf.get("CLK_Q_PARAM", 1.0e-6) 
    Qk[3, 3] = CLK_Q_PARAM * dt

    
    # Ambiguity Noise (Index 4 onwards) - modeled as static (small random walk)
    AMB_Q_PARAM = 1.0e-12 
    for i in range(4, Xk_1.shape[0]):
        Qk[i, i] = AMB_Q_PARAM * dt
        
    # Predicted State Vector: Xk = Xk-1 (Static model)
    Xk = Fk @ Xk_1
    
    # Predicted Covariance Matrix: Pk = Fk * Pk-1 * Fk^T + Qk
    Pk = Fk @ Pk_1 @ Fk.T + Qk
    
    # --- 4. Update Step ---

    # Predicted Receiver Position and Clock (from Xk)
    RcvrPos = Xk[0:3, 0] # [Rx, Ry, Rz]
    RcvrClkBias = Xk[3, 0]  # dtr * C (in meters)

    # Compute Measurement Innovation Vector (Z)
    Z = _computeZVector(NumSats, ValidCorrInfo, RcvrPos, RcvrClkBias) 
    
    # Compute Design Matrix (H)
    NumStates = Xk.shape[0]
    H = _computeHMatrix(NumStates, NumSats, ValidCorrInfo, RcvrPos)

    # Compute Measurement Noise Covariance Matrix (W/R)
    W = _computeWMatrix(NumSats, ValidCorrInfo)
    
    # V = H * Pk * H^T + W (Innovation Covariance)
    V = H @ Pk @ H.T + W

    # Kalman Gain: K = Pk * H^T * V^-1
    try:
        V_inv = np.linalg.inv(V)
        K = Pk @ H.T @ V_inv
    except np.linalg.LinAlgError:
        print("WARNING KP: Singular matrix in Kalman Update.")
        # Return previous state if update fails
        return PosInfo, Xk_1, Pk_1, SatAmbiguityInfo

    # Update State Vector: Xk = Xk + K * Z
    Xk_update = Xk + K @ Z

    # Update Covariance Matrix: Pk = (I - K * H) * Pk
    I = np.eye(NumStates)
    Pk_update = (I - K @ H) @ Pk
    
    # --- 5. Prepare Output ---
    
    SolFlag = 1

    # 5.1 ECEF to Geodetic (XYZ to LLA) Conversion
    X_ecef = Xk_update[0, 0]
    Y_ecef = Xk_update[1, 0]
    Z_ecef = Xk_update[2, 0]

    # Call the conversion function (Lat and Lon are now in radians)
    Lon, Lat, Alt = convertXyz2Lla(X_ecef, Y_ecef, Z_ecef) 

    # 5.2 Compute Position Error Statistics (HPE, VPE, EPE, NPE)
    # These are placeholders. If your code expects calculated values from Pk_update, 
    # you must add the covariance rotation logic here.
    Hpe, Vpe, Epe, Npe = Const.NAN, Const.NAN, Const.NAN, Const.NAN
    NumSat = NumSats # Total number of satellites used (equal to NumSatVis for this case)

    # 5.3 Compute PDOP
    try:
        Pk_pos = Pk_update[0:3, 0:3] 
        PDOP = math.sqrt(np.trace(Pk_pos)) / 3.0 # Simplified
    except:
        PDOP = Const.NAN

    # Final PosInfo dictionary (NOW WITH ALL REQUIRED KEYS)
    PosInfo = {
        'Sod': Sod,
        'Doy': Doy,
        'Sol': SolFlag, 
        'X': X_ecef, 
        'Y': Y_ecef, 
        'Z': Z_ecef,
        'dt': Xk_update[3, 0] / Const.SPEED_OF_LIGHT, 
        # ADDED FIELDS FOR COMPATIBILITY WITH InputOutput.py
        'Lon': Lon,        # New: Longitude (radians)
        'Lat': Lat,        # New: Latitude (radians)
        'Alt': Alt,        # New: Altitude (meters)
        'Clk': Xk_update[3, 0] / Const.SPEED_OF_LIGHT, # New: Clock Bias in seconds (InputOutput Requirement)
        'NumSatVis': NumSats,
        'NumSat': NumSat,  # New: Number of Satellites (InputOutput Requirement)
        'Hpe': Hpe,        # New: Horizontal Error (Placeholder)
        'Vpe': Vpe,        # New: Vertical Error (Placeholder)
        'Epe': Epe,        # New: East Error (Placeholder)
        'Npe': Npe,        # New: North Error (Placeholder)
        # END OF ADDED FIELDS
        'PDOP': PDOP,
    }
    
    return PosInfo, Xk_update, Pk_update, SatAmbiguityInfo

# Kpvt.py (Add ECEF to LLA function)
def convertXyz2Lla(X, Y, Z):
    """
    Converts ECEF coordinates (X, Y, Z) to Geodetic (Lon, Lat, Alt) [rad, rad, m].
    Uses WGS-84 and an iterative method.
    """
    # Required WGS-84 constants
    a = Const.EARTH_SEMIAXIS     # 6378137.0 m
    f = Const.FLATTENING         # 1.0/298.257223563
    e2 = Const.E2                # f * (2 - f)
    
    # Length (p is the radial distance in the XY plane)
    p = math.sqrt(X**2 + Y**2)
    
    if p < 1.0E-6:
        # Position on the Z-axis (Pole)
        Lon = Const.NAN
        Lat = math.atan2(Z, 0.0) # pi/2 or -pi/2
        Alt = abs(Z) - a * math.sqrt(1.0 - e2)
    else:
        # First estimate of Latitude and Altitude
        Lat = math.atan2(Z, p * (1.0 - e2))
        Alt = 0.0 # Initial value (zero)
        
        # Iteration for the desired precision (5 iterations are usually sufficient)
        for i in range(5):
            sin_lat = math.sin(Lat)
            N = a / math.sqrt(1.0 - e2 * sin_lat**2)
            Alt = p / math.cos(Lat) - N
            Lat_new = math.atan2(Z + N * e2 * sin_lat, p)
            if abs(Lat_new - Lat) < 1e-12: # Check convergence
                Lat = Lat_new
                break
            Lat = Lat_new

        # Longitude
        Lon = math.atan2(Y, X)
        
    return Lon, Lat, Alt


# End of Kpvt.py