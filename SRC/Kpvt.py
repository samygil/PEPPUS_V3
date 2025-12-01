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
from collections import OrderedDict
from COMMON import GnssConstants as Const
from COMMON.Coordinates import xyz2llh

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
        
        # 1. Predicted Geometric Range (rho) based on current state estimate (RcvrPos)
        dx = sat_corr['SatX'] - RcvrPos[0]
        dy = sat_corr['SatY'] - RcvrPos[1]
        dz = sat_corr['SatZ'] - RcvrPos[2]
        GeomRange = math.sqrt(dx**2 + dy**2 + dz**2)

        # 2. Direction Cosines (partial derivatives w.r.t Rx, Ry, Rz)
        e_x = -dx / GeomRange # -dx/rho
        e_y = -dy / GeomRange # -dy/rho
        e_z = -dz / GeomRange # -dz/rho

        # --- Position States (Rx, Ry, Rz) ---
        H[i, 0] = e_x
        H[i, 1] = e_y
        H[i, 2] = e_z
        
        # --- Receiver Clock State (dtr) ---
        # The state is the clock bias *in meters* (dtr*c), so the derivative w.r.t it is 1.0
        H[i, 3] = 1.0 
        
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
        # We assume 'SigmaUere' holds the total measurement variance (sigma^2)
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

        # 1. Observed Range (CorrCode is the corrected pseudorange, i.e., P_IF)
        ObservedRange = sat_corr.get('CorrCode', Const.NAN) 
        
        # 2. Estimated Geometric Range (rho) based on predicted Receiver Position
        dx = sat_corr['SatX'] - RcvrPos[0]
        dy = sat_corr['SatY'] - RcvrPos[1]
        dz = sat_corr['SatZ'] - RcvrPos[2]
        EstimatedRange = math.sqrt(dx**2 + dy**2 + dz**2) # Geometric Range (rho)
        
        # 3. Estimated Total Range (R_est) = Geometric Range (rho) + Receiver Clock Bias (dtr * C)
        
        # 4. Innovation (Z) = Observed Range (P_IF) - Estimated Total Range (R_est)
        # Z = ObservedRange - (EstimatedRange + RcvrClkBias)
        Z[i, 0] = ObservedRange - EstimatedRange - RcvrClkBias

    return Z

def _computeRotMatrixEcef2Enu(Lat, Lon):
    """
    Computes the rotation matrix R from ECEF frame to Local Tangent Plane (ENU).
    R is 3x3. Lat/Lon must be in radians.
    """
    sin_lat = math.sin(Lat)
    cos_lat = math.cos(Lat)
    sin_lon = math.sin(Lon)
    cos_lon = math.cos(Lon)

    # R is the matrix (e, n, u)^T where e, n, u are in ECEF frame
    # e = East, n = North, u = Up
    R = np.array([
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    return R

def _computeDops(H, Lat, Lon):
    """
    Computes Dilution of Precision (DOP) values from the Geometry Matrix H.
    H is the Design Matrix (NumSats x NumStates).
    Lat/Lon must be in radians.
    """
    # Extract the geometry part of H (first 4 columns: x, y, z, dt)
    # We only need the geometry for DOP calculation
    G = H[:, 0:4]
    
    try:
        # Compute the Cofactor Matrix Q = (G^T * G)^-1
        Q = np.linalg.inv(G.T @ G)
        
        # Extract elements in ECEF frame
        Qxx, Qyy, Qzz, Qtt = Q[0, 0], Q[1, 1], Q[2, 2], Q[3, 3]
        
        # GDOP (Geometric DOP) = sqrt(Qxx + Qyy + Qzz + Qtt)
        GDOP = math.sqrt(Qxx + Qyy + Qzz + Qtt)
        
        # PDOP (Position DOP) = sqrt(Qxx + Qyy + Qzz)
        PDOP = math.sqrt(Qxx + Qyy + Qzz)
        
        # TDOP (Time DOP) = sqrt(Qtt)
        TDOP = math.sqrt(Qtt)
        
        # To compute HDOP and VDOP, we need to rotate the cofactor matrix to ENU
        # Extract 3x3 position block
        Q_xyz = Q[0:3, 0:3]
        
        # Compute Rotation Matrix ECEF -> ENU
        R = _computeRotMatrixEcef2Enu(Lat, Lon)
        
        # Rotate covariance: Q_enu = R * Q_xyz * R^T
        Q_enu = R @ Q_xyz @ R.T
        
        Qee = Q_enu[0, 0] # East
        Qnn = Q_enu[1, 1] # North
        Quu = Q_enu[2, 2] # Up
        
        # HDOP (Horizontal DOP) = sqrt(Qee + Qnn)
        HDOP = math.sqrt(Qee + Qnn)
        
        # VDOP (Vertical DOP) = sqrt(Quu)
        VDOP = math.sqrt(Quu)
        
        return GDOP, PDOP, HDOP, VDOP, TDOP
        
    except np.linalg.LinAlgError:
        # Return NaNs if matrix inversion fails (e.g., bad geometry)
        return Const.NAN, Const.NAN, Const.NAN, Const.NAN, Const.NAN
    
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
    
    # Initial Position (from RcvrInfo)
    # CORRECTION: RcvrInfo[8] contains the tuple (X, Y, Z) in ECEF
    # We access the tuple elements [0], [1], [2]
    Xk_1[0, 0] = RcvrInfo[8][0] + 3.0 
    Xk_1[1, 0] = RcvrInfo[8][1] + 3.0 
    Xk_1[2, 0] = RcvrInfo[8][2] + 3.0 
    
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
        RcvrInfo: Receiver static information (list with ID, X, Y, Z).
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
    
    # Access required keys for PosInfo output (from InputOutput)
    # These keys are needed for the output structure in InputOutput.py
    # Note: RcvrInfo is a list, index 0 is the ID (e.g., 'TLSA')
    RcvrID = RcvrInfo[0]

    for sat_corr in CorrInfoList:
        # Check if Flag > 0 (valid) and core data (CorrCode and Sat Position) is available
        if (sat_corr.get('Flag', 0) > 0 and 
            sat_corr.get('CorrCode', Const.NAN) != 0.0 and 
            sat_corr.get('CorrCode', Const.NAN) != Const.NAN and
            sat_corr.get('SatX', 0.0) != 0.0): 
            
            ValidCorrInfo.append(sat_corr)
            
    NumSats = len(ValidCorrInfo)
    
    # --- 2. Check for Solution Availability ---
    # Use Sod from the first valid satellite, or from the raw list if none valid
    Sod = ValidCorrInfo[0]['Sod'] if NumSats > 0 else CorrInfoList[0]['Sod']

    if NumSats < MIN_NUM_SATS:
        # Return no solution (Sol=0) and previous state
        
        PosInfo = {
            'Sod': Sod,
            'Doy': Doy,
            'Rcvr': RcvrID,
            'Sol': 0, # Sol Flag = 0 (No solution)
            'X': Const.NAN, 
            'Y': Const.NAN, 
            'Z': Const.NAN,
            'dt': Const.NAN,
            'Lon': Const.NAN, 
            'Lat': Const.NAN, 
            'Alt': Const.NAN, 
            'Clk': Const.NAN, 
            'NumSatVis': NumSats,
            'NumSat': 0,      
            'Hpe': Const.NAN, 
            'Vpe': Const.NAN, 
            'Epe': Const.NAN, 
            'Npe': Const.NAN, 
            'PDOP': Const.NAN,
            'HDOP': Const.NAN,
            'VDOP': Const.NAN,
            'TDOP': Const.NAN,
        }
        
        # Return previous state for the next prediction step
        return PosInfo, Xk_1, Pk_1, SatAmbiguityInfo 

    # --- 3. Prediction Step (Time Update) ---
    
    Fk = np.eye(Xk_1.shape[0])
    Qk = np.zeros(Pk_1.shape)
    dt = Conf["SAMPLING_RATE"] 
    
    # Receiver Clock Noise (dtr) - Index 3 (in meters^2/s)
    # Use a default value if CLK_Q_PARAM is missing in configuration
    CLK_Q_PARAM = Conf.get("CLK_Q_PARAM", 1.0e-6) 
    Qk[3, 3] = CLK_Q_PARAM * dt

    # Ambiguity Noise (Index 4 onwards) - modeled as static (small random walk)
    AMB_Q_PARAM = 1.0e-12 
    for i in range(4, Xk_1.shape[0]):
        Qk[i, i] = AMB_Q_PARAM * dt
        
    # Predicted State Vector: Xk = Fk * Xk-1 (Static model: Fk is Identity)
    Xk = Fk @ Xk_1
    
    # Predicted Covariance Matrix: Pk = Fk * Pk-1 * Fk^T + Qk
    Pk = Fk @ Pk_1 @ Fk.T + Qk
    
    # --- 4. Update Step (Measurement Update) ---

    # Predicted Receiver Position and Clock (from Xk)
    RcvrPos = Xk[0:3, 0]   # [Rx, Ry, Rz]
    RcvrClkBias = Xk[3, 0] # dtr * C (in meters)

    # 4.1 Compute Measurement Innovation Vector (Z)
    Z = _computeZVector(NumSats, ValidCorrInfo, RcvrPos, RcvrClkBias) 
    
    # 4.2 Compute Design Matrix (H)
    NumStates = Xk.shape[0]
    H = _computeHMatrix(NumStates, NumSats, ValidCorrInfo, RcvrPos)

    # 4.3 Compute Measurement Noise Covariance Matrix (W/R)
    W = _computeWMatrix(NumSats, ValidCorrInfo)
    
    # 4.4 Compute Innovation Covariance (V)
    # V = H * Pk * H^T + W 
    V = H @ Pk @ H.T + W

    # 4.5 Compute Kalman Gain (K)
    # K = Pk * H^T * V^-1
    try:
        V_inv = np.linalg.inv(V)
        K = Pk @ H.T @ V_inv
    except np.linalg.LinAlgError:
        print("WARNING KP: Singular matrix in Kalman Update.")
        # Return previous state if update fails
        return PosInfo, Xk_1, Pk_1, SatAmbiguityInfo

    # 4.6 Update State Vector (Xk_update)
    # Xk_update = Xk + K * Z
    Xk_update = Xk + K @ Z

    # 4.7 Update Covariance Matrix (Pk_update)
    # Pk_update = (I - K * H) * Pk
    I = np.eye(NumStates)
    Pk_update = (I - K @ H) @ Pk
    
    # --- 5. Prepare Output ---
    
    SolFlag = 1

    # 5.1 ECEF to Geodetic (XYZ to LLA) Conversion
    X_ecef = Xk_update[0, 0]
    Y_ecef = Xk_update[1, 0]
    Z_ecef = Xk_update[2, 0]

    # Use the standard ECEF to LLA conversion from Coordinates.py:
    # xyz2llh returns (Lon_deg, Lat_deg, Alt_m - Ellipsoidal Height)
    Lon_deg, Lat_deg, Alt = xyz2llh(X_ecef, Y_ecef, Z_ecef)
    
    # Convert Lon and Lat to Radians for further processing (e.g., rotation matrix)
    Lon_rad = math.radians(Lon_deg)
    Lat_rad = math.radians(Lat_deg)

    # ----------------------------------------------------------------------
    # NO Geoidal Correction applied here.
    # The reference value seems to be Ellipsoidal Height (h).
    # ----------------------------------------------------------------------

    # 5.3 Covariance Rotation (ECEF to ENU/Local) and DOP/PE Calculation

    # 5.3.1 Extract the 3x3 position covariance block in ECEF
    # Pk_update contains ECEF [X, Y, Z, Clk]
    P_ecef_pos = Pk_update[0:3, 0:3]
    
    # 5.3.2 Compute ECEF-to-ENU Rotation Matrix (based on LLA in Radians)
    R_enu_ecef = _computeRotMatrixEcef2Enu(Lat_rad, Lon_rad)
    
    # 5.3.3 Rotate ECEF position covariance to ENU (Local) frame
    # P_ENU = R * P_ECEF * R^T
    P_enu_pos = R_enu_ecef @ P_ecef_pos @ R_enu_ecef.T
    
    # 5.3.4 Extract variances and calculate Position Errors (PEs)
    P_EE = P_enu_pos[0, 0] # East variance
    P_NN = P_enu_pos[1, 1] # North variance
    P_UU = P_enu_pos[2, 2] # Up variance
    P_CC = Pk_update[3, 3] # Clock variance (in meters^2)

    # Position Errors (PEs) - RMS values in meters (Standard Deviations)
    Epe = math.sqrt(abs(P_EE))
    Npe = math.sqrt(abs(P_NN))
    Vpe = math.sqrt(abs(P_UU))
    Hpe = math.sqrt(P_EE + P_NN) # Horizontal Position Error (2D RMS)

    # 5.3.5 Calculate DOPs (Dilution of Precision) from Geometry
    # We compute DOPs using the Design Matrix H and the Rotation Matrix
    # This ensures HDOP != HPE
    GDOP, PDOP, HDOP, VDOP, TDOP = _computeDops(H, Lat_rad, Lon_rad)
    
    NumSat = NumSats # Total number of satellites used

    # 5.4 Final PosInfo dictionary 
    PosInfo = {
        'Sod': Sod,
        'Doy': Doy,
        'Rcvr': RcvrID,
        'Sol': SolFlag, 
        'X': X_ecef, 
        'Y': Y_ecef, 
        'Z': Z_ecef,
        'dt': Xk_update[3, 0] / Const.SPEED_OF_LIGHT, 
        'Lon': Lon_deg, # Longitude (degrees)
        'Lat': Lat_deg, # Latitude (degrees)
        'Alt': Alt,     # Altitude (meters) - Ellipsoidal Height (h)
        # Clock Bias in METERS
        'Clk': Xk_update[3, 0], 
        'NumSatVis': NumSats,
        'NumSat': NumSat, 
        'Hpe': Hpe, # Horizontal Error (meters)
        'Vpe': Vpe, # Vertical Error (meters)
        'Epe': Epe, # East Error (meters)
        'Npe': Npe, # North Error (meters)
        # DOP FIELDS
        'PDOP': PDOP,
        'HDOP': HDOP,
        'VDOP': VDOP,
        'TDOP': TDOP,
    }
    
    return PosInfo, Xk_update, Pk_update, SatAmbiguityInfo

# End of Kpvt.py