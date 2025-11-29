import math
import numpy as np

# Ref.: ESA_GNSS-Book_TM-23_Vol_I.pdf Section B.1.2 (Appendix B)
def xyz2llh(x,y,z):
    """
    Converts ECEF Cartesian coordinates (X, Y, Z) to 
    geodetic Latitude, Longitude, Height (LLH) (WGS84).
    Returns (Longitude [deg], Latitude [deg], Height [m]).
    """
    # --- WGS84 constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    # --- derived constants
    b = a - f*a
    e = math.sqrt(math.pow(a,2.0)-math.pow(b,2.0))/a
    clambda = math.atan2(y,x)
    p = math.sqrt(pow(x,2.0)+pow(y,2))
    h_old = 0.0
    # first guess with h=0 meters
    theta = math.atan2(z,p*(1.0-math.pow(e,2.0)))
    cs = math.cos(theta)
    sn = math.sin(theta)
    N = math.pow(a,2.0)/math.sqrt(math.pow(a*cs,2.0)+math.pow(b*sn,2.0))
    h = p/cs - N
    # Iterative solution
    while abs(h-h_old) > 1.0e-6:
        h_old = h
        theta = math.atan2(z,p*(1.0-math.pow(e,2.0)*N/(N+h)))
        cs = math.cos(theta)
        sn = math.sin(theta)
        N = math.pow(a,2.0)/math.sqrt(math.pow(a*cs,2.0)+math.pow(b*sn,2.0))
        h = p/cs - N
    Rad2Deg = 180.0 / math.pi
    
    # theta is Latitude, clambda is Longitude
    return clambda * Rad2Deg, theta * Rad2Deg, h

# Ref.: ESA_GNSS-Book_TM-23_Vol_I.pdf Section B.1.1 (Appendix B)
def llh2xyz(lon,lat,h):
    """
    Converts geodetic Latitude [deg], Longitude [deg], Height [m] (WGS84) to 
    ECEF Cartesian coordinates (X, Y, Z).
    Returns (X [m], Y [m], Z [m]).
    """
    # WGS84 derived constant: e^2 (First eccentricity squared)
    e2 = 0.0066943799901 
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Radius of Curvature in the Prime Vertical (N)
    N = (6378137.0 / math.sqrt(1 - e2 * (math.sin(lat_rad)**2)))

    X = (N+h)*(math.cos(lat_rad)*math.cos(lon_rad))
    Y = (N+h)*(math.cos(lat_rad)*math.sin(lon_rad))
    Z = ((1-e2)*N + h)*(math.sin(lat_rad)) 

    return X,Y,Z


def enu_rotation_matrix(lat, lon):
    """
    Calculates the rotation matrix R that transforms an ECEF vector (dX, dY, dZ) 
    to the Local Tangent Frame (ENU - East, North, Up) at a specific LLH origin point.

    Parameters:
    lat (float): Geodetic Latitude of the origin point (degrees).
    lon (float): Geodetic Longitude of the origin point (degrees).
    
    Returns:
    numpy.ndarray: The 3x3 rotation matrix.
    """
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    # Rotation Matrix R from ECEF to ENU
    # Rows: (East), (North), (Up)
    R = np.array([
        [-sin_lon,           cos_lon,           0.0      ], # East
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat ], # North
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat ]  # Up
    ])
    
    return R

def xyz2enu(delta_x, delta_y, delta_z, lat, lon):
    """
    Converts a difference vector (dX, dY, dZ) from the ECEF system to 
    the Local Tangent Frame (ENU - East, North, Up) system.
    
    Parameters:
    delta_x, delta_y, delta_z (float): ECEF difference vector (e.g., SatPos - RcvPos).
    lat (float): Geodetic Latitude of the origin point (degrees).
    lon (float): Geodetic Longitude of the origin point (degrees).
    
    Returns:
    tuple: (East, North, Up) - Coordinates in the local system (floats).
    """
    
    # 1. Calculate the Rotation Matrix
    R = enu_rotation_matrix(lat, lon)
    
    # 2. Assemble the ECEF difference vector
    delta_xyz = np.array([delta_x, delta_y, delta_z])
    
    # 3. Matrix multiplication: ENU = R * dXYZ
    enu = R @ delta_xyz
    
    return enu[0], enu[1], enu[2]