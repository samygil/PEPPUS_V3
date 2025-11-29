import numpy as np
from collections import OrderedDict

from Geoid import LLWGS84_SWX,\
LLWGS84_SWY,\
LLWGS84_NWX,\
LLWGS84_NWY,\
LLWGS84_NEX,\
LLWGS84_NEY,\
LLWGS84_SEX,\
LLWGS84_SEY

from Geoid import SIZE_LONG_GEOID_MAT,\
    SIZE_LAT_GEOID_MAT

from Geoid import GEOID_MATRIX

# Misc constants
EPSILON_D = 1.e-12
NOT_A_NUMBER = -999

# altura = computeGeoid - refAltura


def computeGeoidHeight(Lon, Lat):
    # Declare local variables
    NormLon = 0.0
    NormLat = 0.0

    # calculate normalized coordinates from geographic position
    NormLon = (Lon - LLWGS84_SWX)/(LLWGS84_SEX - LLWGS84_SWX)
    NormLat = (Lat - LLWGS84_SWY)/(LLWGS84_NWY - LLWGS84_SWY)   

    # adjust wraparound longitudes
    NormLon = NormLon - np.floor(NormLon)

    # clamp out-of-range latitudes
    if (NormLat < 0.0):
        NormLat = 0.0
    if (NormLat > 1.0):
        NormLat = 1.0
    
    # check normalized coordinates range
    if ((0.0 <= NormLon) and (NormLon <= 1.0) and (0.0 <= NormLat) and (NormLat <= 1.0)):
        NormLon *= SIZE_LONG_GEOID_MAT - 1
        NormLat *= SIZE_LAT_GEOID_MAT - 1
        i = int(NormLon + 0.5)
        j = int(NormLat + 0.5)
        NormLon -= i
        NormLat -= j

        if ((i == (SIZE_LONG_GEOID_MAT - 1)) and (SIZE_LONG_GEOID_MAT > 1)):
            i = SIZE_LONG_GEOID_MAT - 2
            NormLon = 1.0

        # End if ((i == SIZE_LONG_GEOID_MAT - 1) and (SIZE_LONG_GEOID_MAT > 1))

        if ((j == (SIZE_LAT_GEOID_MAT - 1)) and (SIZE_LAT_GEOID_MAT > 1)):
            j = SIZE_LAT_GEOID_MAT - 2
            NormLat = 1.0

        # End if ((j == SIZE_LAT_GEOID_MAT - 1) and (SIZE_LAT_GEOID_MAT > 1))

        # Interpolate to get Geoid Height at the given coordinates
        val = GEOID_MATRIX[(int)(i + (j * SIZE_LONG_GEOID_MAT))]

        if(SIZE_LONG_GEOID_MAT<2):
            return val

        if(abs(val-NOT_A_NUMBER) > EPSILON_D):
            val2 = GEOID_MATRIX[(int)(i + (j * SIZE_LONG_GEOID_MAT) + 1)]

            if(SIZE_LAT_GEOID_MAT<2):
                return ((1.0 - NormLon) * val) + (NormLon * val2)

            if (abs(val2-NOT_A_NUMBER) > EPSILON_D):
                val3 = GEOID_MATRIX[(int)(i + (j * SIZE_LONG_GEOID_MAT) + SIZE_LONG_GEOID_MAT)]
                val4 = GEOID_MATRIX[(int)(i + (j * SIZE_LONG_GEOID_MAT) + SIZE_LONG_GEOID_MAT + 1)] 

                if((abs(val3-NOT_A_NUMBER) > EPSILON_D) and\
                        (abs(val4-NOT_A_NUMBER) > EPSILON_D)):
                    return ((1.0 - NormLat) * (((1.0 - NormLon) * val) + (NormLon * val2))) +\
                        (NormLat * (((1.0 - NormLon) * val3) + (NormLon * val4)))

            # End if(SIZE_LAT_GEOID_MAT<2)

        # End if(SIZE_LONG_GEOID_MAT<2)

    # End if (0.0 <= NormLon and NormLon <= 1.0 and 0.0 <= NormLat and NormLat <= 1.0)

    return NOT_A_NUMBER



