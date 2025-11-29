#!/usr/bin/env python

########################################################################
# PEPPUS/SRC/CorrectionsPlots.py:
# This is the CorrectionsPlots Module of PEPPUS tool
#
#  Project:        PEPPUS
#  File:           CorrectionsPlots.py
#
#   Author: GNSS Academy
#   Copyright GNSS Academy
#
# -----------------------------------------------------------------
# Date       | Author             | Action
# -----------------------------------------------------------------
#
########################################################################

import sys, os
from pandas import unique
from pandas import read_csv
from InputOutput import PreproIdx
from InputOutput import REJECTION_CAUSE_DESC
sys.path.append(os.getcwd() + '/' + \
    os.path.dirname(sys.argv[0]) + '/' + 'COMMON')
from COMMON import GnssConstants
from COMMON.Plots import generatePlot, drawMap
import numpy as np
from collections import OrderedDict
from COMMON import GnssConstants as Const
from COMMON.Coordinates import xyz2llh 
from InputOutput import CorrIdx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re

def plotSatTracks(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (20, 10)

    PlotConf["Title"] = f"Satellite Tracks during visibility periods from {RcvrLabel} on Year {Year} DoY {Doy}"

    PlotConf["LonMin"] = -180
    PlotConf["LonMax"] = 180
    PlotConf["LatMin"] = -90
    PlotConf["LatMax"] = 90

    PlotConf["LonStep"] = 30
    PlotConf["LatStep"] = 15

    PlotConf["xTicks"] = range(PlotConf["LonMin"], PlotConf["LonMax"] + 1, PlotConf["LonStep"])
    PlotConf["yTicks"] = range(PlotConf["LatMin"], PlotConf["LatMax"] + 1, PlotConf["LatStep"])

    PlotConf["xLim"] = [PlotConf["LonMin"], PlotConf["LonMax"]]
    PlotConf["yLim"] = [PlotConf["LatMin"], PlotConf["LatMax"]]

    PlotConf["xTicksLabels"] = [f"{abs(l)}°{'W' if l < 0 else ('E' if l > 0 else '')}" for l in PlotConf["xTicks"]]
    PlotConf["yTicksLabels"] = [f"{abs(l)}°{'S' if l < 0 else ('N' if l > 0 else '')}" for l in PlotConf["yTicks"]]

    PlotConf["xLabel"] = "Longitude [deg]"
    PlotConf["yLabel"] = "Latitude [deg]"

    PlotConf["Grid"] = True
    PlotConf["Map"] = True
    PlotConf["Marker"] = '.'
    PlotConf["LineWidth"] = 1.5

    # Configure color bar for elevation
    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "Elevation [deg]"
    PlotConf["ColorBarMin"] = 0.
    PlotConf["ColorBarMax"] = 90.

    # Extract satellite ECEF coordinates and elevation from corrected data
    X = CorrData["SAT-X"].to_numpy()
    Y = CorrData["SAT-Y"].to_numpy()
    Z = CorrData["SAT-Z"].to_numpy()
    ELEV = CorrData["ELEV"].to_numpy()

    # Convert satellite ECEF coordinates to geodetic (longitude, latitude)
    DataLen = len(X)
    Longitude = np.zeros(DataLen)
    Latitude = np.zeros(DataLen)

    for i in range(DataLen):
        if X[i] == 0.0 and Y[i] == 0.0 and Z[i] == 0.0:
            Longitude[i] = np.nan
            Latitude[i] = np.nan
        else:
            lon, lat, _ = xyz2llh(X[i], Y[i], Z[i])
            Longitude[i] = lon
            Latitude[i] = lat

    # Assign converted coordinates and elevation to plot configuration
    PlotConf["xData"] = {0: Longitude}
    PlotConf["yData"] = {0: Latitude}
    PlotConf["zData"] = {0: ELEV}

    # Path for saving the plot
    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/MONSAT_TRACKS_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")

    


def plotToFvsTime(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Time of Flight (TOF) from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["yLabel"] = "ToF [ms]"
    PlotConf["xLabel"] = f"Hour of DoY {Doy:03d}"

    PlotConf["xTicks"] = range(0, 25)
    PlotConf["xLim"] = [0, 24]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = '|'
    PlotConf["LineWidth"] = 1

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "Elevation [deg]"
    PlotConf["ColorBarMin"] = 0.
    PlotConf["ColorBarMax"] = 90.

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    for prn in sorted(unique(CorrData["PRN"])):
        Label = f"G{int(prn):02d}"
        FilterCond = (CorrData["PRN"] == prn) & (CorrData["FLAG"] == 1)

        x_vals = CorrData["#SOD"][FilterCond] / GnssConstants.S_IN_H
        y_vals = CorrData["TOF"][FilterCond]
        z_vals = CorrData["ELEV"][FilterCond]

        if len(x_vals) == 0 or len(y_vals) == 0 or len(z_vals) == 0:
            # print(f" No valid data for PRN {Label}")
            continue 

        PlotConf["xData"][Label] = x_vals
        PlotConf["yData"][Label] = y_vals
        PlotConf["zData"][Label] = z_vals

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/FLIGHT-TIME_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")

def plotDTRvsTime(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Relativistic Correction (DTR) from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["yLabel"] = "DTR [m]"
    PlotConf["xLabel"] = f"Hour of DoY {Doy:03d}"

    PlotConf["xTicks"] = range(0, 25)
    PlotConf["xLim"] = [0, 24]
    PlotConf["yLim"] = [-20, 15]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = '|'
    PlotConf["LineWidth"] = 1

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "Elevation [deg]"
    PlotConf["ColorBarMin"] = 0.
    PlotConf["ColorBarMax"] = 90.

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    for prn in sorted(unique(CorrData["PRN"])):
        Label = f"G{int(prn):02d}"
        FilterCond = (CorrData["PRN"] == prn) & (CorrData["FLAG"] == 1)

        PlotConf["xData"][Label] = CorrData["#SOD"][FilterCond] / GnssConstants.S_IN_H
        PlotConf["yData"][Label] = CorrData["DTR"][FilterCond]
        PlotConf["zData"][Label] = CorrData["ELEV"][FilterCond]

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/DTR_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")

def plotStropoVsTime(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Slant Tropospheric Delay from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["yLabel"] = "Slant Tropo Delay [m]"
    PlotConf["xLabel"] = f"Hour of DoY {Doy:03d}"

    PlotConf["xTicks"] = range(0, 25)
    PlotConf["xLim"] = [0, 24]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = '|'
    PlotConf["LineWidth"] = 1

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "Elevation [deg]"
    PlotConf["ColorBarMin"] = 0.
    PlotConf["ColorBarMax"] = 90.

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    for prn in sorted(unique(CorrData["PRN"])):
        Label = f"G{int(prn):02d}"
        FilterCond = (CorrData["PRN"] == prn) & (CorrData["FLAG"] == 1)

        PlotConf["xData"][Label] = CorrData["#SOD"][FilterCond] / GnssConstants.S_IN_H
        PlotConf["yData"][Label] = CorrData["STD"][FilterCond]
        PlotConf["zData"][Label] = CorrData["ELEV"][FilterCond]

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/STD_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")

def plotStdVsElevation(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"STD vs Elevation from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["xLabel"] = "Elevation [deg]"
    PlotConf["yLabel"] = "STD [m]"

    PlotConf["xLim"] = [0, 90]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = 'o'
    PlotConf["MarkerSize"] = 8
    PlotConf["Alpha"] = 0.7 

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "PRN"
    PlotConf["ColorBarTicks"] = sorted(set(CorrData["PRN"][CorrData["FLAG"] == 1]))
    PlotConf["ColorBarMin"] = min(PlotConf["ColorBarTicks"])
    PlotConf["ColorBarMax"] = max(PlotConf["ColorBarTicks"])

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    FilterCond = CorrData["FLAG"] == 1

    Elev = CorrData["ELEV"][FilterCond]
    Std = CorrData["STD"][FilterCond]
    Prn = CorrData["PRN"][FilterCond]

    PlotConf["xData"]["POLAR"] = Elev
    PlotConf["yData"]["POLAR"] = Std
    PlotConf["zData"]["POLAR"] = Prn

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/STDvsELEV_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")

def plotSigmaTropoVsElevation(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Sigma Troposphere vs Elevation from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["xLabel"] = "Elevation [deg]"
    PlotConf["yLabel"] = "Sigma Tropo [m]"

    PlotConf["xLim"] = [0, 90]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = 'o'
    PlotConf["MarkerSize"] = 8
    PlotConf["Alpha"] = 0.7

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "PRN"
    PlotConf["ColorBarTicks"] = sorted(set(CorrData["PRN"][CorrData["FLAG"] == 1]))
    PlotConf["ColorBarMin"] = min(PlotConf["ColorBarTicks"])
    PlotConf["ColorBarMax"] = max(PlotConf["ColorBarTicks"])

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    FilterCond = CorrData["FLAG"] == 1

    Elev = CorrData["ELEV"][FilterCond]
    SigmaTropo = CorrData["STROPO"][FilterCond]  
    Prn = CorrData["PRN"][FilterCond]

    PlotConf["xData"][0] = Elev
    PlotConf["yData"][0] = SigmaTropo
    PlotConf["zData"][0] = Prn

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/SigmaTROPOvsELEV_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")

def plotSigmaMPvsElevation(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Sigma Multipath vs Elevation from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["xLabel"] = "Elevation [deg]"
    PlotConf["yLabel"] = "Sigma Multipath [m]"

    PlotConf["xLim"] = [0, 90]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = 'o'
    PlotConf["MarkerSize"] = 8
    PlotConf["Alpha"] = 0.7

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "PRN"
    PlotConf["ColorBarTicks"] = sorted(set(CorrData["PRN"][CorrData["FLAG"] == 1]))
    PlotConf["ColorBarMin"] = min(PlotConf["ColorBarTicks"])
    PlotConf["ColorBarMax"] = max(PlotConf["ColorBarTicks"])

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    FilterCond = CorrData["FLAG"] == 1

    Elev = CorrData["ELEV"][FilterCond]
    SigmaMP = CorrData["SMP"][FilterCond]  
    Prn = CorrData["PRN"][FilterCond]

    PlotConf["xData"][0] = Elev
    PlotConf["yData"][0] = SigmaMP
    PlotConf["zData"][0] = Prn

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/SigmaMPvsELEV_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")

def plotSigmaNoiseDivVsElevation(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Sigma Noise vs Elevation from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["xLabel"] = "Elevation [deg]"
    PlotConf["yLabel"] = "Sigma Noise [m]"

    PlotConf["xLim"] = [0, 90]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = 'o'
    PlotConf["MarkerSize"] = 8
    PlotConf["Alpha"] = 0.7

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "PRN"
    PlotConf["ColorBarTicks"] = sorted(set(CorrData["PRN"][CorrData["FLAG"] == 1]))
    PlotConf["ColorBarMin"] = min(PlotConf["ColorBarTicks"])
    PlotConf["ColorBarMax"] = max(PlotConf["ColorBarTicks"])

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    FilterCond = CorrData["FLAG"] == 1

    Elev = CorrData["ELEV"][FilterCond]
    SigmaNoise = CorrData["SNOISEDIV"][FilterCond] 
    Prn = CorrData["PRN"][FilterCond]

    PlotConf["xData"][0] = Elev
    PlotConf["yData"][0] = SigmaNoise
    PlotConf["zData"][0] = Prn

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/SigmaNOISEDIVvsELEV_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")


def plotSigmaAIRvsElevation(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Sigma Airborne vs Elevation from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["xLabel"] = "Elevation [deg]"
    PlotConf["yLabel"] = "Sigma Airborne [m]"

    PlotConf["xLim"] = [0, 90]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = 'o'
    PlotConf["MarkerSize"] = 8
    PlotConf["Alpha"] = 0.7

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "PRN"
    PlotConf["ColorBarMin"] = 1
    PlotConf["ColorBarMax"] = 32
    PlotConf["ColorBarTicks"] = list(range(1, 33)) 

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    FilterCond = CorrData["FLAG"] == 1

    Elev = CorrData["ELEV"][FilterCond]
    SigmaAIR = CorrData["SAIR"][FilterCond]
    Prn = CorrData["PRN"][FilterCond]

    PlotConf["xData"][0] = Elev
    PlotConf["yData"][0] = SigmaAIR
    PlotConf["zData"][0] = Prn

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/SigmaAIRvsELEV_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")

def plotSigmaUEREvsElevation(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Sigma UERE vs Elevation from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["xLabel"] = "Elevation [deg]"
    PlotConf["yLabel"] = "Sigma UERE [m]"

    PlotConf["xLim"] = [0, 90]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = 'o'
    PlotConf["MarkerSize"] = 8
    PlotConf["Alpha"] = 0.7

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "PRN"
    PlotConf["ColorBarTicks"] = sorted(set(CorrData["PRN"][CorrData["FLAG"] == 1]))
    PlotConf["ColorBarMin"] = min(PlotConf["ColorBarTicks"])
    PlotConf["ColorBarMax"] = max(PlotConf["ColorBarTicks"])

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    FilterCond = CorrData["FLAG"] == 1

    Elev = CorrData["ELEV"][FilterCond]
    SigmaUERE = CorrData["SUERE"][FilterCond]
    Prn = CorrData["PRN"][FilterCond]

    PlotConf["xData"][0] = Elev
    PlotConf["yData"][0] = SigmaUERE
    PlotConf["zData"][0] = Prn

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/SigmaUEREvsELEV_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")

def plotSigmaUEREstacked(CorrData, RcvrLabel, Year, Doy):

    # Filter valid line-of-sight entries
    FilterCond = CorrData["FLAG"] == 1
    ValidData = CorrData[FilterCond]

    # Extract PRNs and format labels
    PRNs = sorted(np.unique(ValidData["PRN"]))
    Labels = [f"G{int(prn):02d}" for prn in PRNs]

    # Compute UERE statistics per PRN
    MaxVals = np.array([ValidData["SUERE"][ValidData["PRN"] == prn].max() for prn in PRNs])
    P95Vals = np.array([np.percentile(ValidData["SUERE"][ValidData["PRN"] == prn], 95) for prn in PRNs])
    RMSVals = np.array([np.sqrt(np.mean(ValidData["SUERE"][ValidData["PRN"] == prn]**2)) for prn in PRNs])
    MinVals = np.array([ValidData["SUERE"][ValidData["PRN"] == prn].min() for prn in PRNs])

    # Compute stacked bar segment heights
    Red = MaxVals - P95Vals
    Orange = P95Vals - RMSVals
    Green = RMSVals - MinVals
    Blue = MinVals

    # Prepare main plot
    x = np.arange(len(PRNs))
    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(bottom=0.30)  # Reserve space for table

    bar_width = 0.4

    # Plot stacked bars
    ax.bar(x, Blue, width=bar_width, color='blue')
    ax.bar(x, Green, bottom=Blue, width=bar_width, color='green')
    ax.bar(x, Orange, bottom=Blue + Green, width=bar_width, color='orange')
    ax.bar(x, Red, bottom=Blue + Green + Orange, width=bar_width, color='red')

    # Configure main axis
    ax.set_title(f"Sigma UERE Statistics per Satellite from {RcvrLabel} on Year {Year} Doy {Doy}")
    ax.set_xlabel("")
    ax.set_ylabel("Sigma UERE [m]")
    ax.set_xticks(x)
    ax.set_xticklabels(Labels)
    ax.set_xlim(-1.5, len(PRNs) - 0.5)
    ax.set_ylim(0.0, np.max(MaxVals) * 1.1)  # No negative values
    ax.grid(True, axis='y')

    # Table row positions (Y coordinates below the bars)
    row_y = [np.max(MaxVals) * -0.13,  # Max
             np.max(MaxVals) * -0.20,  # 95%
             np.max(MaxVals) * -0.27,  # RMS
             np.max(MaxVals) * -0.34]  # Min

    # Plot table values aligned with PRNs
    for i in range(len(PRNs)):
        ax.text(x[i], row_y[0], f"{MaxVals[i]:.2f}", ha='center', va='top', color='red', fontsize=8)
        ax.text(x[i], row_y[1], f"{P95Vals[i]:.2f}", ha='center', va='top', color='orange', fontsize=8)
        ax.text(x[i], row_y[2], f"{RMSVals[i]:.2f}", ha='center', va='top', color='green', fontsize=8)
        ax.text(x[i], row_y[3], f"{MinVals[i]:.2f}", ha='center', va='top', color='blue', fontsize=8)

    # Add left-side labels for each row
    label_x = -1.0
    ax.text(label_x, row_y[0], "Max", ha='right', va='top', color='red', fontsize=9)
    ax.text(label_x, row_y[1], "95%", ha='right', va='top', color='orange', fontsize=9)
    ax.text(label_x, row_y[2], "RMS", ha='right', va='top', color='green', fontsize=9)
    ax.text(label_x, row_y[3], "Min", ha='right', va='top', color='blue', fontsize=9)

    # Save figure to output path
    OutPath = sys.argv[1] + f'/OUT/PCOR/figures/SigmaUERE_STATS_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'
    plt.savefig(OutPath)
    plt.close()

    print(f"    - Generated: Sigma UERE stacked → {OutPath}")

def plotReceiverClockVsTime(CorrData, RcvrLabel, Year, Doy):
    
    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Receiver Clock Estimation from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["yLabel"] = "Receiver Clock [m]"
    PlotConf["xLabel"] = f"Hour of DoY {Doy:03d}"

    PlotConf["FormatYAxisPlain"] = True # Custom flag to format Y axis plainly

    PlotConf["xTicks"] = range(0, 25)
    PlotConf["xLim"] = [0, 24]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = '|'
    PlotConf["LineWidth"] = 1

    # Disable colorbar and zData
    PlotConf["ColorBar"] = None
    PlotConf["ColorBarLabel"] = ""
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    for prn in sorted(unique(CorrData["PRN"])):
        Label = f"G{int(prn):02d}"
        FilterCond = (CorrData["PRN"] == prn) & (CorrData["FLAG"] == 1)

        x_vals = CorrData["#SOD"][FilterCond] / GnssConstants.S_IN_H
        y_vals = CorrData["RCVR-CLK"][FilterCond]

        PlotConf["xData"][Label] = x_vals
        PlotConf["yData"][Label] = y_vals
        PlotConf["zData"][Label] = [0] * len(x_vals)  

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/RCVR-CLK_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'
    PlotConf["ReturnAx"] = True

    ax = generatePlot(PlotConf)
    fig = ax.figure

    fig.canvas.draw()
    
    print(f"    - Generated: {PlotConf['Title']}\n")
  
def plotCodeResidualsVsTime(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Code Residuals from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["yLabel"] = "Code Residual [m]"
    PlotConf["xLabel"] = f"Hour of DoY {Doy:03d}"

    PlotConf["xTicks"] = range(0, 25)
    PlotConf["xLim"] = [0, 24]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = '|'
    PlotConf["LineWidth"] = 1

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "PRN"
    PlotConf["ColorBarTicks"] = sorted(set(CorrData["PRN"][CorrData["FLAG"] == 1]))
    PlotConf["ColorBarMin"] = min(PlotConf["ColorBarTicks"])
    PlotConf["ColorBarMax"] = max(PlotConf["ColorBarTicks"])

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    for prn in sorted(unique(CorrData["PRN"])):
        Label = f"G{int(prn):02d}"
        FilterCond = (CorrData["PRN"] == prn) & (CorrData["FLAG"] == 1)

        PlotConf["xData"][Label] = CorrData["#SOD"][FilterCond] / GnssConstants.S_IN_H
        PlotConf["yData"][Label] = CorrData["CODE-RES"][FilterCond]
        PlotConf["zData"][Label] = np.full(np.sum(FilterCond), prn)  # PRN as Z value

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/CODE-RES_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")
    

def plotPhaseResidualsVsTime(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Phase Residuals from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["yLabel"] = "Phase Residual [m]"
    PlotConf["xLabel"] = f"Hour of DoY {Doy:03d}"

    PlotConf["xTicks"] = range(0, 25)
    PlotConf["xLim"] = [0, 24]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = '|'
    PlotConf["LineWidth"] = 1

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "PRN"
    PlotConf["ColorBarTicks"] = sorted(set(CorrData["PRN"][CorrData["FLAG"] == 1]))
    PlotConf["ColorBarMin"] = min(PlotConf["ColorBarTicks"])
    PlotConf["ColorBarMax"] = max(PlotConf["ColorBarTicks"])

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    for prn in sorted(unique(CorrData["PRN"])):
        Label = f"G{int(prn):02d}"
        FilterCond = (CorrData["PRN"] == prn) & (CorrData["FLAG"] == 1)

        PlotConf["xData"][Label] = CorrData["#SOD"][FilterCond] / GnssConstants.S_IN_H
        PlotConf["yData"][Label] = CorrData["PHASE-RES"][FilterCond]
        PlotConf["zData"][Label] = np.full(np.sum(FilterCond), prn)  # PRN as Z value

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/PHASE-RES_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")


def plotPhaseResidualsVsTimeZoom(CorrData, RcvrLabel, Year, Doy):

    PlotConf = {}

    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (8.4, 6.6)

    PlotConf["Title"] = f"Phase Residuals from {RcvrLabel} on Year {Year} Doy {Doy}"
    PlotConf["yLabel"] = "Phase Residual [m]"
    PlotConf["xLabel"] = f"Hour of DoY {Doy:03d}"

    PlotConf["xTicks"] = range(0, 25)
    PlotConf["xLim"] = [0, 24]
    PlotConf["Grid"] = True
    PlotConf["Marker"] = '|'
    PlotConf["LineWidth"] = 1

    valid_prns = sorted(set(CorrData["PRN"][CorrData["FLAG"] == 1]))
    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "PRN"
    PlotConf["ColorBarTicks"] = valid_prns
    PlotConf["ColorBarMin"] = min(valid_prns)
    PlotConf["ColorBarMax"] = max(valid_prns)

    # Adaptive Y-axis limits based on data distribution
    residuals = CorrData["PHASE-RES"][CorrData["FLAG"] == 1]
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Expand slightly to avoid tight cropping
    margin = 0.1 * (upper_bound - lower_bound)
    PlotConf["yLim"] = [lower_bound - margin, upper_bound + margin]

    # If the range is too narrow, fallback to full range
    if (PlotConf["yLim"][1] - PlotConf["yLim"][0]) < 10:
        PlotConf["yLim"] = [np.min(residuals), np.max(residuals)]

    # Data assignment
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    for prn in valid_prns:
        Label = f"G{int(prn):02d}"
        FilterCond = (CorrData["PRN"] == prn) & (CorrData["FLAG"] == 1)

        PlotConf["xData"][Label] = CorrData["#SOD"][FilterCond] / GnssConstants.S_IN_H
        PlotConf["yData"][Label] = CorrData["PHASE-RES"][FilterCond]
        PlotConf["zData"][Label] = np.full(np.sum(FilterCond), prn)

    PlotConf["Path"] = sys.argv[1] + f'/OUT/PCOR/figures/PHASE-RES-ZOOM_{RcvrLabel}_Y{Year % 100:02d}D{Doy:03d}.png'

    generatePlot(PlotConf)
    print(f"    - Generated (zoomed): {PlotConf['Title']}\n")

def generateCorrPlots(CorrFile, RcvrInfo):
    """
    Purpose
    -------
    Generate output plots based on corrected GNSS measurements (PCOR), including
    satellite geometry, residuals, and receiver clock behavior.

    Parameters
    ----------
    CorrFile : str
        Path to the corrected measurements file (PCOR).
    RcvrInfo : list
        List containing receiver metadata (e.g., name, position, etc.).

    Returns
    -------
    None
        This function generates and saves plots but does not return any value.

    Notes
    -----
    This function is part of the PEPPUS pipeline and is called after correction
    outputs are written. It visualizes key metrics such as:
    - Satellite visibility and geometry
    - Code and phase residuals
    - Receiver clock drift and corrections
    """
    
    # Extract metadata
    RcvrLabel = RcvrInfo[0]
    # Year = int(CorrFile.split("Y")[1][:2]) + 2000
    # Doy = int(CorrFile.split("D")[1][:3])

    match = re.search(r'_Y(\d{2})D(\d{3})', CorrFile)
    if match:
        Year = int(match.group(1)) + 2000
        Doy = int(match.group(2))
    else:
        raise ValueError(f"Could not extract Year/DoY from filename: {CorrFile}")
    OutPath = os.path.join(os.path.dirname(CorrFile), "figures")
    if not os.path.exists(OutPath):
        os.makedirs(OutPath)

    # Read corrected data
    # CorrData = read_csv(CorrFile, delim_whitespace=True)
    CorrData = read_csv(CorrFile, delim_whitespace=True, dtype={"PRN": int})


    # Plot satellite tracks
    plotSatTracks(CorrData, RcvrLabel, Year, Doy)

    # Plot Time of Flight vs Time
    plotToFvsTime(CorrData, RcvrLabel, Year, Doy)

    # Plot DTR vs Time
    plotDTRvsTime(CorrData, RcvrLabel, Year, Doy)

    # Plot STD vs Time
    plotStropoVsTime(CorrData, RcvrLabel, Year, Doy)

    # Plot STD vs Elevation
    plotStdVsElevation(CorrData, RcvrLabel, Year, Doy)

    # Plot Sigma Tropo vs Elevation
    plotSigmaTropoVsElevation(CorrData, RcvrLabel, Year, Doy)

    # Plot Sigma MP vs Elevation
    plotSigmaMPvsElevation(CorrData, RcvrLabel, Year, Doy)

    # Plot Sigma Noise Div vs Elevation
    plotSigmaNoiseDivVsElevation(CorrData, RcvrLabel, Year, Doy)

    # Plot Sigma Airborne vs Elevation
    plotSigmaAIRvsElevation(CorrData, RcvrLabel, Year, Doy)

    # Plot Sigma UERE vs Elevation
    plotSigmaUEREvsElevation(CorrData, RcvrLabel, Year, Doy)

    # Plot Sigma UERE statistics per satellite
    plotSigmaUEREstacked(CorrData, RcvrLabel, Year, Doy)

    # Plot Receiver Clock vs Time
    plotReceiverClockVsTime(CorrData, RcvrLabel, Year, Doy)

    # Plot Code Residuals vs Time
    plotCodeResidualsVsTime(CorrData, RcvrLabel, Year, Doy)

    # Plot Phase Residuals vs Time
    plotPhaseResidualsVsTime(CorrData, RcvrLabel, Year, Doy)
    plotPhaseResidualsVsTimeZoom(CorrData, RcvrLabel, Year, Doy)




    