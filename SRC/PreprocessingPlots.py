#!/usr/bin/env python

########################################################################
# PEPPUS/SRC/PreprocessingPlots.py:
# This is the PreprocessingPlots Module of PEPPUS tool
#
#  Project:        PEPPUS
#  File:           PreprocessingPlots.py
#
#   Author: GNSS Academy
#   Copyright GNSS Academy
#
########################################################################

import sys, os
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from pandas import unique
from pandas import read_csv
from InputOutput import PreproIdx
from InputOutput import REJECTION_CAUSE_DESC
sys.path.append(os.getcwd() + '/' + \
    os.path.dirname(sys.argv[0]) + '/' + 'COMMON')
from COMMON import GnssConstants
from COMMON.Plots import generatePlot
import numpy as np
from collections import OrderedDict
from COMMON import GnssConstants as Const
from matplotlib import cm



def initPlot(PreproObsFile, PlotConf, Title, Label):
    PreproObsFileName = os.path.basename(PreproObsFile)
    PreproObsFileNameSplit = PreproObsFileName.split('_')
    Rcvr = PreproObsFileNameSplit[2]
    DatepDat = PreproObsFileNameSplit[3]
    Date = DatepDat.split('.')[0]
    Year = Date[1:3]
    Doy = Date[4:]

    os.makedirs(sys.argv[1] + '/OUT/PPVE/figures/' + Rcvr, exist_ok=True)

    PlotConf["xLabel"] = "Hour of Day %s" % Doy

    PlotConf["Title"] = "%s from %s on Year %s"\
        " DoY %s" % (Title, Rcvr, Year, Doy)
    
    PlotConf["Path"] = sys.argv[1] + '/OUT/PPVE/figures/' + Rcvr + '/' + \
    '%s_%s_Y%sD%s.png' % (Label, Rcvr, Year, Doy)


# Plot Satellite Visibility
def plotSatVisibility(PreproObsFile, PreproObsData):
    PlotConf = {}
    
    # X LABEL
    PlotConf["xTicks"] = range(0, 25) # Defining tick marks (labels) for the x-axis, from 0 to 24 hours
    PlotConf["xLim"] = [0, 24] # Setting the visible range of the x-axis (0 to 24 hours)

    # Y LABEL
    PreproObsData["PRN"] = PreproObsData["PRN"].astype(int)
    prns = sorted(unique(PreproObsData["PRN"]))
    PreproObsData["C"] = PreproObsData["C"]
    PreproIdx["PRN"] = PreproObsData.columns.get_loc("PRN")

    PlotConf["yTicks"] = prns
    PlotConf["yLim"] = [0, len(prns) + 2]
    PlotConf["yTickLabels"] = prns
    PlotConf["yLabel"] = "PRN"

    PlotConf["Grid"] = True # Enabling grid lines on the plot

    PlotConf["Marker"] = '.' # Defining the marker to plot, could be a ".", "x", "+" and so on
    PlotConf["LineWidth"] = 1 # Defining the line width
    PlotConf["Type"] = "Lines"

    PlotConf["FigSize"] = (12, 8)

    PlotConf["ColorBar"] = "gnuplot" # Defining the graphic color gradient style
    PlotConf["ColorBarLabel"] = "Elevation [deg]" # Defining the elevation label
    PlotConf["ColorBarMin"] = 0. # Defining the minimum for elevation label
    PlotConf["ColorBarMax"] = 90. # Defining the maximum for elevation label (at 90º for zenith)
    PlotConf["Legend"] = False

    # Initializing the dictionary
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}
    PlotConf["StatusData"] = {}

    # Loop through each unique satellite PRN:
    for prn in prns:
        # LabelSat = "G%02d" % prn  # Create a label for each satellite using its PRN
        FilterCond = PreproObsData["PRN"] == prn # Generate a filter condition to select only the rows that correspond to the current PRN

        PlotConf["xData"][prn] = PreproObsData["#SOD"][FilterCond].astype(float) / Const.S_IN_H
        PlotConf["yData"][prn] = [prn] * sum(FilterCond)
        PlotConf["zData"][prn] = PreproObsData["ELEV"][FilterCond].astype(float)
        PlotConf["StatusData"][prn] = PreproObsData["STATUS"][FilterCond].astype(int).tolist()

    Title = "Satellite Visibility"
    Label = "SatVisibility"

    # Initializing Plot (defined previously)
    initPlot(PreproObsFile, PlotConf, Title, Label)

    # Call generatePlot from Plots library
    generatePlot(PlotConf)

    # Informing the graphic has been generated successfully 
    print(f"    - Generated: {PlotConf['Title']}\n")
    
# Plot Number of Satellites
def plotNumSats(PreproObsFile, PreproObsData):
    PlotConf = {}
    PreproObsData["STATUS"] = PreproObsData["STATUS"].astype(int)

    # X LABEL
    PreproObsData["#SOD"] = PreproObsData["#SOD"].astype(float)
    PlotConf["xTicks"] = range(0, 25) # Defining tick marks (labels) for the x-axis, from 0 to 24 hours
    PlotConf["xLim"] = [0, 24] # Setting the visible range of the x-axis (0 to 24 hours)

    # Y LABEL
    PreproObsData["PRN"] = PreproObsData["PRN"].astype(int)
    prns = sorted(unique(PreproObsData["PRN"]))
    PlotConf["yLim"] = [0, 13]
    PlotConf["yLabel"] = "NSAT"

    PlotConf["Grid"] = True # Enabling grid lines on the plot

    PlotConf["Marker"] = 'o' # Defining the marker to plot, could be a ".", "x", "+" and so on
    PlotConf["MarkerSize"] = 6
    PlotConf["LineWidth"] = 0 # Defining the line width
    PlotConf["Type"] = "Lines"

    PlotConf["FigSize"] = (14, 6)
    PlotConf["Legend"] = True

    # Initializing the dictionary
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}

    # Total NSAT
    nsat_total = PreproObsData.groupby("#SOD")["PRN"].nunique()
    x_total = nsat_total.index / Const.S_IN_H
    y_total = nsat_total.values
    PlotConf["xData"]["Raw"] = x_total.tolist()
    PlotConf["yData"]["Raw"] = y_total.tolist()

    # NSAT with STATUS == 1

    filterCond = PreproObsData[PreproObsData["STATUS"] == 1].copy()
    filterCond["SOD_int"] = filterCond["#SOD"].astype(int)
    nsat_status = filterCond.groupby("SOD_int")["PRN"].nunique()

    
    all_seconds = PreproObsData["#SOD"].astype(int).unique()
    all_seconds.sort()
    nsat_status = nsat_status.reindex(all_seconds, fill_value=0)

    x_status = nsat_status.index / Const.S_IN_H
    y_status = nsat_status.values
    
    PlotConf["xData"]["OK"] = x_status.tolist()
    PlotConf["yData"]["OK"] = y_status.tolist()

    # Determining colors
    PlotConf["Colors"] = {
        "Raw": "orange",
        "OK": "green"
    }

    PlotConf["zData"] = {
        "Raw": [0] * len(PlotConf["xData"]["Raw"]),
        "OK": [0] * len(PlotConf["xData"]["OK"])
    }

    PlotConf["Legend"] = True
    PlotConf["LegendLoc"] = "upper right"
    PlotConf["LegendLabels"] = {
        "Raw": "Raw",
        "OK": "OK"
    }

    Title = "Number of Satellites"
    Label = "NSAT"
    PlotConf["LegendLoc"] = "upper right"

    # Initializing Plot (defined previously)
    initPlot(PreproObsFile, PlotConf, Title, Label)

    # Call generatePlot from Plots library
    generatePlot(PlotConf)

    # Informing the graphic has been generated successfully 
    print(f"    - Generated: {PlotConf['Title']}\n")


# Plot Rejection Flags
def plotRejectionFlags(PreproObsFile, PreproObsData):
    
    # Initializing the dictionary
    PlotConf = {}
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}
    PlotConf["ReturnAx"] = True

    
    # Converting data
    PreproObsData["REJ"] = PreproObsData["REJ"].astype(int)
    PreproObsData["PRN"] = PreproObsData["PRN"].astype(int)
    PreproObsData["#SOD"] = PreproObsData["#SOD"].astype(float) / Const.S_IN_H
    # X LABEL
    PlotConf["xTicks"] = range(0, 25) # Defining tick marks (labels) for the x-axis, from 0 to 24 hours
    PlotConf["xLim"] = [0, 24] # Setting the visible range of the x-axis (0 to 24 hours)

    # Y LABEL
    REJECTION_CAUSE_DESC = [
        "0: Valid",
        "1: Number of Channels for GPS",
        "2: Mask Angle",
        "3: Minimum C/N0 in L1",
        "4: Minimum C/N0 in L2",
        "5: Minimum C/N0 in L1 and L2",
        "6: Maximum PR in L1",
        "7: Maximum PR in L2",
        "8: Maximum PR in L1 and L2",
        "9: Data Gap",
        "10: Cycle Slip",
        "11: Maximum Phase Rate",
        "12: Maximum Phase Rate Step",
        "13: Maximum Code Rate",
        "14: Maximum Code Rate Step"
    ]
    PlotConf["yTicks"] = list(range(1, 15))  # 1 to 14
    PlotConf["yLim"] = [0.5, 14.5]
    PlotConf["yTickLabels"] = REJECTION_CAUSE_DESC[1:]
    PlotConf["yLabel"] = "Rejection Flags"

    # Z LABEL
    fixed_colors = {prn: cm.get_cmap("jet")(prn / 32) for prn in range(1, 33)}
    # color_array = [fixed_colors.get(prn, "black") for prn in PreproObsData["PRN"]]
    
    PlotConf["ColorBar"] = "gist_ncar"
    PlotConf["ColorBarTicks"] = list(range(1, 33))  # PRNs de 1 a 32
    PlotConf["ColorBarLabel"] = "GPS-PRN" # Defining the elevation label
    PlotConf["ColorBarMin"] = 1 # Defining the minimum for elevation label
    PlotConf["ColorBarMax"] = 32 # Defining the maximum for elevation label (at 90º for zenith)

    PlotConf["Grid"] = True # Enabling grid lines on the plot
    PlotConf["Marker"] = '.' # Defining the marker to plot, could be a ".", "x", "+" and so on
    PlotConf["LineWidth"] = 1 # Defining the line width
    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (12, 8)
    PlotConf["Legend"] = False
    PlotConf["MarkerSize"] = 5

    Title = "Rejection Flags"
    Label = "RejFlags"

    PlotConf["xData"] = {"REJ_ALL": PreproObsData["#SOD"]}
    PlotConf["yData"] = {"REJ_ALL": PreproObsData["REJ"]}
    PlotConf["zData"] = {"REJ_ALL": PreproObsData["PRN"].astype(int)}

    filtered = PreproObsData[(PreproObsData["REJ"] != 0) & (PreproObsData["REJ"] != 2)] # Do not plot the sat label over Rejection Flag number 0 and 2 to keep the graphic clean
    filtered["HOUR"] = filtered["#SOD"].astype(int)
    PlotConf["Labels"] = filtered
    
    # Initializing Plot (defined previously)
    initPlot(PreproObsFile, PlotConf, Title, Label)

    # Call generatePlot from Plots library
    ax = generatePlot(PlotConf)
    

    # Informing the graphic has been generated successfully 
    print(f"    - Generated: {PlotConf['Title']}\n")



# Plot Code Rate
def plotCodeRate(PreproObsFile, PreproObsData):  
    PlotConf = {}

    # Initializing the dictionary
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    # Converting to float
    PreproObsData["PRN"] = pd.to_numeric(PreproObsData["PRN"], errors="coerce")
    PreproObsData["CODERATE"] = pd.to_numeric(PreproObsData["CODERATE"], errors="coerce")
    PreproObsData["#SOD"] = pd.to_numeric(PreproObsData["#SOD"], errors="coerce")
    PreproObsData["ELEV"] = pd.to_numeric(PreproObsData["ELEV"], errors="coerce")

    # Removing invalid numbers
    PreproObsData = PreproObsData.dropna(subset=["CODERATE", "#SOD", "ELEV"])

    # X LABEL
    PlotConf["xTicks"] = range(0, 25) # Defining tick marks (labels) for the x-axis, from 0 to 24 hours
    PlotConf["xLim"] = [0, 24] # Setting the visible range of the x-axis (0 to 24 hours)

    # Y LABEL

    PlotConf["yTicks"] = np.arange(-800, 810, 200)
    PlotConf["yLim"] = [-800, 800]
    PlotConf["yTickLabels"] = [str(v) for v in PlotConf["yTicks"]]
    PlotConf["yLabel"] = "Code Rate [m/s]"

    PlotConf["Grid"] = True # Enabling grid lines on the plot

    PlotConf["Marker"] = '.' # Defining the marker to plot, could be a ".", "x", "+" and so on
    PlotConf["LineWidth"] = 1 # Defining the line width
    PlotConf["Type"] = "Lines"

    PlotConf["FigSize"] = (12, 8)

    PlotConf["ColorBar"] = "gnuplot" # Defining the graphic color gradient style
    PlotConf["ColorBarLabel"] = "Elevation [deg]" # Defining the elevation label
    PlotConf["ColorBarMin"] = 0. # Defining the minimum for elevation label
    PlotConf["ColorBarMax"] = 90. # Defining the maximum for elevation label (at 90º for zenith)
    PlotConf["Legend"] = False
    

    # Loop through each unique satellite PRN:
    for prn, group in PreproObsData.groupby("PRN"):
        LabelSat = f"G{int(prn):02d}"  # Create a label for each satellite using its PRN

        group = group[group["ELEV"] > 10]
        PlotConf["xData"][LabelSat] = group["#SOD"].values / Const.S_IN_H
        PlotConf["yData"][LabelSat] = group["CODERATE"].values
        PlotConf["zData"][LabelSat] = group["ELEV"].values

    Title = "Code Rate"
    Label = "CodeRate"

    filtered = PreproObsData[PreproObsData["ELEV"] > 10]

    if not filtered.empty:
        PlotConf["xData"]["CodeRate"] = filtered["#SOD"].values / Const.S_IN_H
        PlotConf["yData"]["CodeRate"] = filtered["CODERATE"].values
        PlotConf["zData"]["CodeRate"] = filtered["ELEV"].values

    # Initializing Plot (defined previously)
    initPlot(PreproObsFile, PlotConf, Title, Label)

    # Call generatePlot from Plots library
    generatePlot(PlotConf)

    # Informing the graphic has been generated successfully 
    print(f"    - Generated: {PlotConf['Title']}\n")

    ############################################################################
    # CODE RATE STEP
    ############################################################################

    PlotConf = {}

    # Converting to float
    PreproObsData["PRN"] = pd.to_numeric(PreproObsData["PRN"], errors="coerce")
    PreproObsData["CODEACC"] = pd.to_numeric(PreproObsData["CODEACC"], errors="coerce")
    PreproObsData["#SOD"] = pd.to_numeric(PreproObsData["#SOD"], errors="coerce")
    PreproObsData["ELEV"] = pd.to_numeric(PreproObsData["ELEV"], errors="coerce")

    # Removing invalid numbers
    PreproObsData = PreproObsData.dropna(subset=["CODEACC", "#SOD", "ELEV"])
    
    # X LABEL
    PlotConf["xTicks"] = range(0, 25) # Defining tick marks (labels) for the x-axis, from 0 to 24 hours
    PlotConf["xLim"] = [0, 24] # Setting the visible range of the x-axis (0 to 24 hours)

    # Y LABEL
    PlotConf["yTicks"] = np.arange(-20, 20.5, 10)
    PlotConf["yLim"] = [-20.5, 20.5]
    PlotConf["yTickLabels"] = [str(v) for v in PlotConf["yTicks"]]
    PlotConf["yLabel"] = "Code Rate Step [m/s²]"

    PlotConf["Grid"] = True # Enabling grid lines on the plot

    PlotConf["Marker"] = '.' # Defining the marker to plot, could be a ".", "x", "+" and so on
    PlotConf["LineWidth"] = 1 # Defining the line width
    PlotConf["Type"] = "Lines"

    PlotConf["FigSize"] = (12, 8)

    PlotConf["ColorBar"] = "gnuplot" # Defining the graphic color gradient style
    PlotConf["ColorBarLabel"] = "Elevation [deg]" # Defining the elevation label
    PlotConf["ColorBarMin"] = 0. # Defining the minimum for elevation label
    PlotConf["ColorBarMax"] = 90. # Defining the maximum for elevation label (at 90º for zenith)
    PlotConf["Legend"] = False

    # Initializing the dictionary
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}
    

    # Loop through each unique satellite PRN:
    for prn, group in PreproObsData.groupby("PRN"):
        LabelSat = f"G{int(prn):02d}"  # Create a label for each satellite using its PRN
        group = group[group["ELEV"] > 10]
        PlotConf["xData"][LabelSat] = group["#SOD"].values / Const.S_IN_H
        PlotConf["yData"][LabelSat] = group["CODEACC"].values
        PlotConf["zData"][LabelSat] = group["ELEV"].values

    Title = "Code Rate Step"
    Label = "CodeACC"

    filtered = PreproObsData[PreproObsData["ELEV"] > 10]

    if not filtered.empty:
        PlotConf["xData"]["CodeACC"] = filtered["#SOD"].values / Const.S_IN_H
        PlotConf["yData"]["CodeACC"] = filtered["CODEACC"].values
        PlotConf["zData"]["CodeACC"] = filtered["ELEV"].values

    # Initializing Plot (defined previously)
    initPlot(PreproObsFile, PlotConf, Title, Label)

    # Call generatePlot from Plots library
    generatePlot(PlotConf)

    # Informing the graphic has been generated successfully 
    print(f"    - Generated: {PlotConf['Title']}\n")


# Plot Phase Rate
def plotPhaseRate(PreproObsFile, PreproObsData):
    PlotConf = {}

    # Converting to float
    PreproObsData["PRN"] = pd.to_numeric(PreproObsData["PRN"], errors="coerce")
    PreproObsData["PHASERATE"] = pd.to_numeric(PreproObsData["PHASERATE"], errors="coerce")
    PreproObsData["#SOD"] = pd.to_numeric(PreproObsData["#SOD"], errors="coerce")
    PreproObsData["ELEV"] = pd.to_numeric(PreproObsData["ELEV"], errors="coerce")

    # Removing invalid numbers
    PreproObsData = PreproObsData.dropna(subset=["PHASERATE", "#SOD", "ELEV"])
    
    # X LABEL
    PlotConf["xTicks"] = range(0, 25) # Defining tick marks (labels) for the x-axis, from 0 to 24 hours
    PlotConf["xLim"] = [0, 24] # Setting the visible range of the x-axis (0 to 24 hours)

    # Y LABEL

    PlotConf["yTicks"] = np.arange(-800, 810, 200)
    PlotConf["yLim"] = [-800, 800]
    PlotConf["yTickLabels"] = [str(v) for v in PlotConf["yTicks"]]
    PlotConf["yLabel"] = "Phase Rate [m/s]"

    PlotConf["Grid"] = True # Enabling grid lines on the plot

    PlotConf["Marker"] = '.' # Defining the marker to plot, could be a ".", "x", "+" and so on
    PlotConf["LineWidth"] = 1 # Defining the line width
    PlotConf["Type"] = "Lines"

    PlotConf["FigSize"] = (12, 8)

    PlotConf["ColorBar"] = "gnuplot" # Defining the graphic color gradient style
    PlotConf["ColorBarLabel"] = "Elevation [deg]" # Defining the elevation label
    PlotConf["ColorBarMin"] = 0. # Defining the minimum for elevation label
    PlotConf["ColorBarMax"] = 90. # Defining the maximum for elevation label (at 90º for zenith)
    PlotConf["Legend"] = False

    # Initializing the dictionary
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    # Loop through each unique satellite PRN:
    for prn, group in PreproObsData.groupby("PRN"):
        LabelSat = f"G{int(prn):02d}"  # Create a label for each satellite using its PRN
        group = group[group["ELEV"] > 10]

        PlotConf["xData"][LabelSat] = group["#SOD"].values / Const.S_IN_H
        PlotConf["yData"][LabelSat] = group["PHASERATE"].values
        PlotConf["zData"][LabelSat] = group["ELEV"].values


    Title = "Phase Rate"
    Label = "PhaseeRate"
    
    filtered = PreproObsData[PreproObsData["ELEV"] > 10]

    if not filtered.empty:
        PlotConf["xData"]["PhaseRate"] = filtered["#SOD"].values / Const.S_IN_H
        PlotConf["yData"]["PhaseRate"] = filtered["PHASERATE"].values
        PlotConf["zData"]["PhaseRate"] = filtered["ELEV"].values

    # Initializing Plot (defined previously)
    initPlot(PreproObsFile, PlotConf, Title, Label)

    # Call generatePlot from Plots library
    generatePlot(PlotConf)

    # Informing the graphic has been generated successfully 
    print(f"    - Generated: {PlotConf['Title']}\n")


    ############################################################################
    # PHASE RATE STEP
    ############################################################################
    PlotConf = {}

    # Converting to float
    PreproObsData["PRN"] = pd.to_numeric(PreproObsData["PRN"], errors="coerce")
    PreproObsData["PHASEACC"] = pd.to_numeric(PreproObsData["PHASEACC"], errors="coerce")
    PreproObsData["#SOD"] = pd.to_numeric(PreproObsData["#SOD"], errors="coerce")
    
    PreproObsData["ELEV"] = pd.to_numeric(PreproObsData["ELEV"], errors="coerce")

    # Removing invalid numbers
    PreproObsData = PreproObsData.dropna(subset=["PHASEACC", "#SOD", "ELEV"])
    
    # X LABEL
    PlotConf["xTicks"] = range(0, 25) # Defining tick marks (labels) for the x-axis, from 0 to 24 hours
    PlotConf["xLim"] = [0, 24] # Setting the visible range of the x-axis (0 to 24 hours)

    # Y LABEL

    PlotConf["yTicks"] = np.round(np.arange(-0.1, 0.21, 0.05), 2)
    PlotConf["yLim"] = [-0.07, 0.22]
    PlotConf["yTickLabels"] = [str(v) for v in PlotConf["yTicks"]]
    PlotConf["yLabel"] = "Phase Rate Step [m/s²]"

    PlotConf["Grid"] = True # Enabling grid lines on the plot

    PlotConf["Marker"] = '.' # Defining the marker to plot, could be a ".", "x", "+" and so on
    PlotConf["LineWidth"] = 1 # Defining the line width
    PlotConf["Type"] = "Lines"

    PlotConf["FigSize"] = (12, 8)

    PlotConf["ColorBar"] = "gnuplot" # Defining the graphic color gradient style
    PlotConf["ColorBarLabel"] = "Elevation [deg]" # Defining the elevation label
    PlotConf["ColorBarMin"] = 0. # Defining the minimum for elevation label
    PlotConf["ColorBarMax"] = 90. # Defining the maximum for elevation label (at 90º for zenith)
    PlotConf["Legend"] = False

    # Initializing the dictionary
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}
    
    # Loop through each unique satellite PRN:
    for prn, group in PreproObsData.groupby("PRN"):
        LabelSat = f"G{int(prn):02d}"  # Create a label for each satellite using its PRN
        group = group[group["ELEV"] > 10]
        PlotConf["xData"][LabelSat] = group["#SOD"].values / Const.S_IN_H
        PlotConf["yData"][LabelSat] = group["PHASEACC"].values
        PlotConf["zData"][LabelSat] = group["ELEV"].values

    Title = "Phase Rate Step"
    Label = "PhaseACC"

    filtered = PreproObsData[PreproObsData["ELEV"] > 10]

    if not filtered.empty:
        PlotConf["xData"]["PhaseACC"] = filtered["#SOD"].values / Const.S_IN_H
        PlotConf["yData"]["PhaseACC"] = filtered["PHASEACC"].values
        PlotConf["zData"]["PhaseACC"] = filtered["ELEV"].values

    # Initializing Plot (defined previously)
    initPlot(PreproObsFile, PlotConf, Title, Label)

    # Call generatePlot from Plots library
    generatePlot(PlotConf)

    # Informing the graphic has been generated successfully 
    print(f"    - Generated: {PlotConf['Title']}\n")

# VTEC Gradient
def plotVtecGradient(PreproObsFile, PreproObsData):
    PlotConf = {}

    # Converting to float
    PreproObsData["PRN"] = pd.to_numeric(PreproObsData["PRN"], errors="coerce")
    PreproObsData["VTECRATE"] = pd.to_numeric(PreproObsData["VTECRATE"], errors="coerce")
    PreproObsData["#SOD"] = pd.to_numeric(PreproObsData["#SOD"], errors="coerce")
    PreproObsData["ELEV"] = pd.to_numeric(PreproObsData["ELEV"], errors="coerce")

    # Removing invalid numbers
    PreproObsData = PreproObsData.dropna(subset=["VTECRATE", "#SOD", "ELEV"])
    
    # X LABEL
    PlotConf["xTicks"] = range(0, 25) # Defining tick marks (labels) for the x-axis, from 0 to 24 hours
    PlotConf["xLim"] = [0, 24] # Setting the visible range of the x-axis (0 to 24 hours)

    # Y LABEL

    PlotConf["yTicks"] = np.arange(-60, 10, 10)
    PlotConf["yLim"] = [-60, 10]
    PlotConf["yTickLabels"] = [str(v) for v in PlotConf["yTicks"]]
    PlotConf["yLabel"] = "VTEC Gradient [mm/s]"

    PlotConf["Grid"] = True # Enabling grid lines on the plot

    PlotConf["Marker"] = '.' # Defining the marker to plot, could be a ".", "x", "+" and so on
    PlotConf["LineWidth"] = 1 # Defining the line width
    PlotConf["Type"] = "Lines"

    PlotConf["FigSize"] = (12, 8)

    PlotConf["ColorBar"] = "gnuplot" # Defining the graphic color gradient style
    PlotConf["ColorBarLabel"] = "Elevation [deg]" # Defining the elevation label
    PlotConf["ColorBarMin"] = 0. # Defining the minimum for elevation label
    PlotConf["ColorBarMax"] = 90. # Defining the maximum for elevation label (at 90º for zenith)
    PlotConf["Legend"] = False

    # Initializing the dictionary
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    # Loop through each unique satellite PRN:
    for prn, group in PreproObsData.groupby("PRN"):
        LabelSat = f"G{int(prn):02d}"  # Create a label for each satellite using its PRN
        PlotConf["xData"][LabelSat] = group["#SOD"].values / Const.S_IN_H
        
        PlotConf["yData"][LabelSat] = group["VTECRATE"].values
        PlotConf["zData"][LabelSat] = group["ELEV"].values

    Title = "VTEC Gradient"
    Label = "VTECRate"
    PlotConf["xData"]["VTECRATE"] = PreproObsData["#SOD"].values / Const.S_IN_H
    PlotConf["yData"]["VTECRATE"] = PreproObsData["VTECRATE"].values
    PlotConf["zData"]["VTECRATE"] = PreproObsData["ELEV"].values

    # Initializing Plot (defined previously)
    initPlot(PreproObsFile, PlotConf, Title, Label)

    # Call generatePlot from Plots library
    generatePlot(PlotConf)

    # Informing the graphic has been generated successfully 
    print(f"    - Generated: {PlotConf['Title']}\n")


# AATR index
def plotAatr(PreproObsFile, PreproObsData):
    PlotConf = {}

    # Converting to float
    PreproObsData["PRN"] = pd.to_numeric(PreproObsData["PRN"], errors="coerce")
    PreproObsData["iAATR"] = pd.to_numeric(PreproObsData["iAATR"], errors="coerce")
    PreproObsData["#SOD"] = pd.to_numeric(PreproObsData["#SOD"], errors="coerce")
    PreproObsData["ELEV"] = pd.to_numeric(PreproObsData["ELEV"], errors="coerce")

    # Removing invalid numbers
    PreproObsData = PreproObsData.dropna(subset=["iAATR", "#SOD", "ELEV"])
    
    # X LABEL
    PlotConf["xTicks"] = range(0, 25) # Defining tick marks (labels) for the x-axis, from 0 to 24 hours
    PlotConf["xLim"] = [0, 24] # Setting the visible range of the x-axis (0 to 24 hours)

    # Y LABEL

    PlotConf["yTicks"] = np.arange(-25, 5, 5)
    PlotConf["yLim"] = [-25, 5]
    PlotConf["yTickLabels"] = [str(v) for v in PlotConf["yTicks"]]
    PlotConf["yLabel"] = "AATR [mm/s]"

    PlotConf["Grid"] = True # Enabling grid lines on the plot

    PlotConf["Marker"] = '.' # Defining the marker to plot, could be a ".", "x", "+" and so on
    PlotConf["LineWidth"] = 1 # Defining the line width
    PlotConf["Type"] = "Lines"

    PlotConf["FigSize"] = (12, 8)

    PlotConf["ColorBar"] = "gnuplot" # Defining the graphic color gradient style
    PlotConf["ColorBarLabel"] = "Elevation [deg]" # Defining the elevation label
    PlotConf["ColorBarMin"] = 0. # Defining the minimum for elevation label
    PlotConf["ColorBarMax"] = 90. # Defining the maximum for elevation label (at 90º for zenith)
    PlotConf["Legend"] = False

    # Initializing the dictionary
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    # Loop through each unique satellite PRN:
    for prn, group in PreproObsData.groupby("PRN"):
        LabelSat = f"G{int(prn):02d}"  # Create a label for each satellite using its PRN
        PlotConf["xData"][LabelSat] = group["#SOD"].values / Const.S_IN_H
        PlotConf["yData"][LabelSat] = group["iAATR"].values
        PlotConf["zData"][LabelSat] = group["ELEV"].values

    Title = "AATR"
    Label = "AATR"
    PlotConf["xData"]["iAATR"] = PreproObsData["#SOD"].values / Const.S_IN_H
    PlotConf["yData"]["iAATR"] = PreproObsData["iAATR"].values
    PlotConf["zData"]["iAATR"] = PreproObsData["ELEV"].values

    # Initializing Plot (defined previously)
    initPlot(PreproObsFile, PlotConf, Title, Label)

    # Call generatePlot from Plots library
    generatePlot(PlotConf)

    # Informing the graphic has been generated successfully 
    print(f"    - Generated: {PlotConf['Title']}\n")

    



def plotPolarView(PreproObsFile, PreproObsData):
    PlotConf = {}

    # Converting to float
    PreproObsData["PRN"] = pd.to_numeric(PreproObsData["PRN"], errors="coerce")
    PreproObsData["AZIM"] = pd.to_numeric(PreproObsData["AZIM"], errors="coerce")
    PreproObsData["ELEV"] = pd.to_numeric(PreproObsData["ELEV"], errors="coerce")

    PreproObsData = PreproObsData.dropna(subset=["PRN", "AZIM", "ELEV"])
    PreproObsData = PreproObsData[PreproObsData["ELEV"] > 10]  

    # Initializing the dictionary
    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}

    # Converting to polar coordinates
    theta = np.deg2rad(PreproObsData["AZIM"].values)
    r = PreproObsData["ELEV"].values
    prn = PreproObsData["PRN"].astype(int).values

    PlotConf["xData"]["POLAR"] = theta
    PlotConf["yData"]["POLAR"] = r
    PlotConf["zData"]["POLAR"] = prn

    # Graphic conf
    PlotConf["Type"] = "Polar"
    PlotConf["FigSize"] = (8, 8)
    PlotConf["ColorBar"] = "gist_ncar"
    PlotConf["ColorBarLabel"] = "PRN"
    
    PlotConf["ColorBarMin"] = 1
    PlotConf["ColorBarMax"] = 32
    PlotConf["Marker"] = 'o'
    PlotConf["MarkerSize"] = 10
    PlotConf["Grid"] = True

    Title = "Satellite Polar View "
    Label = "PolarPRN"

    initPlot(PreproObsFile, PlotConf, Title, Label)
    generatePlot(PlotConf)
    print(f"    - Generated: {PlotConf['Title']}\n")


def generatePreproPlots(PreproObsFile):
    
    # Purpose: generate output plots regarding Preprocessing results

    # Parameters
    # ==========
    # PreproObsFile: str
    #         Path to PREPRO OBS output file

    # Returns
    # =======
    # Nothing
    
    # Satellite Visibility
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file

    PreproObsData = read_csv(
        PreproObsFile,
        sep='\s+',
        header=0,
        dtype=str,
        low_memory=False
    )
    
    plotCodeRate(PreproObsFile, PreproObsData)
    plotPhaseRate(PreproObsFile, PreproObsData)
    plotVtecGradient(PreproObsFile, PreproObsData)
    plotAatr(PreproObsFile, PreproObsData)
    plotSatVisibility(PreproObsFile, PreproObsData)
    plotNumSats(PreproObsFile, PreproObsData)
    plotRejectionFlags(PreproObsFile, PreproObsData)
    plotPolarView(PreproObsFile, PreproObsData)
