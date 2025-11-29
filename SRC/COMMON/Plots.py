
import sys, os
import matplotlib as mpl
mpl.use("Agg")   # to avoid crashing in newer Anaconda versions
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import conda
CondaFileDir = conda.__file__
CondaDir = CondaFileDir.split('lib')[0]
ProjLib = os.path.join(os.path.join(CondaDir, 'share'), 'proj')
os.environ["PROJ_LIB"] = ProjLib
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import to_rgba
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.ticker as mticker

# import PlotsConstants as Const

def createFigure(PlotConf):
    try:
        fig, ax = plt.subplots(1, 1, figsize = PlotConf["FigSize"])
    
    except:    
        print(f"⚠️  Failed to apply FigSize: {e}")
        fig, ax = plt.subplots(1, 1)

    return fig, ax

def saveFigure(fig, Path):
    Dir = os.path.dirname(Path)
    try:
        os.makedirs(Dir)
    except: pass
    fig.savefig(Path, dpi=150., bbox_inches='tight')

def prepareAxis(PlotConf, ax):
    for key in PlotConf:
        if key == "Title":
            ax.set_title(PlotConf["Title"])

        for axis in ["x", "y"]:
            if axis == "x":
                if key == axis + "Label":
                    ax.set_xlabel(PlotConf[axis + "Label"])

                if key == axis + "Ticks":
                    ax.set_xticks(PlotConf[axis + "Ticks"])

                if key == axis + "TicksLabels":
                    ax.set_xticklabels(PlotConf[axis + "TicksLabels"])
                
                if key == axis + "Lim":
                    ax.set_xlim(PlotConf[axis + "Lim"])

            if axis == "y":
                if key == axis + "Label":
                    ax.set_ylabel(PlotConf[axis + "Label"])

                if key == axis + "Ticks":
                    ax.set_yticks(PlotConf[axis + "Ticks"])

                if key == axis + "TicksLabels":
                    print("⚠️  Using legacy key 'TicksLabels'. Consider updating to 'TickLabels'.")
                    ax.set_yticklabels(PlotConf[axis + "TicksLabels"])

                if key == axis + "TickLabels":  # new - using to generate Rej Flag graphics
                    ax.set_yticklabels(PlotConf[axis + "TickLabels"])

                if key == axis + "Lim":
                    ax.set_ylim(PlotConf[axis + "Lim"])

        if key == "Grid" and PlotConf[key] == True:
            ax.grid(linestyle='--', linewidth=0.5, which='both')

def prepareColorBar(PlotConf, ax, Values):
    try:
        Min = PlotConf["ColorBarMin"]
    except:
        Mins = []
        for v in Values.values():
            Mins.append(min(v))
        Min = min(Mins)
    try:
        Max = PlotConf["ColorBarMax"]
    except:
        Maxs = []
        for v in Values.values():
            Maxs.append(max(v))
        Max = max(Maxs)
    normalize = mpl.cm.colors.Normalize(vmin=Min, vmax=Max)

    divider = make_axes_locatable(ax)
    # size size% of the plot and gap of pad% from the plot
    color_ax = divider.append_axes("right", size="3%", pad="2%")
    cmap = mpl.cm.get_cmap(PlotConf["ColorBar"])

    cbar = mpl.colorbar.ColorbarBase(
        color_ax,
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=Min, vmax=Max),
        label=PlotConf["ColorBarLabel"]
    )

    if "ColorBarTicks" in PlotConf:
        cbar.set_ticks(PlotConf["ColorBarTicks"])
        cbar.set_ticklabels([str(t) for t in PlotConf["ColorBarTicks"]])


    return normalize, cmap

def drawMap(PlotConf, ax,):
    Map = Basemap(projection = 'cyl',
    llcrnrlat  = PlotConf["LatMin"]-0,
    urcrnrlat  = PlotConf["LatMax"]+0,
    llcrnrlon  = PlotConf["LonMin"]-0,
    urcrnrlon  = PlotConf["LonMax"]+0,
    lat_ts     = 10,
    resolution = 'l',
    ax         = ax)

    # Draw map meridians
    Map.drawmeridians(
    np.arange(PlotConf["LonMin"],PlotConf["LonMax"]+1,PlotConf["LonStep"]),
    labels = [0,0,0,1],
    fontsize = 6,
    linewidth=0.2)
        
    # Draw map parallels
    Map.drawparallels(
    np.arange(PlotConf["LatMin"],PlotConf["LatMax"]+1,PlotConf["LatStep"]),
    labels = [1,0,0,0],
    fontsize = 6,
    linewidth=0.2)

    # Draw coastlines
    Map.drawcoastlines(linewidth=0.5)

    # Draw countries
    Map.drawcountries(linewidth=0.25)

def generateLinesPlot(PlotConf):    
    LineWidth = 1.5
    cmap = None
    normalize = None

    fig, ax = createFigure(PlotConf)
    prepareAxis(PlotConf, ax)

    if PlotConf.get("Map", False):
        drawMap(PlotConf, ax)

    if PlotConf.get("FormatYAxisFloat", False):
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    # Configure colormap
    if "ColorBar" in PlotConf:
        if "ColorBarTicks" in PlotConf:
            bounds = PlotConf["ColorBarTicks"]
            normalize = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds)-1)
            cmap = mpl.cm.get_cmap(PlotConf.get("ColorBar", "gist_ncar"), len(bounds)-2)

            # Discret lateral colorbar
            color_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = mpl.colorbar.ColorbarBase(
                color_ax,
                cmap=cmap,
                norm=normalize,
                label=PlotConf.get("ColorBarLabel", "PRN"),
                ticks=bounds
            )
            cbar.ax.tick_params(labelsize=8)
        elif PlotConf.get("ColorBar") not in [None, "", False]:
            normalize, cmap = prepareColorBar(PlotConf, ax, PlotConf["zData"])
        # else:
        #     normalize, cmap = prepareColorBar(PlotConf, ax, PlotConf["zData"])

    if PlotConf.get("FormatYAxisPlain", False): # Condition to format Y axis plainly
        formatter = mticker.ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        ax.get_yaxis().get_offset_text().set_visible(False)

    for Label in PlotConf["yData"].keys():
        x = np.array(PlotConf["xData"][Label])
        y = np.array(PlotConf["yData"][Label])
        z = np.array(PlotConf["zData"][Label])

        if "ColorBar" in PlotConf and "StatusData" in PlotConf:
            status = np.array(PlotConf["StatusData"][Label])
            colors = cmap(normalize(z)) if cmap and normalize else 'black'

            valid_mask = status == 1
            reject_mask = status == 0

            ax.scatter(
                x[valid_mask],
                y[valid_mask],
                marker=PlotConf.get("Marker", 'o'),
                s=PlotConf.get("MarkerSize", 40),
                c=colors[valid_mask] if isinstance(colors, np.ndarray) else colors,
                edgecolors='none',
                zorder=2
            )

            ax.scatter(
                x[reject_mask],
                y[reject_mask],
                marker='o',
                facecolors='none',
                edgecolors='darkgray',
                s=PlotConf.get("MarkerSize", 10),
                linewidths=0.8,
                zorder=3
            )

        elif "ColorBar" in PlotConf:
            first = z[0] if hasattr(z, "__getitem__") and len(z) > 0 else None
            if isinstance(first, (list, tuple, np.ndarray)):
                ax.scatter(x, y, marker=PlotConf["Marker"], s=PlotConf.get("MarkerSize", 5), c=z)
            else:
                colors = cmap(normalize(z)) if cmap and normalize else 'black'
                ax.scatter(x, y, marker=PlotConf["Marker"], s=PlotConf.get("MarkerSize", 5), c=colors)

        else:
            color = PlotConf.get("Colors", {}).get(Label, PlotConf.get("Color", None))
            ax.plot(x, y, linestyle='-', linewidth=LineWidth, label=Label, color=color)

    if "LegendLoc" in PlotConf:
        ax.legend(loc=PlotConf["LegendLoc"])

    if "Labels" in PlotConf:
        for _, row in PlotConf["Labels"].iterrows():
            prn = row["PRN"]
            rej = row["REJ"]
            sod = row["#SOD"]
            if cmap and normalize:
                color = cmap(normalize(prn))
            else:
                color = 'black'
            ax.text(
                sod, rej, f"G{prn:02d}", fontsize=5, ha='center', va='bottom',
                rotation=45, zorder=4, color=color,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5)
            )

    fig.canvas.draw()
    saveFigure(fig, PlotConf["Path"])
    plt.close(fig)

    if PlotConf.get("ReturnAx", False):
        return ax


def generatePolarPlot(PlotConf):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=PlotConf.get("FigSize", (8, 8)))

    theta = np.array(PlotConf["xData"]["POLAR"])  # azimute in radianos
    r = np.array(PlotConf["yData"]["POLAR"])      # radius = 90 - elevation
    prn = np.array(PlotConf["zData"]["POLAR"])    # PRN 

    vmin = PlotConf.get("ColorBarMin", 1)
    vmax = PlotConf.get("ColorBarMax", 32)
    cmap = mpl.cm.get_cmap("gist_ncar")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(prn))

    # Plot
    ax.scatter(theta, r, c=colors, s=6, alpha=0.7, marker='o', edgecolors='none')

    # Setting N, E, S and W axis
    ax.set_theta_zero_location("N")  # 90° at top
    ax.set_theta_direction(-1)        # clockwise 
    ax.set_rlim(90, 0)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])
    ax.grid(PlotConf.get("Grid", True))
    ax.set_title(PlotConf.get("Title", ""), fontsize=10)

    # Lateral colorbar

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax
    )
    cbar.set_label(PlotConf.get("ColorBarLabel", "Z"))
    cbar.ax.tick_params(labelsize=8)

    fig.canvas.draw()
    saveFigure(fig, PlotConf["Path"])
    plt.close(fig)

    if PlotConf.get("ReturnAx", False):
        return ax
    

def generatePlot(PlotConf):
    if (PlotConf["Type"] == "Lines"):
        return generateLinesPlot(PlotConf)
    elif PlotConf["Type"] == "Polar":
        return generatePolarPlot(PlotConf)



