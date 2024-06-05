# ## Results reading

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.ticker as tkr
from data_manip.extraction.telemac_file import TelemacFile
from IPython.display import clear_output, display
from scipy.interpolate import griddata

from matplotlib import font_manager

font_files = font_manager.findSystemFonts()

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


def show_video(i):
    telemac_file = TelemacFile(f"results/results_{i}.slf")
    poly_coord, abs_curv, values_polylines = telemac_file.get_timeseries_on_polyline(
        "WATER DEPTH", poly_points, poly_number
    )
    telemac_file.close()
    x_length, y_length = values_polylines.shape
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [])
    display(fig)

    x_data = np.linspace(0, 12, x_length)
    y_data = values_polylines[:, 0]

    # Your data-generating loop
    for frame in range(y_length):
        y_data = values_polylines[:, frame]
        line.set_data(x_data, y_data)
        ax.relim()
        ax.autoscale_view(True, True, True)
        clear_output(wait=True)
        display(fig)

    plt.close()


def read_parameters(file_path):
    """
    Read parameters from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing parameters.

    Returns:
        pandas.DataFrame: DataFrame containing parameters with 'id' as index.
    """

    return pd.read_csv(file_path, index_col="id")


def plot_results(indices, parameters_df):
    """
    Plot water depth results for multiple cases on the same graph.

    This function reads Telemac files corresponding to each case index provided
    in the 'indices' list, extracts water depth data along a predefined polyline,
    and plots the water depth profiles along with critical and normal water levels.

    Parameters:
        indices (list): List of case indices to plot.
        parameters_df (pandas.DataFrame): DataFrame containing parameters,
            with 'id' as index, including 'yn' (normal water level) and 'yc'
            (critical water level) columns.

    Returns:
        None
    """
    # Define polyline points and segment discretization
    poly_points = [[0.0, 0.15], [12.0, 0.15]]
    poly_number = [1000]

    # Plotting
    fig, ax = plt.subplots()
    for i in indices:
        # Read Telemac file
        telemac_file = TelemacFile(f"results/results_{i}.slf")
        _, _, values_polylines = telemac_file.get_timeseries_on_polyline(
            "FROUDE NUMBER", poly_points, poly_number
        )
        froude_number = values_polylines.mean()
        poly_coord, abs_curv, values_polylines = (
            telemac_file.get_timeseries_on_polyline(
                "WATER DEPTH", poly_points, poly_number
            )
        )
        telemac_file.close()

        # Retrieve parameters for the given index
        yn = parameters_df.loc[i, "yn"]
        yc = parameters_df.loc[i, "yc"]

        # Plot
        ax.axhline(
            y=yn,
            linestyle="--",
            color="tab:green",
            linewidth=0.75,
            label=f"Case {i}: $y_{{normal}} = {yn:.3f}$ / m",
        )
        ax.axhline(
            y=yc,
            linestyle="--",
            color="tab:red",
            linewidth=0.75,
            label=f"Case {i}: $y_{{critical}} = {yc:.3f}$ / m",
        )
        ax.plot(
            abs_curv[:],
            values_polylines[:, -1],
            color="tab:blue",
            linewidth=1,
            label=f"Case {i}: $h_{{outlet}}=0.02$ / m",
        )
        # Plot triangles based on conditions
        if froude_number > 1:
            x_triangle = 12
        else:
            x_triangle = 0
        y_triangle = parameters_df.loc[i, "H0"]

        ax.scatter(
            [x_triangle],  # x coordinates
            [y_triangle],  # y coordinates
            marker=7,  # triangle marker
            color="tab:blue",  # color of the marker
            zorder=5,  # make sure it's on top of other elements
        )
    #    ax.set_xlim(0,12)
    #    ax.set_ylim(0,0.3)
    ax.set_xlabel("x / m")
    ax.set_ylabel("Water depth / m")
    ax.set_title("Results Comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.125), ncol=3)
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.minorticks_on()
    plt.show()

# Main entry point
if __name__ == "__main__":
    parameters_df = read_parameters("parameters.csv")


# def plot_results_2d(i):
#     num_points_y = 11
#     num_points_x = 401
#     ds = xr.open_dataset(f"results/results_{i}.slf", engine="selafin")
#     x = ds["x"].values
#     y = ds["y"].values
#     u = ds["U"][-1].values
#     v = ds["V"][-1].values
#     b = ds["B"][-1].values - parameters_df.loc[i, "S"]*(12 - x)
#     # xi = np.linspace(0, 12, num_points_x)
#     # yi = np.linspace(0, 0.3, num_points_y)
#     # ui = griddata((x, y), u, (xi[None, :], yi[:, None]), method="linear")
#     # vi = griddata((x, y), v, (xi[None, :], yi[:, None]), method="linear")
#     fig, ax = plt.subplots(nrows = 3, ncols=1,figsize = (15, 9), dpi=300)
#     # cntr = ax.tricontourf(x, y, v, levels=128, cmap='viridis')
#     cntr = ax[0].tricontourf(x, y, b, levels=128, cmap='terrain')
#     fig.colorbar(cntr, anchor = (-0.3, 0.5), label = 'Elevación / m')
#     cntr = ax[1].tricontourf(x, y, b, levels=128, cmap='terrain')
#     fig.colorbar(cntr, anchor = (-0.3, 0.5), label = 'Elevación / m')
#     cntr = ax[2].tricontourf(x, y, b, levels=128, cmap='terrain')
#     fig.colorbar(cntr, anchor = (-0.3, 0.5), label = 'Elevación / m')    
#     # fig.colorbar(cntr, anchor = (-0.3, 0.5), label = 'Velocidad en y / (m/s)')
#     # ax.tricontour(x, y, v, levels=4, linewidths=0.5, colors = 'k')
#     # ax.tricontour(x, y, b, levels=32, linewidths=0.5, colors = 'k')
#     # ax.streamplot(xi, yi, ui, vi, density=0.4, color='white', linewidth=0.75, broken_streamlines=False)
#     ax[0].set(xlim=(0, 12), ylim=(0, 0.3))
#     ax[0].set_xlabel("x / m")
#     ax[0].set_ylabel("y / m")
#     plt.tight_layout()
#     plt.savefig("img/figb.png", )
#     plt.show()    

def plot_results_2d(i):
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams.update({'font.size': 15})    
    num_points_y = 11
    num_points_x = 401
    ds = xr.open_dataset(f"results/results_{i}.slf", engine="selafin")
    x = ds["x"].values
    y = ds["y"].values
    u = ds["U"][-1].values
    v = ds["V"][-1].values
    b = ds["B"][-1].values - parameters_df.loc[i, "S"]*(12 - x)

    fig, ax = plt.subplots(nrows = 3, ncols=1,figsize = (15, 9), dpi=300)
    
    plt1 = ax[0].tricontourf(x, y, b, levels=64, cmap='terrain')
    fig.colorbar(plt1, anchor = (-0.3, 0.5), label = 'Elevación / m', format='%0.2f', ticks=np.arange(0.04,0.16,0.04))
    ax[0].set(xlim=(0, 6), ylim=(0, 0.3))
    ax[0].set_xlabel("x / m")
    ax[0].set_ylabel("y / m")
    
    plt2 = ax[1].tricontourf(x, y, u, levels=64, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(plt2, anchor = (-0.3, 0.5), label = 'Velocidad en x / (m/s) ', format='%0.2f', ticks=np.arange(0.15,1,0.3))
    ax[1].set(xlim=(0, 6), ylim=(0, 0.3))
    ax[1].set_xlabel("x / m")
    ax[1].set_ylabel("y / m")
    
    plt3 = ax[2].tricontourf(x, y, v, levels=64, cmap='viridis', vmin=-0.1, vmax=0.1, )
    fig.colorbar(plt3, anchor = (-0.3, 0.5), label = 'Velocidad y / (m/s)', format='%0.2f', ticks=np.arange(-0.12,0.12,0.08))
    ax[2].set(xlim=(0, 6), ylim=(0, 0.3))
    ax[2].set_xlabel("x / m")
    ax[2].set_ylabel("y / m")
    
    
    plt.suptitle("Terreno", x=0.43)
    plt.tight_layout()
    plt.savefig("img/figa.png", bbox_inches='tight')
    plt.show()    


def plot_results_2d(i):
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams.update({'font.size': 15})    
    num_points_y = 11
    num_points_x = 401
    ds = xr.open_dataset(f"results/results_{i}.slf", engine="selafin")
    x = ds["x"].values
    y = ds["y"].values
    u = ds["U"][-1].values
    v = ds["V"][-1].values
    b = ds["B"][-1].values - parameters_df.loc[i, "S"]*(12 - x)

    fig, ax = plt.subplots(nrows = 3, ncols=2,figsize = (15, 6), dpi=300)
    
    plt2 = ax[0].tricontourf(x, y, u, levels=64, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(plt2, anchor = (-0.3, 0.5), label = 'Velocidad en x / (m/s) ', format='%0.2f', ticks=np.arange(0.15,1,0.3))
    ax[0].set(xlim=(0, 6), ylim=(0, 0.3))
    ax[0].set_xlabel("x / m")
    ax[0].set_ylabel("y / m")
    
    plt3 = ax[1].tricontourf(x, y, v, levels=64, cmap='viridis', vmin=-0.1, vmax=0.1, )
    fig.colorbar(plt3, anchor = (-0.3, 0.5), label = 'Velocidad y / (m/s)', format='%0.2f', ticks=np.arange(-0.12,0.12,0.08))
    ax[1].set(xlim=(0, 6), ylim=(0, 0.3))
    ax[1].set_xlabel("x / m")
    ax[1].set_ylabel("y / m")
    
    plt.suptitle("Velocidad", x=0.43)
    plt.tight_layout()
    plt.savefig("img/figb.png", bbox_inches='tight')
    plt.show()    


def plot_results_2d(i):
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams.update({'font.size': 15})    
    num_points_y = 11
    num_points_x = 401
    ds = xr.open_dataset(f"results/results_{i}.slf", engine="selafin")
    x = ds["x"].values
    y = ds["y"].values
    u = ds["U"][-1].values
    v = ds["V"][-1].values
    b = ds["B"][-1].values - parameters_df.loc[i, "S"]*(12 - x)

    fig, ax = plt.subplots(nrows = 1, ncols=1,figsize = (15, 3), dpi=300)
    
    plt1 = ax.tricontourf(x, y, b, levels=64, cmap='terrain')
    fig.colorbar(plt1, anchor = (-0.3, 0.5), label = 'Elevación / m', format='%0.2f', ticks=np.arange(0.04,0.16,0.04))
    ax.set(xlim=(0, 6), ylim=(0, 0.3))
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
      
    plt.suptitle("Terreno", x=0.43)
    plt.tight_layout()
    plt.savefig("img/figa.png", bbox_inches='tight')
    plt.show()  


plot_results_2d(0)

plot_results([0], parameters_df)
