# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:nomarker
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Results reading

import matplotlib.pyplot as plt
import pandas as pd
from data_manip.extraction.telemac_file import TelemacFile
from IPython.display import display, clear_output


def show_video(i):
    telemac_file = TelemacFile(f"results/results_{i}.slf")
    poly_coord, abs_curv, values_polylines = (
        telemac_file.get_timeseries_on_polyline(
            "WATER DEPTH", poly_points, poly_number
        )
    )
    telemac_file.close()
    x_length, y_length = values_polylines.shape
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    display(fig)
    
    x_data = np.linspace(0,12,x_length)
    y_data = values_polylines[:,0]
    
    
    # Your data-generating loop
    for frame in range(y_length):
        y_data = values_polylines[:,frame]
        line.set_data(x_data, y_data)
        ax.relim() 
        ax.autoscale_view(True,True,True)
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
        _, _, values_polylines = (
            telemac_file.get_timeseries_on_polyline(
                "FROUDE NUMBER", poly_points, poly_number
            )
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
            linewidth= 0.75,
            label=f"Case {i}: $y_{{normal}} = {yn:.3f}$ / m",
        )
        ax.axhline(
            y=yc,
            linestyle="--",
            color="tab:red",
            linewidth= 0.75,
            label=f"Case {i}: $y_{{critical}} = {yc:.3f}$ / m",
        )
        ax.plot(
            abs_curv[:],
            values_polylines[:, -1],
            color="tab:blue",
            linewidth= 1,
            label=f"Case {i}: $h_{{outlet}}=0.02$ / m",
        )
        # Plot triangles based on conditions
        if froude_number < 1:
            x_triangle = 12
        else:
            x_triangle = 0
        y_triangle = parameters_df.loc[i, "H0"]
        
        ax.scatter(
            [x_triangle],  # x coordinates
            [y_triangle],  # y coordinates
            marker=7,  # triangle marker
            color='tab:blue',  # color of the marker
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
    plot_results([11], parameters_df)
