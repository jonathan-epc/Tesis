# ## Imports

# Standard Library Imports
import math
from itertools import product

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Third-party Library Imports
import numpy as np
import pandas as pd
import xarray as xr

# Local Imports
from data_manip.extraction.telemac_file import TelemacFile
from matplotlib import cm, ticker
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from postel.plot1d import plot1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import fsolve
from tqdm.autonotebook import tqdm


# ## Functions

def manning_equation(depth, flow_rate, bottom_width, slope, roughness_coefficient):
    """
    Calculate the depth of flow using Manning's equation.

    Parameters:
        depth (float): Depth of flow (meters).
        flow_rate (float): Flow rate (cubic meters per second).
        bottom_width (float): Bottom width of the channel (meters).
        slope (float): Slope of the channel.
        roughness_coefficient (float): Manning's roughness coefficient.

    Returns:
        float: Depth of flow (meters).
    """
    area = depth * bottom_width
    wetted_perimeter = bottom_width + 2 * depth * 0
    hydraulic_radius = area / wetted_perimeter
    return (
        flow_rate
        - area * hydraulic_radius ** (2 / 3) * slope ** (1 / 2) / roughness_coefficient
    )


@np.vectorize
def froude_number(depth, flow_rate, top_width):
    """
    Calculate the Froude number.

    Parameters:
        depth (float): Depth of flow (meters).
        flow_rate (float): Flow rate (cubic meters per second).
        top_width (float): Top width of the channel (meters).

    Returns:
        float: Froude number.
    """
    gravity = 9.81
    area = top_width * depth
    numerator = flow_rate**2 * top_width
    denominator = area**3 * gravity
    froude_number = (numerator / denominator) ** 0.5
    return froude_number - 1


@np.vectorize
def normal_depth(flow_rate, bottom_width, slope, roughness_coefficient):
    """
    Calculate the normal depth of flow using Manning's equation and numerical methods.

    Parameters:
        flow_rate (float or array_like): Flow rate (cubic meters per second).
        bottom_width (float or array_like): Bottom width of the channel (meters).
        slope (float): Slope of the channel.
        roughness_coefficient (float): Manning's roughness coefficient.

    Returns:
        float or ndarray: Normal depth of flow (meters).
    """
    solution = fsolve(
        manning_equation,
        x0=1,
        args=(flow_rate, bottom_width, slope, roughness_coefficient),
    )[0]
    return solution


@np.vectorize
def critical_depth(flow_rate, bottom_width):
    """
    Calculate the critical depth of flow using Froude's equation and numerical methods.

    Parameters:
        flow_rate (float or array_like): Flow rate (cubic meters per second).
        bottom_width (float or array_like): Bottom width of the channel (meters).

    Returns:
        float or ndarray: Critical depth of flow (meters).
    """
    solution = fsolve(froude_number, x0=0.001, args=(flow_rate, bottom_width))[0]
    return solution


def critical_slope(flow_rate, bottom_width, roughness_coefficient):
    def eq(slope, flow_rate_bottom_width, roughness_coefficient):
        return normal_depth(
            flow_rate, bottom_width, slope, roughness_coefficient
        ) - critical_depth(flow_rate, bottom_width)

    solution = fsolve(
        eq, x0=1e-10, args=(flow_rate, bottom_width, roughness_coefficient)
    )
    return solution


def normal_depth_simple(flow_rate, bottom_width, slope, roughness_coefficient):
    """
    Calculate the normal depth of flow using a simplified form of Manning's equation.

    Parameters:
        flow_rate (float): Flow rate (cubic meters per second).
        bottom_width (float): Bottom width of the channel (meters).
        slope (float): Slope of the channel.
        roughness_coefficient (float): Manning's roughness coefficient.

    Returns:
        float: Normal depth of flow (meters).
    """
    return (roughness_coefficient * flow_rate / bottom_width * slope ** (-1 / 2)) ** (
        3 / 5
    )


def critical_depth_simple(flow_rate, top_width):
    """
    Calculate the critical depth of flow using a simplified form of Froude's equation.

    Parameters:
        flow_rate (float): Flow rate (cubic meters per second).
        top_width (float): Top width of the channel (meters).

    Returns:
        float: Critical depth of flow (meters).
    """
    return (flow_rate / top_width) ** (2 / 3) * 9.81 ** (-1 / 3)


def critical_slope_simple(flow_rate, bottom_width, roughness_coefficient):
    """
    Calculate the critical slope of flow using a simplified form

    Parameters:
        flow_rate (float): Flow rate (cubic meters per second).
        bottom_width (float): Bottom width of the channel (meters).
        roughness_coefficient (float): Manning's roughness coefficient.

    Returns:
        float: Critical slope of flow.
    """
    return (
        (bottom_width / flow_rate) ** (2 / 9)
        * 9.81 ** (10 / 9)
        * roughness_coefficient**2
    )


def plot_critical_slope(x, y, slopes):
    """
    Plot the critical slope.

    Parameters:
        x (ndarray): Array of x values.
        y (ndarray): Array of y values.
        slopes (ndarray): Array of slope values.
    """
    # Create a grid of x, y values
    X, Y = np.meshgrid(x, y)

    # Calculate the critical slope
    F = critical_slope_simple(X, 0.3, Y)

    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Plot the filled contour plot
    cf = ax.pcolormesh(X, Y, F, cmap=cm.turbo, antialiased=True)

    # Plot the contour lines
    cl = ax.contour(X, Y, F, levels=slopes, linewidths=0.75, colors="black")

    # Label each contour line with its corresponding slope value
    ax.clabel(
        cl, inline=True, fontsize=10, fmt=FuncFormatter(lambda x, _: f"{x*100:.1f}%")
    )

    # Add a colorbar
    cbar = fig.colorbar(cf, extend="both")
    cbar.set_label("Critical Slope")

    # Set scientific notation for the x-axis
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits([0.001, 0.04])
    ax.xaxis.set_major_formatter(formatter)

    # Set labels and grid
    ax.set_xlabel("Flow Rate / $(m^3s^{-1})$")
    ax.set_ylabel("Roughness coefficient / $(sm^{-1/3})$")
    ax.grid(
        True, which="both", linestyle="-", linewidth=0.25, alpha=0.25, color="white"
    )
    ax.minorticks_on()

    # Show the plot
    plt.show()


# ## Parameter exploration and sensitivity analysis

# Define parameters
x = np.linspace(1 / 1000, 50 / 1000, 128)
y = np.linspace(0.010, 0.200, 128)
slopes = [5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]

# Plot the critical slope
plot_critical_slope(x, y, slopes)

# ## Bathymetry creation

# Define dimensions
channel_width = 0.3  # in m
channel_length = 12  # in m
channel_depth = 0.3  # in m
slope = 5 / 100  # 1% slope
wall_thickness = 0.00  # in m
flat_zone = 0
# Adjusting channel dimensions for walls
channel_width += 2 * wall_thickness
channel_length += 2 * wall_thickness
# Generate base mesh for the channel
num_points_y = 11  # Adjust as needed for resolution
num_points_x = 401

x = np.linspace(-wall_thickness * 0, channel_length + wall_thickness * 0, num_points_x)
y = np.linspace(-wall_thickness * 0, channel_width + wall_thickness * 0, num_points_y)
X, Y = np.meshgrid(x, y)
Z_slope = slope * channel_length - slope * X
# Z_slope[X <= flat_zone] = -slope * x[x<=flat_zone][-1] + slope * x.max()
plt.imshow(Z_slope)
plt.show()
plt.close()

S_values = np.linspace(1e-3, 50e-3, 5)

ds = xr.open_dataset("geometry/mesh_3x3.slf", engine="selafin")
S_values = np.linspace(1e-3, 50e-3, 5)
rng = np.random.default_rng()
sigma = 3  # Standard deviation for Gaussian blur
for i in tqdm(range(len(S_values))):
    Z_slope = slope * channel_length - slope * ds["x"].values
    # Z_blur = (gaussian_filter(rng.standard_normal(size=(1, ds.y.shape[0])), sigma=sigma)        * 0.15    )
    Z = Z_slope  # + Z_blur
    ds["B"].values = Z.reshape(1, ds.y.shape[0])
    ds.selafin.write(f"geometry/geometry_3x3_{i}.slf")


# ## Steering file generation

def generate_steering_file(
    geometry_file,
    boundary_file,
    results_file,
    title,
    duration,
    time_step,
    initial_depth,
    prescribed_flowrates,
    prescribed_elevations,
    friction_coefficient,
):
    steering_text = f"""/-------------------------------------------------------------------/
/                        TELEMAC-2D                                 /
/-------------------------------------------------------------------/
/
/----------------------------------------------
/  COMPUTER INFORMATIONS
/----------------------------------------------
/
GEOMETRY FILE                   = '{geometry_file}'
BOUNDARY CONDITIONS FILE        = '{boundary_file}'
RESULTS FILE                    = '{results_file}'
/
/----------------------------------------------
/  GENERAL INFORMATIONS - OUTPUTS
/----------------------------------------------
/
TITLE = '{title}'
/
VARIABLES FOR GRAPHIC PRINTOUTS = 'U,V,S,B,Q,F,H'
NUMBER OF PRIVATE ARRAYS        = 6
/
GRAPHIC PRINTOUT PERIOD         = 100000
LISTING PRINTOUT PERIOD         = 100000
/
DURATION                        = {duration}
/TIME STEP                       = {time_step}
VARIABLE TIME-STEP              = YES
DESIRED COURANT NUMBER          = 0.8
MASS-BALANCE                    = YES
/STOP IF A STEADY STATE IS REACHED = YES
/STOP CRITERIA                   = 1e-8; 1e-8; 1e-8
/
/----------------------------------------------
/  INITIAL CONDITIONS
/----------------------------------------------
/
INITIAL CONDITIONS               = 'CONSTANT DEPTH'
INITIAL DEPTH                    = {initial_depth}
/
/----------------------------------------------
/  BOUNDARY CONDITIONS
/----------------------------------------------
/
PRESCRIBED FLOWRATES            =  {prescribed_flowrates[0]}  ;  {prescribed_flowrates[1]}
PRESCRIBED ELEVATIONS           = {prescribed_elevations[0]} ; {prescribed_elevations[1]}
VELOCITY PROFILES               =  1    ;  4
/
/----------------------------------------------
/  PHYSICAL PARAMETERS
/----------------------------------------------
/
LAW OF BOTTOM FRICTION          = 4
FRICTION COEFFICIENT            = {friction_coefficient}
TURBULENCE MODEL                = 1
/
/----------------------------------------------
/  NUMERICAL PARAMETERS
/----------------------------------------------
/SCHEMES
EQUATIONS                       = 'SAINT-VENANT FV'
TREATMENT OF THE LINEAR SYSTEM  = 2 /1:PRIM 2:WAVE EQUATION
/
DISCRETIZATIONS IN SPACE        = 11 ; 11
/
SOLVER                          = 1
SOLVER ACCURACY                 = 1.E-7
/
TIDAL FLATS                     = NO
FREE SURFACE GRADIENT COMPATIBILITY = 0.9

SCHEME FOR ADVECTION OF VELOCITIES : 1 
SCHEME FOR ADVECTION OF TRACERS : 5
SCHEME FOR ADVECTION OF K-EPSILON : 4"""

    return steering_text


# Arrays for each column
S_indices = range(len(S_values))
n_values = np.linspace(5e-3, 5e-1, 5)
Q_values = np.linspace(0.001, 0.040, 5)
H0_values = np.linspace(0.01, 0.30, 5)
BOTTOM_values = ["FLAT"]

# Generate all combinations of values
combinations = list(product(S_indices, n_values, Q_values, H0_values, BOTTOM_values))

# Create DataFrame
parameters = pd.DataFrame(combinations, columns=["S_i", "n", "Q", "H0", "BOTTOM"])


parameters["S"] = S_values[parameters["S_i"]]
parameters["yn"] = normal_depth_simple(
    parameters["Q"], 0.3, parameters["S"], parameters["n"]
)
parameters["yc"] = critical_depth_simple(parameters["Q"], 0.3)
parameters["subcritical"] = parameters["yn"] > parameters["yc"]
parameters["R2L"] = parameters["H0"] < parameters["yc"]

parameters.to_csv("parameters.csv", index=True, index_label="id")

for index, case in tqdm(parameters.iterrows(), total=len(parameters)):
    if case["R2L"]:
        steering_file_content = generate_steering_file(
            geometry_file=f"geometry/geometry_3x3_{case['S_i']}.slf",
            boundary_file="boundary/boundary_3x3_tor.cli",
            results_file=f"results/results_{index}.slf",
            title=f"Caso {index}",
            duration=1200,
            time_step=0.02,
            initial_depth=case["H0"],
            prescribed_flowrates=(0.0, case["Q"]),
            prescribed_elevations=(0.0, 0.12 + case["H0"]),
            friction_coefficient=case["n"],
        )
    else:
        steering_file_content = generate_steering_file(
            geometry_file="geometry/geometry_3x3_0.slf",
            boundary_file="boundary/boundary_3x3_riv.cli",
            results_file=f"results/results_{index}.slf",
            title=f"Caso {index}",
            duration=600,
            time_step=0.02,
            initial_depth=0.06,
            prescribed_flowrates=(0.0, case["Q"]),
            prescribed_elevations=(case["H0"], 0.0),
            friction_coefficient=case["n"],
        )

    # Write the steering file content to a file
    with open(f"steering_{index}.cas", "w") as f:
        f.write(steering_file_content)

# ## Results reading

yn = normal_depth(
    flow_rate=0.01, bottom_width=0.3, slope=1 / 100, roughness_coefficient=0.04
)
yc = critical_depth(flow_rate=0.01, bottom_width=0.3)

parameters = pd.read_csv("parameters.csv", index_col="id")


def plot_ith(i):
    # List of points defining the polyline
    poly_points = [[0.0, 0.15], [12.0, 0.15]]
    # List of number of discretized points for each polyline segment
    poly_number = [1000]    
    res1 = TelemacFile(f"results/results_{i}.slf")
    poly_coord1, abs_curv1, values_polylines1 = res1.get_timeseries_on_polyline(
        "WATER DEPTH", poly_points, poly_number
    )
    res1.close()

    # Determine yn and yc based on the ith row of parametros DataFrame
    yn = parameters.loc[i, "yn"]
    yc = parameters.loc[i, "yc"]

    fig, ax = plt.subplots()
    ax.plot(
        abs_curv1[:],
        values_polylines1[:, -1],
        color="tab:blue",
        label="$h_{outlet}=0.02$ / m",
    )
    ax.axhline(
        y=yn, linestyle="--", color="tab:green", label=f"$y_{{normal}} = {yn:.3f}$ / m"
    )
    ax.axhline(
        y=yc, linestyle="--", color="tab:red", label=f"$y_{{critical}} = {yc:.3f}$ / m"
    )
    ax.set_xlabel("x / m")
    ax.set_ylabel("Water depth / m")
    ax.set_title(f"Case {i}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.125), ncol=3)
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.minorticks_on()
    # plt.savefig(f"test_plot_{i}.png")
    plt.show()


plot_ith(527)

res0 = TelemacFile("results/results_0.slf")
res1 = TelemacFile("results/results_1.slf")
res2 = TelemacFile("results/results_2.slf")
res3 = TelemacFile("results/results_3.slf")
res4 = TelemacFile("results/results_4.slf")
res5 = TelemacFile("results/results_5.slf")
# Getting water depth values over time for each discretized points of the polyline
poly_coord0, abs_curv0, values_polylines0 = res0.get_timeseries_on_polyline(
    "WATER DEPTH", poly_points, poly_number
)
poly_coord1, abs_curv1, values_polylines1 = res1.get_timeseries_on_polyline(
    "WATER DEPTH", poly_points, poly_number
)
poly_coord2, abs_curv2, values_polylines2 = res2.get_timeseries_on_polyline(
    "WATER DEPTH", poly_points, poly_number
)
poly_coord3, abs_curv3, values_polylines3 = res3.get_timeseries_on_polyline(
    "WATER DEPTH", poly_points, poly_number
)
poly_coord4, abs_curv4, values_polylines4 = res4.get_timeseries_on_polyline(
    "WATER DEPTH", poly_points, poly_number
)
poly_coord5, abs_curv5, values_polylines5 = res5.get_timeseries_on_polyline(
    "WATER DEPTH", poly_points, poly_number
)

res0.close()
res1.close()
res2.close()
res3.close()
res4.close()
res5.close()

# Create a figure and axis objects
fig, ax = plt.subplots()

# Plot the lines
ax.plot(abs_curv0[:], values_polylines0[:, -1], label="$h_{outlet}=0.01$ / m")
ax.plot(abs_curv1[:], values_polylines1[:, -1], label="$h_{outlet}=0.01$ / m")
ax.plot(abs_curv2[:], values_polylines2[:, -1], label="$h_{outlet}=0.06$ / m")
ax.plot(abs_curv3[:], values_polylines3[:, -1], label="$h_{outlet}=0.11$ / m")
ax.plot(abs_curv4[:], values_polylines4[:, -1], label="$h_{outlet}=0.01$ / m")
ax.plot(abs_curv5[:], values_polylines5[:, -1], label="$h_{outlet}=0.01$ / m")

ax.axhline(
    y=parameters.iloc[0]["yn"],
    linestyle="--",
    color="tab:orange",
    label=f"$y_{{normal}} = {yn:.3f}$ / m",
)
ax.axhline(
    y=parameters.iloc[0]["yc"],
    linestyle="--",
    color="tab:red",
    label=f"$y_{{critical}} = {yc:.3f}$ / m",
)

# Add labels and title
ax.set_xlabel("x / m")
ax.set_ylabel("Water depth / m")
ax.set_title("Test")

# Add legend
ax.legend()
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.grid(
    True, which="both", linestyle="-", linewidth=0.5
)  # Turn on gridlines for both major and minor ticks
ax.minorticks_on()  # Turn on minor ticks
plt.savefig("test_plot.png")
# Show the plot
plt.show()
