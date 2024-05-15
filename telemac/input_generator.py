# ## Imports

from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
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
VARIABLES FOR GRAPHIC PRINTOUTS = 'U,V,H,S,B,F,Q'
NUMBER OF PRIVATE ARRAYS        = 6
/
GRAPHIC PRINTOUT PERIOD         = 100
LISTING PRINTOUT PERIOD         = 100
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
TREATMENT OF THE LINEAR SYSTEM  = 2
/
DISCRETIZATIONS IN SPACE        = 11 ; 11
/
SOLVER                          = 1
SOLVER ACCURACY                 = 1.E-8
/
FREE SURFACE GRADIENT COMPATIBILITY = 0.9

SCHEME FOR ADVECTION OF VELOCITIES : 1 
SCHEME FOR ADVECTION OF TRACERS : 5
SCHEME FOR ADVECTION OF K-EPSILON : 4"""

    return steering_text


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

# ## Steering file generation

# Arrays for each column
S_values = np.linspace(1e-3, 50e-3, 5)
S_indices = range(len(S_values))
n_values = np.linspace(5e-3, 5e-1, 5)
Q_values = np.linspace(0.001, 0.040, 5)
H0_values = np.linspace(0.01, 0.30, 5)
BOTTOM_values = ["FLAT", "NOISE"]

# Load old parameters from CSV if it exists
try:
    old_parameters_df = pd.read_csv("parameters.csv")
    max_id = old_parameters_df["id"].max() + 1
except FileNotFoundError:
    old_parameters_df = pd.DataFrame(
        columns=[
            "id",
            "S_index",
            "n",
            "Q",
            "H0",
            "BOTTOM",
            "S",
            "yn",
            "yc",
            "subcritical",
            "direction",
        ]
    )
    max_id = 0

testing = False

# Generate new parameter combinations
if testing:
    new_parameter_combinations = [
        (0, 4, 0.055, 2e-2, 0.1, "FLAT"),
        (1, 4, 0.055, 2e-2, 0.08, "FLAT"),
        (2, 4, 0.055, 2e-2, 0.06, "FLAT"),
        (3, 4, 0.035, 2e-2, 0.08, "FLAT"),
        (4, 4, 0.035, 2e-2, 0.07, "FLAT"),
        (5, 4, 0.035, 2e-2, 0.05, "FLAT"),
        (6, 4, 0.055, 2e-2, 0.1, "NOISE"),
        (7, 4, 0.055, 2e-2, 0.08, "NOISE"),
        (8, 4, 0.055, 2e-2, 0.06, "NOISE"),
        (9, 4, 0.035, 2e-2, 0.08, "NOISE"),
        (10, 4, 0.035, 2e-2, 0.07, "NOISE"),
        (11, 4, 0.035, 2e-2, 0.05, "NOISE"),
    ]
else:
    new_parameter_combinations = [
        (max_id + index, S_i, n, Q, H0, BOTTOM)
        for index, (S_i, n, Q, H0, BOTTOM) in enumerate(
            product(range(len(S_values)), n_values, Q_values, H0_values, BOTTOM_values)
        )
    ]

# Create DataFrame for new parameters
new_parameters_df = pd.DataFrame(
    new_parameter_combinations, columns=["id", "S_index", "n", "Q", "H0", "BOTTOM"]
)

# Calculate additional parameters for new entries
new_parameters_df["S"] = S_values[new_parameters_df["S_index"]]
new_parameters_df["yn"] = normal_depth_simple(
    new_parameters_df["Q"], 0.3, new_parameters_df["S"], new_parameters_df["n"]
)
new_parameters_df["yc"] = critical_depth_simple(new_parameters_df["Q"], 0.3)
new_parameters_df["subcritical"] = new_parameters_df["yn"] > new_parameters_df["yc"]
new_parameters_df["direction"] = new_parameters_df["H0"] > new_parameters_df["yc"]
new_parameters_df["direction"] = new_parameters_df["direction"].apply(
    lambda x: "Right to left" if x else "Left to right"
)

# Combine old and new parameters
combined_parameters_df = pd.concat(
    [old_parameters_df, new_parameters_df], ignore_index=True
)

# Write to CSV
combined_parameters_df.to_csv("parameters.csv", index=False)

parameters_df = pd.read_csv("parameters.csv", index_col="id")

# Generate steering files
for index, case in tqdm(parameters_df.iterrows(), total=len(parameters_df)):
    boundary_file = (
        "boundary/boundary_3x3_tor.cli"
        if case["direction"] == "Left to right"
        else "boundary/boundary_3x3_riv.cli"
    )
    prescribed_elevations = (
        (0.0, case["S"] * channel_length + case["H0"])
        if case["direction"] == "Left to right"
        else (case["H0"], 0.0)
    )

    steering_file_content = generate_steering_file(
        geometry_file=f"geometry/geometry_3x3_{case['BOTTOM']}_{case['S_index']}.slf",
        boundary_file=boundary_file,
        results_file=f"results/results_{index}.slf",
        title=f"Caso {index}",
        duration=120,
        time_step=0.02,
        initial_depth=case["H0"],
        prescribed_flowrates=(0.0, case["Q"]),
        prescribed_elevations=prescribed_elevations,
        friction_coefficient=case["n"],
    )

    # Write the steering file content to a file
    with open(f"steering_{index}.cas", "w") as f:
        f.write(steering_file_content)

# ## Geometry generation

min_value = 0
max_value = 0.03
sigma = 1.5
ds = xr.open_dataset("geometry/mesh_3x3.slf", engine="selafin")
x = ds["x"].values
y = ds["y"].values
xg = np.linspace(0, channel_length, num_points_x)
yg = np.linspace(0, channel_width, num_points_y)

for i in tqdm(range(len(S_values))):
    random_noise = np.random.rand(num_points_y, num_points_x)
    scaled_random_noise = min_value + (random_noise * (max_value - min_value))
    smoothed_random_noise = gaussian_filter(scaled_random_noise, sigma=sigma)
    interpolator = RegularGridInterpolator(
        (
            yg,
            xg,
        ),
        smoothed_random_noise,
        bounds_error=False,
        fill_value=None,
    )
    z_slope = S_values[i] * (channel_length - ds["x"].values)
    z_noise = interpolator((y, x))
    z = z_slope + z_noise
    ds["B"].values = z_slope.reshape(1, ds.y.shape[0])
    ds.selafin.write(f"geometry/geometry_3x3_{'FLAT'}_{i}.slf")
    ds["B"].values = z.reshape(1, ds.y.shape[0])
    ds.selafin.write(f"geometry/geometry_3x3_{'NOISE'}_{i}.slf")
