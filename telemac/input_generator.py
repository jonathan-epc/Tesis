# ## Imports

from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import fsolve
from scipy.stats import qmc
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
    """
    Generates a TELEMAC-2D steering file with user-provided parameters.

    Args:
        geometry_file (str): Path to the geometry file.
        boundary_file (str): Path to the boundary conditions file.
        results_file (str): Path to the results file.
        title (str): Title for the simulation.
        duration (float): Simulation duration in seconds.
        time_step (float): Time step for the simulation in seconds.
        initial_depth (float): Initial water depth for the simulation.
        prescribed_flowrates (list, optional): List of prescribed flow rates
            at boundaries (default: None).
        prescribed_elevations (list, optional): List of prescribed water
            elevations at boundaries (default: None).
        friction_coefficient (float, optional): Bottom friction coefficient
            (default: 0.0025).

    Returns:
        str: The complete steering file content as a string.
    """
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


def all_combinations(n,
    SLOPE_min, SLOPE_max, n_min, n_max, Q0_min, Q0_max, H0_min, H0_max, BOTTOM_values
):
    """
    Generates a DataFrame with all combinations of parameters within given ranges.

    Parameters:
    n (int): Number of linearly spaced values for each parameter.
    SLOPE_min (float): Minimum value for S.
    SLOPE_max (float): Maximum value for S.
    n_min (float): Minimum value for n.
    n_max (float): Maximum value for n.
    Q0_min (float): Minimum value for Q.
    Q0_max (float): Maximum value for Q.
    H0_min (float): Minimum value for H0.
    H0_max (float): Maximum value for H0.
    BOTTOM_values (List[float]): List of specific BOTTOM values.

    Returns:
    pd.DataFrame: DataFrame containing all combinations of the parameters.
    """
    # Arrays for each column
    SLOPE_values = np.linspace(SLOPE_min, SLOPE_max, n)
    SLOPE_indices = range(len(SLOPE_values))
    n_values = np.linspace(n_min, n_max, n)
    Q0_values = np.linspace(Q0_min, Q0_max, n)
    H0_values = np.linspace(H0_min, H0_max, n)
    
    combinations = [
        [SLOPE_i, n, Q0, H0, BOTTOM]
        for (SLOPE_i, n, Q0, H0, BOTTOM) in product(
            SLOPE_indices, n_values, Q0_values, H0_values, BOTTOM_values
        )
    ]
    column_names = ["SLOPE_index", "n", "Q0", "H0", "BOTTOM"]
    df = pd.DataFrame(
        combinations,
        columns=column_names,
    )
    return df


def sample_combinations(n, SLOPE_min, SLOPE_max, n_min, n_max, Q0_min, Q0_max, H0_min, H0_max, BOTTOM_values):
    """
    Generate a DataFrame of sample combinations using Latin Hypercube Sampling (LHS).

    Parameters:
    - n (int): Number of samples to generate.
    - SLOPE_min (float): Minimum value for S.
    - SLOPE_max (float): Maximum value for S.
    - n_min (float): Minimum value for n.
    - n_max (float): Maximum value for n.
    - Q0_min (float): Minimum value for Q.
    - Q0_max (float): Maximum value for Q.
    - H0_min (float): Minimum value for H0.
    - H0_max (float): Maximum value for H0.
    - BOTTOM_values (list): List of values for BOTTOM to be combined with each sample.

    Returns:
    - pd.DataFrame: DataFrame containing the sample combinations with columns ["SLOPE", "n", "Q0", "H0", "BOTTOM"].
    """
    # Initialize Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=4, strength=2, seed=1618)
    sample = sampler.random(n=n, )
    
    # Define lower and upper bounds
    lower_bounds = [SLOPE_min, n_min, Q0_min, H0_min]
    upper_bounds = [SLOPE_max, n_max, Q0_max, H0_max]
    
    # Scale the samples to the defined bounds
    sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
    
    # Generate combinations
    combinations = []
    for combination in sample_scaled:
        for bottom in BOTTOM_values:
            combinations.append(combination.tolist() + [bottom])
    
    # Create DataFrame
    column_names = ["SLOPE", "n", "Q0", "H0", "BOTTOM"]
    df = pd.DataFrame(combinations, columns=column_names)
    
    return df


def generate_geometry(idx, SLOPE, ds, x, y, xg, yg, num_points_x, num_points_y, channel_length):
    """
    Generate geometry with random noise and save it to a file.

    Parameters:
    - idx (int): Index to identify the geometry file.
    - SLOPE (float): Slope value for the geometry.
    - ds (xarray.Dataset): Dataset containing the geometry data.
    - x (array-like): X-coordinates for interpolation.
    - y (array-like): Y-coordinates for interpolation.
    - xg (array-like): Grid X-coordinates for the random noise.
    - yg (array-like): Grid Y-coordinates for the random noise.
    - num_points_x (int): Number of points in the X direction for the random noise grid.
    - num_points_y (int): Number of points in the Y direction for the random noise grid.
    - channel_length (float): Length of the channel.

    Returns:
    - None: The function saves the generated geometry to a file and does not return a value.
    """
    # Define parameters for the random noise
    min_value = 0
    max_value = 0.15
    sigma = 0.95
    
    # Generate random noise and scale it
    random_noise = np.random.rand(num_points_y, num_points_x)
    scaled_random_noise = min_value + (random_noise * (max_value - min_value))
    
    # Smooth the random noise using a Gaussian filter
    smoothed_random_noise = gaussian_filter(scaled_random_noise, sigma=sigma)
    
    # Create an interpolator for the smoothed noise
    interpolator = RegularGridInterpolator(
        (yg, xg),
        smoothed_random_noise,
        bounds_error=False,
        fill_value=None,
    )
    
    # Compute the slope and noise values for the Z dimension
    z_slope = SLOPE * (channel_length - ds["x"].values)
    z_noise = interpolator((y, x))
    
    # Combine slope and noise to get the final Z values
    z = z_slope + z_noise
    z_right = z[num_points_x-1::num_points_x].max()
    z_left = z[0::num_points_x].max()
    # Update the dataset with the new Z values
    ds["B"].values = z.reshape(1, ds.y.shape[0])
    
    # Save the dataset to a file
    ds.selafin.write(f"geometry/geometry_3x3_NOISE_{idx}.slf")
    return z_left, z_right


# Define dimensions
channel_width = 0.3  # in m
channel_length = 12  # in m
channel_depth = 0.3  # in m
slope = 5 / 100  
wall_thickness = 0.00  # in m
flat_zone = 0
# Adjusting channel dimensions for walls
channel_width += 2 * wall_thickness
channel_length += 2 * wall_thickness
# Generate base mesh for the channel
num_points_y = 11  # Adjust as needed for resolution
num_points_x = 401

# ## Steering file generation

SLOPE_min, SLOPE_max = 3e-6, 1e-1
n_min, n_max = 1e-3, 2e-1
Q0_min, Q0_max = 5e-3, 2e-2
H0_min, H0_max = 1e-2, 3e-2

BOTTOM_values = ["NOISE"]

# Load old parameters from CSV if it exists
try:
    old_parameters_df = pd.read_csv("parameters.csv")
except FileNotFoundError:
    old_parameters_df = pd.DataFrame(
        columns=[
            "SLOPE",
            "n",
            "Q0",
            "H0",
            "BOTTOM",
            "yn",
            "yc",
            "subcritical",
            "direction",
        ]
    )

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
    column_names = ["SLOPE", "n", "Q0", "H0", "BOTTOM"]
    new_parameters_df = pd.DataFrame(
            combinations,
            columns=column_names,
    )
else:
    new_parameters_df = sample_combinations(5**2, SLOPE_min, SLOPE_max, n_min, n_max, Q0_min, Q0_max, H0_min, H0_max, BOTTOM_values)

# Calculate additional parameters for new entries
new_parameters_df["yn"] = normal_depth_simple(
    new_parameters_df["Q0"], 0.3, new_parameters_df["SLOPE"], new_parameters_df["n"]
)
new_parameters_df["yc"] = critical_depth_simple(new_parameters_df["Q0"], 0.3)
new_parameters_df["subcritical"] = new_parameters_df["yn"] > new_parameters_df["yc"]
new_parameters_df["direction"] = new_parameters_df["H0"] > new_parameters_df["yc"]
new_parameters_df["direction"] = new_parameters_df["direction"].apply(
    lambda x: "Right to left" if x else "Left to right"
)

# Combine old and new parameters
combined_parameters_df = pd.concat([old_parameters_df,new_parameters_df], ignore_index=True)

# Write to CSV
combined_parameters_df.to_csv("parameters.csv", index=True, index_label = "id")

parameters_df = pd.read_csv("parameters.csv", index_col="id")

ds = xr.open_dataset("geometry/mesh_3x3.slf", engine="selafin")
x = ds["x"].values
y = ds["y"].values
xg = np.linspace(0, channel_length, num_points_x)
yg = np.linspace(0, channel_width, num_points_y)

# Generate steering files
for index, case in tqdm(parameters_df.iterrows(), total=len(parameters_df)):
    z_left, z_right = generate_geometry(index, case["SLOPE"], ds, x, y, xg, yg, num_points_x, num_points_y, channel_length)
    boundary_file = (
        "boundary/boundary_3x3_tor.cli"
        if case["direction"] == "Left to right"
        else "boundary/boundary_3x3_riv.cli"
    )
    prescribed_elevations = (
        (0.0, z_left + case["H0"])
        if case["direction"] == "Left to right"
        else (z_right + case["H0"], 0.0)
    )
    steering_file_content = generate_steering_file(
        geometry_file=f"geometry/geometry_3x3_{case['BOTTOM']}_{index}.slf",
        boundary_file=boundary_file,
        results_file=f"results/results_{index}.slf",
        title=f"Caso {index}",
        duration=120,
        time_step=0.02,
        initial_depth=case["H0"],
        prescribed_flowrates=(0.0, case["Q0"]),
        prescribed_elevations=prescribed_elevations,
        friction_coefficient=case["n"],
    )

    # Write the steering file content to a file
    with open(f"steering_{index}.cas", "w") as f:
        f.write(steering_file_content)
