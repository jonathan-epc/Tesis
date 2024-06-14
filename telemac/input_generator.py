# ## Imports

from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from loguru import logger
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
        slope (float): Slope of the channel bed.
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
    """
    Calculate the critical slope of flow using numerical methods.

    Parameters:
        flow_rate (float): Flow rate (cubic meters per second).
        bottom_width (float): Bottom width of the channel (meters).
        roughness_coefficient (float): Manning's roughness coefficient.

    Returns:
        float: Critical slope of flow.
    """

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
    prescribed_flowrates=None,
    prescribed_elevations=None,
    friction_coefficient=0.0025,
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
    if prescribed_flowrates is None:
        prescribed_flowrates = [0.0, 0.0]
    if prescribed_elevations is None:
        prescribed_elevations = [0.0, 0.0]

    if len(prescribed_flowrates) < 2 or len(prescribed_elevations) < 2:
        raise ValueError(
            "prescribed_flowrates and prescribed_elevations must have at least two elements"
        )

    if duration < 0 or time_step < 0 or initial_depth < 0 or friction_coefficient < 0:
        raise ValueError(
            "duration, time_step, initial_depth, and friction_coefficient must be non-negative"
        )

    # Read the template text from the file
    with open("steering_template.txt", "r") as f:
        template_text = f.read()

    # Use the template text to generate the steering file content
    steering_text = template_text.format(
        geometry_file=geometry_file,
        boundary_file=boundary_file,
        results_file=results_file,
        title=title,
        duration=duration,
        time_step=time_step,
        initial_depth=initial_depth,
        prescribed_flowrates=prescribed_flowrates,
        prescribed_elevations=prescribed_elevations,
        friction_coefficient=friction_coefficient,
    )

    return steering_text


def sample_combinations(
    n, SLOPE_min, SLOPE_max, n_min, n_max, Q0_min, Q0_max, H0_min, H0_max, BOTTOM_values
):
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
    sample = sampler.random(
        n=n,
    )

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


def generate_geometry(
    idx,
    SLOPE,
    flat_mesh,
    x,
    y,
    noise_grid_x,
    noise_grid_x,
    num_points_x,
    num_points_y,
    channel_length,
):
    """
    Generate geometry with random noise and save it to a file.

    Parameters:
    - idx (int): Index to identify the geometry file.
    - SLOPE (float): Slope value for the geometry.
    - flat_mesh (xarray.Dataset): Dataset containing the geometry data.
    - x (array-like): X-coordinates for interpolation.
    - y (array-like): Y-coordinates for interpolation.
    - noise_grid_x (array-like): Grid X-coordinates for the random noise.
    - noise_grid_x (array-like): Grid Y-coordinates for the random noise.
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
        (noise_grid_x, noise_grid_x),
        smoothed_random_noise,
        bounds_error=False,
        fill_value=None,
    )

    # Compute the slope and noise values for the Z dimension
    z_slope = SLOPE * (channel_length - flat_mesh["x"].values)
    z_noise = interpolator((y, x))

    # Combine slope and noise to get the final Z values
    z = z_slope + z_noise
    z_right = z[num_points_x - 1 :: num_points_x].max()
    z_left = z[0::num_points_x].max()
    # Update the dataset with the new Z values
    flat_mesh["B"].values = z.reshape(1, flat_mesh.y.shape[0])

    # Save the dataset to a file
    flat_mesh.selafin.write(f"geometry/geometry_3x3_NOISE_{idx}.slf")
    return z_left, z_right


# Add logging configuration
logger.add("logfile.log", format="{time} {level} {message}", level="INFO")

with open("constants.yml") as f:
    constants = yaml.safe_load(f)
    logger.info("Loaded constants from constants.yml")

# Adjusting channel dimensions for walls
if constants["channel"]["wall_thickness"] > 0:
    constants["channel"]["width"] += 2 * constants["channel"]["wall_thickness"]
    constants["channel"]["length"] += 2 * constants["channel"]["wall_thickness"]

# ## Steering file generation

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

new_parameters_df = sample_combinations(
    79**2,
    constants["parameters"]["slope_min"],
    constants["parameters"]["slope_max"],
    constants["parameters"]["n_min"],
    constants["parameters"]["n_max"],
    constants["parameters"]["q0_min"],
    constants["parameters"]["q0_max"],
    constants["parameters"]["h0_min"],
    constants["parameters"]["h0_max"],
    constants["parameters"]["bottom_values"],
)

# Calculate additional parameters for new entries
new_parameters_df["yn"] = normal_depth_simple(
    new_parameters_df["Q0"],
    constants["channel"]["width"],
    new_parameters_df["SLOPE"],
    new_parameters_df["n"],
)
new_parameters_df["yc"] = critical_depth_simple(
    new_parameters_df["Q0"], constants["channel"]["width"]
)
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
combined_parameters_df.to_csv("parameters.csv", index=True, index_label="id")
logger.info("Wrote combined parameters to parameters.csv")

parameters_df = pd.read_csv("parameters.csv", index_col="id")
logger.info("Loaded parameters from parameters.csv")

# Load geometry dataset
flat_mesh_path = "geometry/mesh_3x3.slf"
flat_mesh = xr.open_dataset(flat_mesh_path, engine="selafin")
x = flat_mesh["x"].values
y = flat_mesh["y"].values
noise_grid_x = np.linspace(
    0, constants["channel"]["length"], constants["mesh"]["num_points_x"]
)
noise_grid_x = np.linspace(
    0, constants["channel"]["width"], constants["mesh"]["num_points_y"]
)

# Generate steering files
for index, case in tqdm(parameters_df.iterrows(), total=len(parameters_df)):
    z_left, z_right = generate_geometry(
        index,
        case["SLOPE"],
        flat_mesh,
        x,
        y,
        noise_grid_x,
        noise_grid_x,
        constants["mesh"]["num_points_x"],
        constants["mesh"]["num_points_y"],
        constants["channel"]["length"],
    )
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
    logger.info(f"Wrote steering file for case {index}")
