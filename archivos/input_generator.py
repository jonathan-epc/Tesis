# ## Imports

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.optimize import fsolve
import xarray as xr
import math
from data_manip.extraction.telemac_file import TelemacFile
from postel.plot1d import plot1d


# ## Functions

def normal_depth(flow_rate, bottom_width, slope, roughness_coefficient):
    """
    Calculate normal depth of flow in a rectangular channel using Manning's equation.

    Parameters:
        flow_rate (float): Flow rate (m³/s).
        bottom_width (float): Bottom width of the channel (m).
        slope (float): Slope of the channel bed (-).
        roughness_coefficient (float): Manning's roughness coefficient (-).

    Returns:
        float: Normal depth of flow (m).
    """
    def manning_equation(depth):
        area = depth * bottom_width
        hydraulic_radius = (depth * bottom_width) / (bottom_width + 2 * depth)
        return flow_rate * math.sqrt(slope) - area * hydraulic_radius**(2/3)

    solution = fsolve(manning_equation, 1)[0]
    print(f"Normal depth: {solution:.5f} [m]")
    
def critical_depth(flow_rate, bottom_width, slope):
    """
    Calculate critical depth of flow in a rectangular channel.

    Parameters:
        flow_rate (float): Flow rate (m³/s).
        bottom_width (float): Bottom width of the channel (m).
        slope (float): Slope of the channel bed (-).

    Returns:
        float: Critical depth of flow (m).
    """
    def equation_c(h):
        area = h * bottom_width
        top_width = bottom_width
        return (flow_rate**2 * top_width) / (area**3 * 9.81 * math.cos(math.atan(slope))) - 1

    solution = fsolve(equation_c, 0.01)[0]
    print(f"Critical depth: {solution:.5f} [m]")


normal_depth(
    flow_rate = 0.01,
    bottom_width = 0.3,
    slope = 1/100,
    roughness_coefficient  = 0.02)

critical_depth(
    flow_rate = 0.01,
    bottom_width = 0.3,
    slope = 1/100)

# ## Bathymetry creation

# Define dimensions
channel_width = 0.3  # in m
channel_length = 12  # in m
channel_depth = 0.3  # in m
slope = 1/100  # 1% slope
wall_thickness = 0.00  # in m
flat_zone = 0
# Adjusting channel dimensions for walls
channel_width += 2 * wall_thickness
channel_length += 2 * wall_thickness
# Generate base mesh for the channel
num_points_y = 11  # Adjust as needed for resolution
num_points_x = 401

# Read original geometry file in selafin format
ds = xr.open_dataset("geometry/geometry_0.slf", engine="selafin")
plt.imshow(ds['B'].values.reshape(num_points_y,num_points_x))
plt.show()
plt.close()

x = np.linspace(-wall_thickness*0, channel_length + wall_thickness*0, num_points_x)
y = np.linspace(-wall_thickness*0, channel_width + wall_thickness*0, num_points_y)
X, Y = np.meshgrid(x, y)
Z_slope =  -slope * X + slope * x.max()
Z_slope[X <= flat_zone] = -slope * x[x<=flat_zone][-1] + slope * x.max()
plt.imshow(Z_slope)
plt.show()
plt.close()

Z = Z_slope
#Z[0,:] = np.linspace(channel_depth, channel_depth, num_points_x)  # bottom wall
#Z[-1,:] = np.linspace(channel_depth, channel_depth, num_points_x)  # top wall
ds['B'].values = Z.reshape(1,ds.y.shape[0])
ds.selafin.write("geometry/geometry_0.slf")

plt.imshow(Z)

rng = np.random.default_rng()
sigma = 3  # Standard deviation for Gaussian blur

for i in range(1,100):
    Z_blur = gaussian_filter(rng.standard_normal(size=(num_points_y, num_points_x)), sigma=sigma)*0.15
    Z = Z_slope + Z_blur
    Z[0:2,:] = np.linspace(channel_depth, channel_depth, num_points_x)  # bottom wall
    Z[-2:Z.shape[0],:] = np.linspace(channel_depth, channel_depth, num_points_x)  # top wall
#     ds['B'].values = Z.reshape(1,ds.y.shape[0])
    plt.imsave(f"imagenes/geometry_{i}.png", Z, cmap='inferno')
#     ds.selafin.write(f"geometry/geometry_{i}.slf")

def write_steering_file(filename, geometry, friction_coefficient, prescribed_elevations, prescribed_flowrates):
    with open(f"steering_{filename}.cas", 'w') as file:   
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ TELEMAC2D Version v8p4 Apr 1, 2024\n")
        file.write(f"/ Caso {filename}\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ EQUATIONS\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("MASS-LUMPING ON TRACERS              =1.\n")
        file.write("IMPLICITATION COEFFICIENT OF TRACERS =0.6\n")
        file.write("LAW OF BOTTOM FRICTION               =4\n")
        file.write(f"FRICTION COEFFICIENT                 ={friction_coefficient}\n")
        file.write("TURBULENCE MODEL                     =3\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ EQUATIONS, ADVECTION\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("SCHEME FOR ADVECTION OF TRACERS    =5\n")
        file.write("SCHEME FOR ADVECTION OF VELOCITIES =14\n")
        file.write("SCHEME FOR ADVECTION OF K-EPSILON  =14\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ EQUATIONS, BOUNDARY CONDITIONS\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write(f"PRESCRIBED ELEVATIONS ={';'.join(map(str, prescribed_elevations))}\n")
        file.write(f"PRESCRIBED FLOWRATES  ={';'.join(map(str, prescribed_flowrates))}\n")
        file.write("VELOCITY PROFILES     =4;1\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ EQUATIONS, INITIAL CONDITIONS\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("INITIAL CONDITIONS ='ZERO DEPTH'\n")
#         file.write("INITIAL DEPTH      =1\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ INPUT-OUTPUT, FILES\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write(f"STEERING FILE            ='steering_{filename}.cas'\n")
        file.write(f"GEOMETRY FILE            ='geometry/geometry_{geometry}.slf'\n")
        file.write(f"RESULTS FILE             ='results/results_{filename}.slf'\n")
        file.write("BOUNDARY CONDITIONS FILE ='boundary/boundary.cli'\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ INPUT-OUTPUT, GRAPHICS AND LISTING\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("VARIABLES FOR GRAPHIC PRINTOUTS =U,V,S,B,Q,F,H\n")
        file.write("LISTING PRINTOUT PERIOD         =100\n")
        file.write("GRAPHIC PRINTOUT PERIOD         =100\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ INPUT-OUTPUT, INFORMATION\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("CONTROL OF LIMITS =true\n")
        file.write(f"TITLE             ='Caso {filename}'\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ NUMERICAL PARAMETERS\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("MATRIX STORAGE                          =3\n")
        file.write("TREATMENT OF NEGATIVE DEPTHS            =2\n")
        file.write("STOP IF A STEADY STATE IS REACHED       =true\n")
        file.write("TIME STEP                               =1\n")
        file.write("CONTINUITY CORRECTION                   =YES\n")
        file.write("OPTION FOR THE TREATMENT OF TIDAL FLATS =1\n")
        file.write("TREATMENT OF THE LINEAR SYSTEM          =2\n")
        file.write("NUMBER OF TIME STEPS                    =500\n")
        file.write("TIDAL FLATS                             =YES\n")
        file.write("SUPG OPTION                             =0;0;2;2\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ NUMERICAL PARAMETERS, SOLVER\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("MAXIMUM NUMBER OF ITERATIONS FOR SOLVER =200\n")
        file.write("SOLVER ACCURACY                         =1.E-4\n")
        file.write("SOLVER                                  =3\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("/ NUMERICAL PARAMETERS, VELOCITY-CELERITY-HIGHT\n")
        file.write("/---------------------------------------------------------------------\n")
        file.write("IMPLICITATION FOR DIFFUSION OF VELOCITY =1.\n")
        file.write("IMPLICITATION FOR VELOCITY              =0.55\n")
        file.write("MASS-LUMPING ON H                       =1.\n")
        file.write("MASS-LUMPING ON VELOCITY                =1.\n")
        file.write("IMPLICITATION FOR DEPTH                 =0.55\n")


count = 0
for i in range(2,21,2):
    n = i / 100
    for j in range(100,121,5):
        Q = j / 10000
        # for k in range(1,100):
            # write_input_file(f"steerings/steering_{n}_{Q}_{k}.cas", f"geometria{k}.slf", n, [30, 0], [0, Q])
        write_steering_file(
            filename = count,
            geometry = 0,
            friction_coefficient = n,
            prescribed_elevations = [0.0, -.03],
            prescribed_flowrates =[Q, 0.0])
        count += 1

# ## Results reading

res = TelemacFile('results/results_0.slf')
# List of points defining the polyline
poly_points = [[0.0,0.15], [12.0,0.15]]

# List of number of discretized points for each polyline segment
poly_number =[1000]

# Getting water depth values over time for each discretized points of the polyline
poly_coord, abs_curv,values_polylines=res.get_timeseries_on_polyline('WATER DEPTH', poly_points, poly_number)

#Initialising figure
fig, ax = plt.subplots(figsize=(10,5))

# plot over the polyline of the initial condition
plot1d(ax, abs_curv,values_polylines[:,-1], 
       x_label='y (m)',
       y_label='water depth (m)', 
       plot_label='initial 1d condition')

# Displaying legend
ax.legend()

#Showing figure
plt.show()
res.close()
