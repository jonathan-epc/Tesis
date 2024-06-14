from scipy.stats import qmc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

n = 47**2
sampler = qmc.LatinHypercube(d=2, strength=2)
sample = sampler.random(n=n)

# Separate the pairs into two arrays for x and y coordinates
x = sample[:, 0]
y = sample[:, 1]

l_bounds = [0.000125, 0.01, 0.0025, 0.05, 0.47, 0.7]
u_bounds = [0.000375, 0.03, 0.0075, 0.15, 0.87, 0.9]
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

# +
# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Samples')

# Add titles and labels
plt.title('Plot of Pairs from NumPy Array')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
# Add grid
plt.grid(True, linewidth=0.5)

# Set the interval for the grid lines
#x_major_locator = MultipleLocator(1/n)
#y_major_locator = MultipleLocator(1/n)

#ax = plt.gca()  # Get the current axis
#ax.xaxis.set_major_locator(x_major_locator)
#ax.yaxis.set_major_locator(y_major_locator)

# Show the plot
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# +
fig, ax = plt.subplots(figsize=(8,8))

# Big bins
ax.hist2d(x, y, bins=(10, 10), cmap=plt.cm.viridis)

plt.show()

# +
sns.jointplot(x=x, y=y, kind='kde', xlim=[0,1], ylim=[0,1], palette='viridis', fill=True)
sns.scatterplot(x=x,y=y)

plt.show()
