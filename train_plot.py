import os
import pandas as pd
import matplotlib.pyplot as plt


# +
# Function to read and process csv files
def read_csv_files(folder_path, keyword):
    data_frames = []
    for file_name in os.listdir(folder_path):
        if keyword in file_name and file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file_name))
            data_frames.append(df[['Step', 'Value']].set_index('Step'))
    return data_frames

# Calculate average time series
def calculate_average(data_frames):
    combined_df = pd.concat(data_frames, axis=1)
    average_series = combined_df.mean(axis=1)
    return average_series

# Plotting function
def plot_time_series(train_dfs, val_dfs, avg_train, avg_val, filename):
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), dpi=150)
    
    # Enhancements
    plot_style = {'alpha': 1.0, 'linewidth': 0.75}
    avg_style = {'color': 'black', 'linewidth': 1, 'label': 'Promedio'}
    
    # Plotting train losses
    for df in train_dfs:
        axs[0].plot(df.index, df['Value'], **plot_style)
    axs[0].plot(avg_train.index, avg_train, **avg_style)
    axs[0].set_title('Entrenamiento', fontsize=14, fontweight='bold')
    axs[0].set_xlabel('Step', fontsize=12)
    axs[0].set_ylabel('Pérdida', fontsize=12)
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].legend()
    
    # Plotting validation losses
    for df in val_dfs:
        axs[1].plot(df.index, df['Value'], **plot_style)
    axs[1].plot(avg_val.index, avg_val, **avg_style)
    axs[1].set_title('Validación', fontsize=14, fontweight='bold')
    axs[1].set_xlabel('Step', fontsize=12)
    axs[1].set_ylabel('Pérdida', fontsize=12)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].legend()
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(filename, bbox_inches='tight', dpi=150)  # Adjusted DPI for high resolution
    plt.show()


# -

# Main script
folder_path = 'runs.tar'
train_keyword = 'Loss_train'
val_keyword = 'Loss_val'

# Read and process the csv files
train_dfs = read_csv_files(folder_path, train_keyword)
val_dfs = read_csv_files(folder_path, val_keyword)

# Calculate the average time series
avg_train = calculate_average(train_dfs)
avg_val = calculate_average(val_dfs)

# Plot the results
plot_time_series(train_dfs, val_dfs, avg_train, avg_val, filename='loss_plot.png')
