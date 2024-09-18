import tkinter as tk
from tkinter import filedialog, Entry, Label, messagebox, ttk
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import median_filter
from tqdm import tqdm
from fdm import FDM
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
import h5py
import math
import sys
import subprocess
import gc
from tkinter import *
from scipy.ndimage import gaussian_filter

DEFAULTS = {
    "initial temperature": 286.15,
    "surface emissivity": 0.94,
    "wall thickness": 0.000795,
    "surface width": 0.9144,
    "surface height": 1.5204,
    "convective hc exposed": 10,
    "convective hc unexposed": 10,
    "maximum temperature": 400,
    "maximum incident heat flux": 30,
    "time step size": 1,
    "time derivative method" : "numpy gradient"
}
STEFAN_BOLTZMANN_CONSTANT = 5.67e-8
global_state = {"source folder": "", "dest folder": "", "time array": "", "constants": DEFAULTS.copy(), "convection method": "constants", "second order grad": False}

# Utility Functions
def nu_gas(T_g):
    return (0.0000627695*T_g**2 + 0.0969429527*T_g + 13.152284446)*10**(-6)

def k_gas(T_g):
    return (0.000000013395106*T_g**2 + 0.000069426548*T_g + 0.0246228047)

def k_metal(T_g):
    global global_state
    metal = global_state["wall material"].get()
    if metal == 'steel (FSRI)':
        k_metal = (-3.38759644163437 * 10**(-6) * T_g**2) + 0.017732749 + 13.6405477360128
    elif metal == 'steel (Engineering Toolbox)':
        k_metal = 8.1119 * np.log(T_g) - 30.377
    elif metal == 'aluminum':
        k_metal = -9.412 * np.log(T_g) + 291.55
    elif metal == 'copper':
        k_metal = -47.15 * np.log(T_g) + 674.98
    else:
        print('Selected material is invalid, default values assumed.')
        k_metal = 45
    return k_metal

def cp_metal(T_g):
    global global_state
    metal = global_state["wall material"].get()
    if metal == 'steel (FSRI)':
        cp_metal = (-1.24093738317144*10**(-7)) * T_g**2 + 0.000288489 * T_g + 0.467198515635077
    elif metal == 'steel (Engineering Toolbox)':
        cp_metal = 0.1414 * np.log(T_g) - 0.3534
    elif metal == 'aluminum':
        cp_metal = 0.2477 * np.log(T_g) - 0.5285
    elif metal == 'copper':
        cp_metal = 0.0779 * np.log(T_g) - 0.0712
    else:
        print('Selected material is invalid, default values assumed.')
        cp_metal = 490
    return 1000*cp_metal

def rho_metal(T_g):
    global global_state
    metal = global_state["wall material"].get()
    if metal == 'steel (FSRI)':
        rho_metal = 7590
    elif metal == 'steel (Engineering Toolbox)':
        rho_metal = 8050
    elif metal == 'aluminum':
        rho_metal = -0.1958*np.log(T_g) + 2758
    elif metal == 'copper':
        rho_metal = -0.5124*np.log(T_g) + 9075.6
    else:
        print('Selected material is invalid, default values assumed.')
        rho_metal = 8050
    return rho_metal

def Grashof(dT_w, T_film, T_amb):
    L = 1
    return 9.81*(1/(T_film))*abs(dT_w)*L**3 /(nu_gas(T_film - 273.15))**2

def natural_conv_correlation(Gr):
    Pr = 0.701
    return ((0.825 + (0.387*(abs(Gr)*Pr)**(1/6))/(1+(0.492/Pr)**(9/16))**(8/27))**2)

def calculate_gradients(T_values_2d, i, j, element_width, element_height, use_higher_order=False):
    rows, cols = T_values_2d.shape
    grad_T = np.zeros(2)  # [grad_T_x, grad_T_y]

    if use_higher_order:
        grad_y, grad_x = np.gradient(T_values_2d, axis=(0, 1))
        grad_T[0] = grad_y[i, j] * element_height / element_width
        grad_T[1] = grad_x[i, j] * element_width / element_height
    else:
        if i > 0:
            grad_T[0] += (-T_values_2d[i, j] + T_values_2d[i-1, j]) * element_height / element_width
        if i < rows - 1:
            grad_T[0] += (T_values_2d[i+1, j] - T_values_2d[i, j]) * element_height / element_width
        if j > 0:
            grad_T[1] += (-T_values_2d[i, j] + T_values_2d[i, j-1]) * element_width / element_height
        if j < cols - 1:
            grad_T[1] += (T_values_2d[i, j+1] - T_values_2d[i, j]) * element_width / element_height
    return grad_T * k_metal(T_values_2d[i, j]) * DEFAULTS['wall thickness']

def inverse_heat_transfer(time_series_data, element_width, element_height, time_step, convection_method):

    if global_state["temperature_unit_var"].get() == "Celsius":
        time_series_data += 273.15
        print('Temperature unit will be converted from Celsius to Kelvin.')
    else:
        print('Temperature unit was set to Kelvin, no conversion needed.')
    rows, cols, num_time_steps = time_series_data.shape
    estimated_flux = np.zeros(time_series_data.shape)
    hfc = np.zeros(time_series_data.shape)

    area = element_height * element_width

    dt_method = global_state["time derivative method"].get()

    temp_grad_time = np.zeros(time_series_data.shape)

    if global_state['Tf_array_checkbox_var'].get() == 1:
        T_film = Tfilm_interp( float(global_state["constants"]["surface width"].get()) , float(global_state["constants"]["surface width"].get()) , time_series_data.shape[-1])
    else:
        
        T_film = DEFAULTS['initial temperature'] * np.ones((rows,cols,num_time_steps))
        
    if global_state['Tamb_array_checkbox_var'].get() == 1:
        Tamb = T0_intrp()
    else:
        Tamb = DEFAULTS['initial temperature'] * np.ones(num_time_steps)

    if dt_method == 'FDM':
        fdm_solver = FDM()
        time_grid = np.linspace(0, num_time_steps * time_step, num_time_steps)
        fdm_solver = FDM(time_grid)
        for i in range(rows):
            for j in range(cols):
                temp_grad_time[i, j, :] = fdm_solver.differentiate(time_series_data[i, j, :], time_step)
    else:  # Default method
        temp_grad_time = np.gradient(time_series_data, axis=2 , edge_order=2) / time_step

    for k in tqdm(range(num_time_steps) , desc = 'Calculating IHT'):
        for i in range(rows):
            for j in range(cols):
                grad_T = calculate_gradients(np.squeeze(time_series_data[:, :, k]), i, j, element_width, element_height)
                q_cond = np.sum(grad_T)/area
                T = time_series_data[i, j, k]
                if convection_method == 'natural convection correlation':
                    #### exposed side
                    Grashov = Grashof (T - Tamb[k] , T_film[i, j, k] , Tamb[k])
                    DEFAULTS['convective hc exposed'] = natural_conv_correlation(Grashov) * k_gas((T + Tamb[k])/2)
                    #### unexposed side
                    Grashov = Grashof (T - DEFAULTS['initial temperature']  ,\
                                        (T + DEFAULTS['initial temperature'] )/2 ,\
                                                DEFAULTS['initial temperature'] )
                    DEFAULTS['convective hc unexposed'] = natural_conv_correlation(Grashov) * k_gas((T + DEFAULTS['initial temperature'])/2)
                
                hfc[i,j,k] = DEFAULTS['convective hc exposed']

                q_conv = (DEFAULTS['convective hc exposed'] + DEFAULTS['convective hc unexposed'])  * (T - Tamb[k])
                q_rad = 2 * DEFAULTS['surface emissivity'] * STEFAN_BOLTZMANN_CONSTANT  * (T**4 - 0.5*Tamb[k]**4)
                q_storage = cp_metal(T) * rho_metal(T)  * DEFAULTS['wall thickness'] * temp_grad_time[i, j, k]
                estimated_flux[i, j, k] = (q_storage + q_conv + q_rad - q_cond) / (1000 * DEFAULTS['surface emissivity'])
    return estimated_flux , hfc , T_film , Tamb

def select_source_folder():
    global global_state
    global_state["source_folder"] = filedialog.askdirectory()
    if global_state["source_folder"]:
        total_size_mb = sum(os.path.getsize(os.path.join(global_state["source_folder"], f)) for f in os.listdir(global_state["source_folder"]) if os.path.isfile(os.path.join(global_state["source_folder"], f))) / (1024**2)
        num_files = len(glob.glob(os.path.join(global_state["source_folder"], '*.csv')))
        proceed = messagebox.askyesno("Proceed", f"Total size of files: {total_size_mb:.2f} MB\nDo you want to proceed?")
        print('Source Folder Selected: '+ global_state["source_folder"])
        if not proceed:
            global_state["source_folder"] = ""

def select_dest_folder():
    global global_state
    global_state["dest_folder"] = filedialog.askdirectory()
    print('Destination Folder Selected: '+ global_state["dest_folder"])

def select_time_array_csv():
    global global_state
    global_state['time_array_csv_file'] = filedialog.askopenfilename(filetypes=[("Time Array CSV", "*.csv")])
    print("Selected Time Array CSV File: ", global_state['time_array_csv_file'])

def select_DAQ_excel():
    global global_state
    global_state['DAQ_path'] = filedialog.askopenfilename(filetypes=[("DAQ path", "*.xlsx")])
    print("Selected DAQ Excel File:", global_state['DAQ_path'])
    
def T0_intrp():
    DAQ_path = global_state['DAQ_path']
    file = pd.read_excel(DAQ_path , header = 0)['T0'].dropna()
    file += 273.15
    T0_data = savgol_filter(np.squeeze(np.array(file)), window_length=21, polyorder=2, axis=0)
    return T0_data

def Tfilm_interp(width , height , nt):
    # select_DAQ_excel()
    DAQ_path = global_state['DAQ_path']
    source_folder = global_state["source_folder"]
    h5py_file_paths = glob.glob(os.path.join(source_folder, '*.h5'))

    if not h5py_file_paths:
        print("No h5py files found in the source directory.")
    h5py_input_file = h5py.File(h5py_file_paths[0], 'r')
    file_paths = list(h5py_input_file.keys())
    temp_data = h5py_input_file[file_paths[0]][:]
    temp_data = temp_data.astype(np.float64)
    nx , ny = temp_data.shape[0] , temp_data.shape[1]
    del temp_data

    xy = pd.read_excel(DAQ_path , header = 0)[['x','y']].dropna()
    points = np.array(list(zip(xy['x'],xy['y'])))
    file = pd.read_excel(DAQ_path , header = 0).iloc[:,:len(list(points))]
    file += 273.15
    
    grid_z = []
    # Define the grid points where the values are known
    grid_x, grid_y = np.mgrid[0:width:nx*1j, 0:height:ny*1j]
    for t_idx , _ in tqdm(enumerate(range(file.shape[0])), desc = 'Generating film temperature interpolated arrays: '):
        values = []
        for col in range(file.shape[1]):
            values.append(file.iloc[t_idx , col])
        # Interpolate using griddata
        grid_z.append(griddata(points, values, (grid_x, grid_y), method='nearest'))
    # Check the shape to confirm it matches the custom resolution
    output = np.stack(grid_z , axis = 2)
    print("Interpolated film temperature grid shape:", output.shape)
    return(output)
    
def check_h5py_files(folder):
    h5py_files = glob.glob(os.path.join(folder, '*.h5'))
    if len(h5py_files) == 1:
        return h5py_files[0]  # Return the single h5py file path
    else:
        print(f"Error: Expected exactly one h5py file in the folder, but found {len(h5py_files)}.")
        return None

def process_directory_and_plot(source_dir, dest_dir, element_width, element_height,  convection_method):
    input_file_type = global_state["source_file_type"].get()
    output_file_type = global_state["dest_file_type"].get()

    if output_file_type == "h5py":
        h5py_qr = h5py.File(os.path.join(dest_dir, "Incident_Radiative_HF.h5"), 'w')
        h5py_Ts = h5py.File(os.path.join(dest_dir, "Surface_Temperature.h5"), 'w')
        h5py_Tf = h5py.File(os.path.join(dest_dir, "Film_Temperature.h5"), 'w')
        h5py_T0 = h5py.File(os.path.join(dest_dir, "Ambient_Temperature.h5"), 'w')
        h5py_hfc = h5py.File(os.path.join(dest_dir, "Exposed_Side_CHTC.h5"), 'w')

    if input_file_type == "h5py":
        h5py_file_paths = glob.glob(os.path.join(source_dir, '*.h5'))
        if not h5py_file_paths:
            print("No h5py files found in the source directory.")
            return
        h5py_input_file = h5py.File(h5py_file_paths[0], 'r')
        file_paths = list(h5py_input_file.keys())
    elif input_file_type == "csv":
        file_paths = glob.glob(os.path.join(source_dir, '*.csv'))

    if not file_paths:
        print("No files found in the source directory.")
        return

    if global_state['time_array_checkbox_var'] == 1:
        time_step = np.gradient(np.squeeze(np.array(pd.read_csv(global_state['time_array csv file'], index_col=0))),edge_order=1)/1000000
    else:
        time_step = float(global_state["constants"]["time step size"].get()) * (np.array(len(file_paths) * [1]))

    batch_files = int(global_state['constants']['batch_files'].get())
    global total_files
    total_files = len(file_paths)
    window_length = int(5 / float(global_state["constants"]["time step size"].get())) # Savgol filter window length

    for batch_idx, batch_start in enumerate(range(0, total_files, batch_files)):
        batch_end = min(batch_start + batch_files, total_files)
        is_last_batch = batch_end == total_files
        current_batch_paths = file_paths[batch_start:batch_end]

        time_series_data = []

        for file_path in current_batch_paths:
            if input_file_type == "csv":
                temp_data = pd.read_csv(os.path.join(source_dir, file_path), header=None).values
            elif input_file_type == "h5py":
                temp_data = h5py_input_file[file_path][:]
            temp_data = temp_data.astype(np.float64)
            time_series_data.append(temp_data)

        time_series_data = np.stack(time_series_data, axis=-1)

        # Handling the last batch for savgol filter
        if is_last_batch and time_series_data.shape[2] < window_length:
            additional_points_needed = window_length - time_series_data.shape[2]
            prev_batch_start = max(0, batch_start - additional_points_needed)
            additional_batch_paths = file_paths[prev_batch_start:batch_start]

            additional_data = []
            for file_path in additional_batch_paths:
                if input_file_type == "csv":
                    additional_temp_data = pd.read_csv(os.path.join(source_dir, file_path), header=None).values
                elif input_file_type == "h5py":
                    additional_temp_data = h5py_input_file[file_path][:]
                additional_temp_data = additional_temp_data.astype(np.float64)
                additional_data.append(additional_temp_data)

            additional_data = np.stack(additional_data, axis=-1)
            combined_data = np.concatenate((additional_data, time_series_data), axis=2)
            time_series_data = combined_data#[:, :, -batch_files:]

        estimated_flux, hfc, Tf, T_inf = inverse_heat_transfer(time_series_data, element_width, element_height, time_step, convection_method)

        sigma_values = [7, 7, 15]

        # Apply Gaussian Filter with different sigmas for each axis
        estimated_flux = gaussian_filter(estimated_flux, sigma=sigma_values)
        # Export results for the current batch
        for i in tqdm(range(estimated_flux.shape[2]), desc='Exporting Results: '):
            # Updated line: Format the frame index with 6 digits padding
            padded_index = f"{i:06d}"
            dataset_name_qr = f'estimated_flux_batch{batch_idx}_frame{padded_index}'
            dataset_name_Ts = f'surface_temperature_batch{batch_idx}_frame{padded_index}'
            dataset_name_Tf = f'film_temperature_batch{batch_idx}_frame{padded_index}'
            dataset_name_T0 = f'Ambient_temperature_batch{batch_idx}_frame{padded_index}'
            dataset_name_hfc = f'exposed_side_CHTC_batch{batch_idx}_frame{padded_index}'

            if output_file_type == "csv":
                output_file_path = os.path.join(dest_dir, dataset_name + '.csv')
                pd.DataFrame(estimated_flux[:, :, i]).to_csv(output_file_path, header=False, index=False, float_format='%.15f')
            elif output_file_type == "h5py":
                if dataset_name_qr in h5py_qr:
                    del h5py_qr[dataset_name_qr]
                if dataset_name_hfc in h5py_hfc:
                    del h5py_hfc[dataset_name_hfc]
                if dataset_name_Tf in h5py_Tf:
                    del h5py_Ts[dataset_name_Tf]
                if dataset_name_Ts in h5py_Ts:
                    del h5py_Ts[dataset_name_Ts]
                if dataset_name_T0 in h5py_T0:
                    del h5py_T0[dataset_name_T0]

                h5py_qr.create_dataset(dataset_name_qr, data=estimated_flux[:, :, i])
                h5py_hfc.create_dataset(dataset_name_hfc, data=hfc[:, :, i])
                h5py_Ts.create_dataset(dataset_name_Ts, data=time_series_data[:, :, i])
                h5py_Tf.create_dataset(dataset_name_Tf, data=Tf[:, :, i])
                h5py_T0.create_dataset(dataset_name_T0, data=T_inf[i])

    if input_file_type == "h5py":
        h5py_input_file.close()
    if output_file_type == "h5py":
        h5py_qr.close()
        h5py_hfc.close()
        h5py_Ts.close()
        h5py_Tf.close()
        h5py_T0.close()

    print("All files have been processed and saved in the destination folder.")  

def apply_inverse_model():
    global global_state, DEFAULTS
    source_folder = global_state["source_folder"]
    dest_folder = global_state["dest_folder"]
    
    if not source_folder or not dest_folder:
        print("Please select both source and destination folders.")
        return

    input_file_type = global_state["source_file_type"].get()

    # Load the first file based on the input file type
    if input_file_type == "csv":
        csv_files = glob.glob(os.path.join(source_folder, '*.csv'))
        if not csv_files:
            print("No CSV files found in the source folder.")
            return
        first_file_path = csv_files[0]
        temperature_file = pd.read_csv(first_file_path, header=None).values
    elif input_file_type == "h5py":
        h5py_files = glob.glob(os.path.join(source_folder, '*.h5'))
        if not h5py_files:
            print("No h5py files found in the source directory.")
            return
        h5py_file_path = h5py_files[0]
        with h5py.File(h5py_file_path, 'r') as h5_file:
            first_dataset_name = list(h5_file.keys())[0]
            temperature_file = h5_file[first_dataset_name][:]

    surface_width_m = float(global_state["constants"]["surface width"].get())
    surface_height_m = float(global_state["constants"]["surface height"].get())

    element_width = surface_width_m / temperature_file.shape[1]
    element_height = surface_height_m / temperature_file.shape[0]

    process_directory_and_plot(source_folder, dest_folder, element_width, element_height,  global_state["convection_method_var"].get())

def export_pngs(root, batch_size):
    global global_state
    source_folder = global_state["source_folder"]
    dest_folder = global_state["dest_folder"]
    source_file_type = global_state["source_file_type"].get()
    dest_file_type = global_state["dest_file_type"].get()
    labels_x  = np.arange(0 , float(global_state["constants"]["surface width"].get()), 0.1)
    labels_y = np.arange(0 , float(global_state["constants"]["surface height"].get()), 0.1)
    labels_width , labels_height = [],[]
    for xs in labels_x:
        labels_width.append(np.round(xs,1))
    for ys in labels_y:
        labels_height.append(np.round(ys,1))

    # Determine the total number of frames based on file type and content
    if dest_file_type == "h5py":
        source_h5py_path = next((f for f in glob.glob(os.path.join(source_folder, 'processed_temperature_array.h5'))), None)
        if source_h5py_path:
            with h5py.File(source_h5py_path, 'r') as h5_file:
                total_frames = len(h5_file.keys())
                print(f"Total frames: {total_frames}")
    else:
        total_frames = max(len(glob.glob(os.path.join(source_folder, '*.csv'))),
                           len(glob.glob(os.path.join(dest_folder, '*.csv'))))

    total_batches = (total_frames + batch_size - 1) // batch_size
    # Ensure the output directory exists
    output_dir = os.path.join(dest_folder, "Exported_PNGs")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the progress bar for the total number of frames
    pbar = tqdm(total=total_frames, desc="Generating PNG frames")
    # Process each batch
    for batch_index in range(total_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, total_frames)

        source_data_frames = load_data_frames(source_folder, source_file_type, "processed_temperature_array.h5", batch_start, batch_end)
        dest_data_frames = load_data_frames(dest_folder, dest_file_type, "Incident_Radiative_HF.h5", batch_start, batch_end)

        source_levels = np.arange(0, float(global_state["constants"]["maximum temperature"].get()), 5)
        dest_levels = np.arange(0, float(global_state["constants"]["maximum incident heat flux"].get()), .5)
        fig, axs = plt.subplots(1, 2, figsize=(12, 7))
        cnt1 = axs[0].contourf([[0, 0], [0, 0]], levels=source_levels, cmap='jet')
        cnt2 = axs[1].contourf([[0, 0], [0, 0]], levels=dest_levels, cmap='jet')
        cbar1 = fig.colorbar(cnt1, ax=axs[0] , ticks=np.arange(0, float(global_state["constants"]["maximum temperature"].get()), 20))
        cbar2 = fig.colorbar(cnt2, ax=axs[1] , ticks=np.arange(0, float(global_state["constants"]["maximum incident heat flux"].get()), 5))
        cbar1.set_label("Temperature (\u00B0C)", fontsize=16)
        cbar2.set_label(r"$\mathrm{Incident~Radiative~Heat~Flux~(kW/m^2)}$", fontsize = 16)
        cbar1.ax.tick_params(labelsize=14)
        cbar2.ax.tick_params(labelsize=14)
        
        for frame_index in range(batch_start, batch_end):
            # Clear the axes for the new frame
            axs[0].clear()
            axs[1].clear()
            if frame_index - batch_start < len(source_data_frames):
                axs[0].contourf(source_data_frames[frame_index - batch_start], levels=source_levels, cmap='jet')
            if frame_index - batch_start < len(dest_data_frames):
                axs[1].contourf(dest_data_frames[frame_index - batch_start], levels=dest_levels, cmap='jet')
            axs[0].set_title(f'Temperature {frame_index}')
            axs[1].set_title(f'Incident Radiative Heat Flux {frame_index}')
            surface_width_m = float(global_state["constants"]["surface width"].get())

            xmin, xmax = axs[0].get_xlim()
            ymin, ymax = axs[0].get_ylim()
            axs[0].set_xticks(np.linspace(xmin,xmax,labels_x.shape[0]) , labels_width, rotation='vertical')
            axs[1].set_xticks(np.linspace(xmin,xmax,labels_x.shape[0]) , labels_width, rotation='vertical')
            axs[0].set_yticks(np.linspace(ymin,ymax,labels_y.shape[0]), labels_height, rotation='horizontal')
            axs[1].set_yticks(np.linspace(ymin,ymax,labels_y.shape[0]), labels_height, rotation='horizontal')

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"batch_{batch_index}_frame_{frame_index:06d}.svg") , dpi = 150, bbox_inches='tight')
            pbar.update(1)

        plt.close(fig)
        del source_data_frames, dest_data_frames
        gc.collect()

    pbar.close()
    print("PNG export completed.")

def load_data_frames(folder, file_type, file_name = "", batch_start=None, batch_end=None):
        if file_type == "h5py":
            file_path = next((f for f in glob.glob(os.path.join(folder, file_name))), None)
            if file_path:
                with h5py.File(file_path, 'r') as h5_file:
                    datasets = list(h5_file.keys())
                    # If batching is used, load only the datasets for the current batch
                    if batch_start is not None and batch_end is not None:
                        selected_datasets = datasets[batch_start:batch_end]
                    else:
                        selected_datasets = datasets
                    return [pd.DataFrame(h5_file[dataset][:]) for dataset in tqdm(selected_datasets, desc = "Loading Source/Destination Data: ")]
        else:  
            files = sorted(glob.glob(os.path.join(folder, '*.csv')))
            # If batching is used, select only the files for the current batch
            if batch_start is not None and batch_end is not None:
                selected_files = files[batch_start:batch_end]
            else:
                selected_files = files
            return [pd.read_csv(file, header=None) for file in tqdm(selected_files, desc = "Loading dataframes for video generation: ")]

def create_video(source_folder, dest_folder, source_file_type, dest_file_type,global_max_source, global_max_dest, Time_Step_Size_s, batch_size):
    # Determine the total number of frames based on file type and content
    if dest_file_type == "h5py":
        source_h5py_path = next((f for f in glob.glob(os.path.join(source_folder, 'processed_temperature_array.h5'))), None)
        if source_h5py_path:
            with h5py.File(source_h5py_path, 'r') as h5_file:
                total_frames = len(h5_file.keys())
                print(f"Total frames: {total_frames}")

    else:
        total_frames = max(len(glob.glob(os.path.join(source_folder, '*.csv'))),
                        len(glob.glob(os.path.join(dest_folder, '*.csv'))))

    total_batches = (total_frames + batch_size - 1) // batch_size

    video_files = []

    for batch_index in range(total_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, total_frames)

        source_data_frames = load_data_frames(source_folder, source_file_type, "processed_temperature_array.h5", batch_start, batch_end)
        dest_data_frames = load_data_frames(dest_folder, dest_file_type, "Incident_Radiative_HF.h5", batch_start, batch_end)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        plt.tight_layout()

        source_levels = np.linspace(0, global_max_source, 100)
        dest_levels = np.linspace(0, global_max_dest, 100)
        # Initial plots to setup the colorbars
        cnt1 = axs[0].contourf([[0, 0], [0, 0]], levels=source_levels, cmap='turbo')
        cnt2 = axs[1].contourf([[0, 0], [0, 0]], levels=dest_levels, cmap='turbo')
        fig.colorbar(cnt1 , ax = axs[0])
        fig.colorbar(cnt2 , ax = axs[1])

        pbar = tqdm(total = len(source_data_frames), desc = "Generating animation frames: ")
        def animate(frame_index):
            axs[0].clear(); axs[1].clear()
            if frame_index < len(source_data_frames):
                axs[0].contourf(source_data_frames[frame_index], levels=source_levels, cmap='turbo')
            if frame_index < len(dest_data_frames):
                axs[1].contourf(dest_data_frames[frame_index], levels=dest_levels, cmap='turbo')
            axs[0].set_title(f'Temperature {frame_index}')
            axs[1].set_title(f'Incident Radiative Heat Flux {frame_index}')
            ######## update the manual progress bar here
            pbar.update(1)

        ani = animation.FuncAnimation(fig, animate, frames=range(len(source_data_frames)), repeat=False, interval=1000 * Time_Step_Size_s, blit = True)
        
        temp_video_path = os.path.join(dest_folder, f'temp_video_batch_{batch_index}.mp4')
        ani.save(temp_video_path, writer='ffmpeg', fps=1/Time_Step_Size_s)
        plt.close(fig)
        ############
        pbar.close()
        video_files.append(temp_video_path)

        # free up memory
        del source_data_frames, dest_data_frames
        gc.collect()

    # Concatenate batch videos
    file_list_path = os.path.join(dest_folder, "video_files.txt")
    with open(file_list_path, 'w') as file_list:
        for video_file in tqdm(video_files, desc = "Writing Videos"):
            file_list.write(f"file '{video_file}'\n")

    final_video_path = os.path.join(dest_folder, 'contour_video.mp4')

    concat_command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", file_list_path, "-c", "copy", final_video_path, "-loglevel", "quiet"]
    
    subprocess.run(concat_command, check=True)
    print(f"Final video saved to {final_video_path}")

    # Remove temporary batch video files
    for file in video_files:
        os.remove(file)
    os.remove(file_list_path)

# GUI
def setup_second_tab(parent):
    # parent.title("Inverse Heat Transfer")
    global_state["source_file_type"] = tk.StringVar(value="h5py")  
    global_state["dest_file_type"] = tk.StringVar(value="h5py") 
    global_state["wall material"] = tk.StringVar(value="steel (FSRI)") 
    row = 0
    # Source File Type Dropdown
    metal_label = Label(parent, text='select wall material:')
    metal_label.grid(row = 0, column = 0, sticky = W, pady = 2)
    metal_options = ['steel (FSRI)', 'steel (Engineering Toolbox)' , 'aluminum','copper']
    metal_menu = tk.OptionMenu(parent, global_state["wall material"], *metal_options)
    metal_menu.grid(row = 0, column = 1, sticky = W, pady = 2)
    row += 1
     
    # Source File Type Dropdown
    source_file_type_label = Label(parent, text='Select Source File Type:')
    source_file_type_label.grid(row = row, column = 0, sticky = W, pady = 2)
    source_file_type_options = ['csv', 'h5py']
    source_file_type_menu = tk.OptionMenu(parent, global_state["source_file_type"], *source_file_type_options)
    source_file_type_menu.grid(row = row, column = 1, sticky = W, pady = 2)
    row += 1
    # Destination File Type Dropdown
    dest_file_type_label = tk.Label(parent, text='Select Destination File Type:')
    dest_file_type_label.grid(row = row, column = 0, sticky = W, pady = 2)
    dest_file_type_options = ['csv', 'h5py']
    dest_file_type_menu = tk.OptionMenu(parent, global_state["dest_file_type"], *dest_file_type_options)
    dest_file_type_menu.grid(row = row, column = 1, sticky = W, pady = 2)
    row += 1
    ####################################
    Label(parent, text='Temperature Unit').grid(row = row, column = 0, sticky = W, pady = 2)
    global_state["temperature_unit_var"] = tk.StringVar(parent)
    temperature_unit_options = ['Kelvin', 'Celsius']
    global_state["temperature_unit_var"].set('Celsius')  # default value
    tk.OptionMenu(parent, global_state["temperature_unit_var"] , *temperature_unit_options).grid(row = row, column = 1, sticky = W, pady = 2)
    row += 1
    ####### Time detivative discritization metthod
    time_derivative_method_options = ['numpy gradient', 'FDM']
    global_state["time derivative method"] = tk.StringVar(parent)
    global_state["time derivative method"].set(DEFAULTS["time derivative method"])
    tk.Label(parent, text='Time Derivative Method').grid(row = row, column = 0, sticky = W, pady = 2)
    tk.OptionMenu(parent, global_state["time derivative method"], *time_derivative_method_options).grid(row = row, column = 1, sticky = W, pady = 2)
    
    row += 1
    ####### convection method ######
    Label(parent, text='Convection Calculation Method').grid(row = row, column = 0, sticky = W, pady = 2)
    global_state["convection_method_var"] = tk.StringVar(parent)
    global_state["convection_method_var"].set('constants')
    convection_method_options = ['constants', 'natural convection correlation']
    tk.OptionMenu(parent, global_state["convection_method_var"], *convection_method_options).grid(row = row, column = 1, sticky = W, pady = 2)
    row += 1
    ###############################
    Label(parent, text='Select Source Folder').grid(row = row, column = 0, sticky = W, pady = 2)
    tk.Button(parent, text="Browse", command=select_source_folder).grid(row = row, column = 1, sticky = W, pady = 2)
    row += 1
    ##############################
    Label(parent, text='Select Destination Folder').grid(row = row, column = 0, sticky = W, pady = 2)
    tk.Button(parent, text="Browse", command=select_dest_folder).grid(row = row, column = 1, sticky = W, pady = 2)
    row += 1
    ##############################
    def toggle_time_array_button_state():
        if global_state['time_array_checkbox_var'].get() == 1:
            time_array_button.config(state=tk.NORMAL)
        else:
            time_array_button.config(state=tk.DISABLED)

    def toggle_Tf_button_state():
        if global_state['Tf_array_checkbox_var'].get() == 1:
            Tf_array_button.config(state=tk.NORMAL)
        else:
            Tf_array_button.config(state=tk.DISABLED)

    def toggle_Tamb_button_state():
        if global_state['Tamb_array_checkbox_var'].get() == 1:
            Tamb_array_button.config(state=tk.NORMAL)
        else:
            Tamb_array_button.config(state=tk.DISABLED)

    global_state['time_array_checkbox_var'] = tk.IntVar()
    Label(parent, text='Enable Time Array File').grid(row = row, column = 0, sticky = W, pady = 2)
    time_array_checkbox = tk.Checkbutton(parent, text="", variable=global_state['time_array_checkbox_var'], command=toggle_time_array_button_state)
    time_array_checkbox.grid(row=row, column=0, sticky=E, pady=2)

    time_array_button = tk.Button(parent, text="Browse", command=select_time_array_csv, state=tk.DISABLED)
    time_array_button.grid(row=row, column=1, sticky=W, pady=2)
    toggle_time_array_button_state()
    row += 1
    ###########################
    global_state['Tf_array_checkbox_var'] = tk.IntVar()
    Label(parent, text='Film Temperature File').grid(row = row, column = 0, sticky = W, pady = 2)
    Tf_array_checkbox = tk.Checkbutton(parent, text="", variable=global_state['Tf_array_checkbox_var'], command=toggle_Tf_button_state)
    Tf_array_checkbox.grid(row=row, column=0, sticky=E, pady=2)

    Tf_array_button = tk.Button(parent, text="Browse", command=select_DAQ_excel, state=tk.DISABLED)
    Tf_array_button.grid(row=row, column=1, sticky=W, pady=2)
    toggle_Tf_button_state()
    row += 1
    ######################################################
    global_state['Tamb_array_checkbox_var'] = tk.IntVar()
    Label(parent, text='Ambient Temperature File').grid(row = row, column = 0, sticky = W, pady = 2)
    Tamb_array_checkbox = tk.Checkbutton(parent, text="", variable=global_state['Tamb_array_checkbox_var'], command=toggle_Tamb_button_state)
    Tamb_array_checkbox.grid(row=row, column=0, sticky=E, pady=2)

    Tamb_array_button = tk.Button(parent, text="Browse", command=select_DAQ_excel, state=tk.DISABLED)
    Tamb_array_button.grid(row=row, column=1, sticky=W, pady=2)
    toggle_Tamb_button_state()
    row += 1
    #######################################################
    Label(parent, text='Apply Inverse Model').grid(row = row, column = 0, sticky = W, pady = 2)
    tk.Button(parent, text="Run", command=apply_inverse_model).grid(row = row, column = 1, sticky = W, pady = 2)
    row += 1
    ##########################################
    Label(parent, text='Export PNG').grid(row = row, column = 0, sticky = W, pady = 2)
    export_png_button = tk.Button(parent, text="Run",
            command=lambda: export_pngs(parent, int(global_state["constants"]["batch_files"].get())))
    export_png_button.grid(row = row, column = 1, sticky = W, pady = 2)
    row += 1
    ##########################################
    Label(parent, text='Export Video').grid(row = row, column = 0, sticky = W, pady = 2)
    create_video_button = tk.Button(parent, text="Run",
                                    command=lambda: create_video(
                                        global_state['source_folder'],
                                        global_state['dest_folder'],
                                        global_state['source_file_type'].get(),
                                        global_state['dest_file_type'].get(),
                                        float(global_state["constants"]["maximum temperature"].get()),
                                        float(global_state["constants"]["maximum incident heat flux"].get()),
                                        int(global_state["constants"]["time step size"].get()),
                                        int(global_state["constants"]["batch_files"].get())
                                    ))
    create_video_button.grid(row = row, column = 1, sticky = W, pady = 5)
    row += 1
    ##################################
    row += 2
    canvas = tk.Canvas(parent)
    Label(parent, text ='Constant Variables', font = "20").grid(row = row, column = 0)

    scrollbar = tk.Scrollbar(parent, orient="vertical", \
        command=canvas.yview , relief = tk.FLAT , cursor = "arrow" , bd = 5 , bg = "white")
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=scrollable_frame,anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set, width=450, height=200)
    units = ['(K)','','(m)','(m)','(m)','(W/m2-K)','(W/m2-K)','(K)','(kW/m2)','(s)','','']
    # row += 1
    unit_idx = 0
    for key, value in DEFAULTS.items():
        if key == "time derivative method":
            break
        Label(scrollable_frame, text=key).grid(row=row, column=0, sticky=W)
        entry = Entry(scrollable_frame)
        entry.insert(0, str(value))
        entry.grid(row=row, column=1, columnspan=2, sticky=W)
        Label(scrollable_frame, text=units[unit_idx]).grid(row=row, column=2, sticky=W)
        unit_idx += 1
        global_state["constants"][key] = entry
        row += 1
    Label(scrollable_frame, text="# files / batch").grid(row=row, column=0, sticky=W)
    batch_files_entry = Entry(scrollable_frame)
    batch_files_entry.insert(0, "100000")  # Default value
    batch_files_entry.grid(row=row, column=1)
    global_state["constants"]["batch_files"] = batch_files_entry
    row += 1

    canvas.grid(row = row, column = 0, sticky = E, pady = 2)
    scrollbar.grid(row = row, column = 3 ,  sticky = NS, rowspan = 2, pady = 2)