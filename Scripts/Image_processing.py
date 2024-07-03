import tkinter as tk
from tkinter import filedialog , messagebox
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import h5py
import math
import csv
from datetime import timedelta
from tqdm import tqdm

def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        pts.append((x, y))
    for p in pts:
        cv2.circle(img, p, thickness, color, -1)

def crop_image(image, pts):
    rect = np.array(pts, dtype='float32')
    width = np.max([np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])])
    height = np.max([np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])])
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))
    return warped

def save_image_to_csv(image, original_filename, dest_folder, padded_names_dict):
    original_to_padded = {value: key for key, value in padded_names_dict.items()}
    
    padded_name = original_to_padded.get(original_filename)
    if padded_name is None:
        raise ValueError(f"Padded name for original filename '{original_filename}' not found.")
    
    csv_filename = os.path.join(dest_folder, padded_name)
    
    pd.DataFrame(image).to_csv(csv_filename, header=False, index=False, float_format='%.15f')

def define_destination(points):
    x_coords, y_coords = zip(*points)
    top_left = (min(x_coords), min(y_coords))
    top_right = (max(x_coords), min(y_coords))
    bottom_right = (max(x_coords), max(y_coords))
    bottom_left = (min(x_coords), max(y_coords))
    return [top_left, top_right, bottom_right, bottom_left]

def on_click(event, points, ax):
    if event.inaxes is not None:
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        ax.plot(x, y, 'ro')  # mark the point
        plt.draw()

def on_key(event, points, confirm, fig):
    if event.key == 'enter':
        confirm[0] = True
        plt.close(fig)
    elif event.key == 'escape':
        confirm[0] = False
        plt.close(fig)

def display_and_select_points(image_data, title="Select Points"):
    points = []
    confirm = [False] 
    fig, ax = plt.subplots()
    ax.set_title(title)
    cp = ax.contourf(image_data,100)
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, points, ax))
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, points, confirm, fig))
    plt.show()
    plt.close(fig)
    return points, confirm[0]

class IPA:

    def __init__(self, parent):
        self.source_folder = None
        self.dest_folder = None
        # self.csv_file = None
        self.total_csv_size = 0
        # self.csv_files = []
        self.num_h5_files = 0
        self.h5py_file_index = 0
        self.save_option = tk.StringVar(parent)
        self.save_option.set("h5py")  
        ##########################
        self.row = 0
        ##########################
        tk.Label(parent, text="Input frame rates per second:").grid(row = self.row, column = 0, sticky = "W", pady = 2)
        self.source_fps_entry = tk.Entry(parent)
        self.source_fps_entry.insert(0, "30")  # Default value or an example value
        self.source_fps_entry.grid(row = self.row, column = 2, sticky = "W", pady = 2)
        self.row += 1
        ###########################
        tk.Label(parent, text="Output frame rates per second:").grid(row = self.row, column = 0, sticky = "W", pady = 2)
        self.dest_fps_entry = tk.Entry(parent)
        self.dest_fps_entry.insert(0, "30")  
        self.dest_fps_entry.grid(row = self.row, column = 2, sticky = "W", pady = 2)
        self.decimal_delimiter = tk.StringVar(parent, value='.')  # Default to .
        self.cell_delimiter = tk.StringVar(parent, value=',')    # Default to ,
        self.row += 1
        ###########################
        tk.Label(parent, text="Skip Rows from Source:").grid(row = self.row, column = 0, sticky = "W", pady = 2)
        self.skiprows = tk.Entry(parent)
        self.skiprows.insert(0, "10")  # Default value to 0, change if needed
        self.skiprows.grid(row = self.row, column = 2, sticky = "W", pady = 2)
        self.row += 1       
        #########################
        tk.Label(parent, text="Decimal Delimiter:").grid(row = self.row, column = 0, sticky = "W", pady = 2)
        self.decimal_delimiter_entry = tk.Entry(parent, textvariable=self.decimal_delimiter)
        self.decimal_delimiter_entry.grid(row = self.row, column = 2, sticky = "W", pady = 2)
        self.row += 1
        #########################
        tk.Label(parent, text="Cell Delimiter:").grid(row = self.row, column = 0, sticky = "W", pady = 2)
        self.cell_delimiter_entry = tk.Entry(parent, textvariable=self.cell_delimiter)
        self.cell_delimiter_entry.grid(row = self.row, column = 2, sticky = "W", pady = 2)
        self.row += 1
        #########################
        self.enable_custom_encoding_var = tk.IntVar()
        tk.Label(parent, text="Custom CSV Encoding").grid(row=self.row, column=0, sticky="W", pady=2)
        self.custom_csv_encoding_checkbox = tk.Checkbutton(parent, text="", variable=self.enable_custom_encoding_var, command=self.toggle_csv_encoding_entry_state)
        self.custom_csv_encoding_checkbox.grid(row=self.row, column=1, pady=2)
        # StringVar for the Entry widget
        self.custom_encoding_var = tk.StringVar()
        # Entry for custom CSV encoding
        self.custom_csv_encoding_entry = tk.Entry(parent, textvariable=self.custom_encoding_var)
        self.custom_csv_encoding_entry.grid(row=self.row, column=2, pady=2, sticky='W')
        self.custom_csv_encoding_entry.config(state=tk.DISABLED)  # Initially disabled
        # Increment row for next widget
        self.row += 1
        #########################
        tk.Label(parent, text="Select Source Directory:").grid(row = self.row, column = 0, sticky = "W", pady = 2)
        tk.Button(parent, text="Browse", command=self.select_source_folder).grid(row = self.row, column = 2, sticky = "W", pady = 2)
        self.row += 1
        tk.Label(parent, text="Select Destination Directory:").grid(row = self.row, column = 0, sticky = "W", pady = 2)
        tk.Button(parent, text="Browse", command=self.select_dest_folder).grid(row = self.row, column = 2, sticky = "W", pady = 2)
        self.row += 1
        tk.Label(parent, text="Select a Sample CSV File:").grid(row = self.row, column = 0, sticky = "W", pady = 2)
        tk.Button(parent, text="Browse", command=self.select_csv_file).grid(row = self.row, column = 2, sticky = "W", pady = 2)
        self.row += 1
        #########################
        self.Time_array_checkbox_var = tk.IntVar()
        tk.Label(parent, text='Variable Time Array Folder').grid(row = self.row, column = 0,columnspan=2, sticky = "W", pady = 2)
        self.Time_array_checkbox = tk.Checkbutton(parent, text="", variable=self.Time_array_checkbox_var, command=self.toggle_Time_button_state)
        self.Time_array_checkbox.grid(row=self.row, column=1, pady=2)
        self.Time_array_button = tk.Button(parent, text="Browse", command=self.select_Folder_for_Time_Array, state=tk.DISABLED)
        self.Time_array_button.grid(row=self.row, column=2, sticky="W", pady=2)
        self.Time_array_run_button = tk.Button(parent, text="Assemble", command=self.execute_time_array_assembly, state=tk.DISABLED)
        self.Time_array_run_button.grid(row = self.row, column = 2, sticky = "E", pady = 2)
        self.toggle_Time_button_state()
        self.row += 1
        #########################
        tk.Label(parent, text="Output Type:").grid(row = self.row, column = 0, sticky = "W", pady = 2)
        save_options = ["csv", "h5py"]
        self.option_menu = tk.OptionMenu(parent, self.save_option, *save_options, command=self.on_format_change)
        self.option_menu.grid(row = self.row, column = 2, sticky = "W", pady = 2)
        self.row += 1
        #########################
        tk.Button(parent, text="Run", command=self.process_images).grid(row = self.row, column = 1, sticky = "E", pady = 2)
        self.row += 1
        #########################

    def read_custom_csv(self):
        file_path = self.csv_file
        decimal = self.decimal_delimiter.get()
        cell = self.cell_delimiter.get()
        sr = int(self.skiprows.get())
        encoding = self.custom_encoding_var.get()
        try:
            if self.enable_custom_encoding_var.get() == 1:
                return pd.read_csv(file_path, sep=cell, skiprows= sr, decimal=decimal, encoding=encoding, header=None).apply(pd.to_numeric, errors='coerce')
            else:
                return pd.read_csv(file_path, decimal=decimal, sep=cell, skiprows= sr , header=None)
        except pd.errors.ParserError as e:
            print(f"ParserError in file {file_path}: {e}")
            return None

    def toggle_Time_button_state(self):
        if self.Time_array_checkbox_var.get() == 1:
            self.Time_array_button.config(state=tk.NORMAL)
            self.Time_array_run_button.config(state=tk.NORMAL)
        else:
            self.Time_array_button.config(state=tk.DISABLED)
            self.Time_array_run_button.config(state=tk.DISABLED)

    def toggle_csv_encoding_entry_state(self):
        if self.enable_custom_encoding_var.get() == 1:
            self.custom_csv_encoding_entry.config(state=tk.NORMAL)
        else:
            self.custom_csv_encoding_entry.config(state=tk.DISABLED)

    def process_file(self , file_path , decimal, cell, sr, encoding , points_step1, destination_corners):
        if self.enable_custom_encoding_var.get() == 1:
            contour_data = pd.read_csv(file_path, sep=cell, skiprows= sr, decimal=decimal, encoding=encoding, header=None)
            contour_data = contour_data.apply(pd.to_numeric, errors='coerce')
        else:
            contour_data = pd.read_csv(file_path, decimal=decimal, sep=cell, skiprows= sr , header=None)
            
        contour_data = contour_data.fillna(0)
        contour_data = contour_data.clip(lower=0)  # Removed upper=255 to avoid capping

        contour_array = contour_data.to_numpy(dtype=float)
        # Normalize only for displaying in the GUI
        contour_image_for_display = cv2.normalize(contour_array, None, 0, 255, cv2.NORM_MINMAX)
        contour_image_for_display = np.uint8(contour_image_for_display)
        contour_image_for_display = cv2.cvtColor(contour_image_for_display, cv2.COLOR_GRAY2BGR)

        # Use the original array for processing
        src_points = np.array(points_step1, dtype='float32')
        dst_points = np.array(destination_corners, dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(contour_array, M, (contour_array.shape[1], contour_array.shape[0]))

        if np.count_nonzero(warped_image) == 0:
            print(f"Failed to apply perspective transform for {file_path}")
            return

        cropped_image = crop_image(warped_image, dst_points)
        if np.count_nonzero(cropped_image) == 0:
            print(f"Failed to crop the image for {file_path}")
            return

        if self.save_option.get() == "csv":
            original_filename = os.path.basename(file_path)
            save_image_to_csv(cropped_image, original_filename, self.dest_folder, self.padded_names_dict)
        elif self.save_option.get() == "h5py":
            self.save_image_to_h5py(cropped_image, os.path.basename(file_path), self.dest_folder, self.padded_names_dict)

    def on_format_change(self, value):
        self.update_h5py_size_entry_state()

    def update_h5py_size_entry_state(self):
        if self.save_option.get() == "h5py":
            self.h5py_size_label.config(state="normal")
            self.h5py_files_entry.config(state="normal")
        else:
            self.h5py_size_label.config(state="disabled")
            self.h5py_files_entry.config(state="disabled")

    def select_source_folder(self):
        self.source_folder = filedialog.askdirectory()
        print("Selected Source Folder:", self.source_folder)

    def select_dest_folder(self):
        self.dest_folder = filedialog.askdirectory()
        print("Selected Destination Folder:", self.dest_folder)

    def select_Folder_for_Time_Array(self):
        self.Time_Array_folder = filedialog.askdirectory()
        print("Selected Time Array Folder:", self.Time_Array_folder)

    def assemble_time_array(self):
        time_row = 3
        time_index , time_array , time_us = [] , [] , []
        # Get a list of all the .txt files in the current directory
        files = sorted(glob.glob(self.source_folder +"/"+ "*.csv"))

        # Iterate over the list and process each file
        for file_idx , file in tqdm(enumerate(files), desc = 'Assembling time array . . .'):
            with open(file, 'r+') as csvfile:
                reader = csv.reader(csvfile)
                for row_idx , row in enumerate(reader):
                    if row_idx == time_row-1:
                        time_str = (row[0][7:])
                        break
            parts = time_str.split(":")
            
            days = int(parts[0])
            hours = int(parts[1])
            minutes = int(parts[2])
            seconds, microseconds = map(int, parts[3].split("."))
            if file_idx == 0 :
                duration = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
            duration2 = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)

            time_index.append(str(duration2 - duration) + '_'+str(file_idx).zfill(9))
            time_array.append(duration2 - duration)
            time_us.append((duration2 - duration).total_seconds() * 1000000)
        source_fps = int(self.source_fps_entry.get())
        dest_fps = int(self.dest_fps_entry.get())
        fps_ratio = max(int(source_fps / dest_fps), 1)  # Ensure fps_ratio is at least 1
        pd.DataFrame({'t(microseconds)':time_us[::fps_ratio]}).to_csv(self.Time_Array_folder +'/Time_array.csv')

    def select_csv_file(self):
        self.csv_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        print("Selected sample CSV File:", self.csv_file)

    def process_images(self):
        if self.source_folder and self.dest_folder and self.csv_file:
            print("Processing images...")
            self.run_image_processing()
        else:
            print("Please select all required paths before processing.")

    def execute_time_array_assembly(self):
        self.assemble_time_array()
        print(glob.glob('Time array saved in ' + self.Time_Array_folder))

    def pad_and_sort_filenames(self):
        padded_names_dict = {}
        for filename in glob.glob(os.path.join(self.source_folder, '*.csv')):
            base_name = os.path.basename(filename)
            parts = base_name.split('_')
            index_part = parts[-1].split('.')[0]  # Assuming the index is always before ".csv"
            padded_index = index_part.zfill(9)  # Pad the index with 9 digits
            padded_name = "_".join(parts[:-1] + [padded_index + ".csv"])
            padded_names_dict[padded_name] = base_name

        # Sort the dictionary by padded names and store it for later use
        self.padded_names_dict = dict(sorted(padded_names_dict.items()))

    def calculate_total_csv_size_and_divide_files(self):
        self.pad_and_sort_filenames()
        self.csv_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self.source_folder, '*.csv'))])

    def run_image_processing(self):
        source_fps = int(self.source_fps_entry.get())
        dest_fps = int(self.dest_fps_entry.get())
        fps_ratio = int(max((source_fps / dest_fps), 1))  # Ensure fps_ratio is at least 1
        
        decimal = self.decimal_delimiter.get()
        cell = self.cell_delimiter.get()
        sr = int(self.skiprows.get())
        contour_data = self.read_custom_csv()
        if contour_data is None:
            print(f"Failed to read file: {self.csv_file}")
            return

        self.contour_array = contour_data.to_numpy()

        # Step 1: Select source points
        self.points_step1, confirmed = display_and_select_points(self.contour_array, "Select the Four Corners")
        if not confirmed:
            print("Selection cancelled.")
            return

        # Step 2: Select destination points
        self.points_step2, confirmed = display_and_select_points(self.contour_array, "Select a Point on Each Edge")
        if not confirmed:
            print("Selection cancelled.")
            return

        if len(self.points_step1) != 4 or len(self.points_step2) < 4:
            print("Error: Invalid number of points selected.")
            return

        # Define destination corners based on selected points
        self.destination_corners = define_destination(self.points_step2)

        # Perspective transform and crop
        src_points = np.array(self.points_step1, dtype='float32')
        dst_points = np.array(self.destination_corners, dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(self.contour_array, M, (self.contour_array.shape[1], self.contour_array.shape[0]))
        cropped_image = crop_image(warped_image, dst_points)

        # Show the cropped image for approval
        plt.figure("Cropped Image")
        plt.imshow(cropped_image, cmap='gray')
        plt.show()

        user_approved = messagebox.askyesno("Confirm", "Do you approve the processed sample image?")
        if not user_approved:
            return
        self.calculate_total_csv_size_and_divide_files()
        # Batch processing with step of fps_ratio
        for i, file_name in tqdm(enumerate(self.csv_files) , desc = 'Image Processing . . .'):
            if i % fps_ratio == 0:  # Process files based on fps_ratio
                file_path = os.path.join(self.source_folder, file_name)
                # print(f"Processing {file_path}")
                self.process_file(file_path, self.decimal_delimiter.get(),\
                     self.cell_delimiter.get(), int(self.skiprows.get()), \
                        self.custom_encoding_var.get(),src_points, dst_points)
        print("Processing Completed Successfully!")

    def select_points(self):
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', lambda event, x, y, flags, param: self.click_event_step1(event, x, y, flags, param))
        cv2.imshow('Image', self.contour_image)

        while len(self.points_step1) < 4:
            cv2.waitKey(1)
            if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.setMouseCallback('Image', lambda event, x, y, flags, param: self.click_event_step2(event, x, y, flags, param))
        cv2.imshow('Image', self.contour_image)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                if self.points_step2:
                    self.destination_corners = define_destination(self.points_step2)
                    print("Step 2 - Destination corners: ", self.destination_corners)
                    break
                else:
                    print("Select additional points and press Enter.")
            elif key == 27 or cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:  # ESC key or window closed
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()

    def show_result_and_confirm(self):
        src_points = np.array(self.points_step1, dtype='float32')
        dst_points = np.array(self.destination_corners, dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        HEIGHT, WIDTH = self.contour_image.shape[:2]
        warped_image = cv2.warpPerspective(self.contour_image, M, (WIDTH, HEIGHT))
        cropped_image = crop_image(warped_image, dst_points)

        cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return messagebox.askyesno("Confirm", "Do you approve the processed image?")

    def click_event_step1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points_step1) < 4:
                self.points_step1.append((x, y))
                cv2.circle(self.contour_image, (x, y), 5, (255, 255, 255), -1)
                if len(self.points_step1) > 1:
                    draw_dotted_line(self.contour_image, self.points_step1[-2], self.points_step1[-1], (255, 255, 255), thickness=1, gap=20)
                cv2.imshow('Image', self.contour_image)
            if len(self.points_step1) == 4:
                draw_dotted_line(self.contour_image, self.points_step1[-1], self.points_step1[0], (255, 255, 255), thickness=1, gap=20)
                cv2.setMouseCallback('Image', lambda *args : None)

    def click_event_step2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points_step2.append((x, y))
            cv2.circle(self.contour_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image', self.contour_image)
    
    def save_image_to_h5py(self, image, original_filename, dest_folder, padded_names_dict):
        original_to_padded = {value: key for key, value in padded_names_dict.items()}
        padded_name = original_to_padded.get(original_filename)
        if padded_name is None:
            raise ValueError(f"Padded name for original filename '{original_filename}' not found.")

        h5py_filename = os.path.join(dest_folder, 'processed_temperature_array.h5')
        
        with h5py.File(h5py_filename, 'a') as h5py_file:
            if padded_name in h5py_file:
                print(f"Warning: Dataset {padded_name} already exists in {h5py_filename}.")
            else:
                h5py_file.create_dataset(padded_name, data=image, dtype='float64')
                # print(f"Dataset '{padded_name}' saved in H5PY file.")