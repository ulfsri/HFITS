# HFITS: Heat Flux Measurements using Infrared Thermography and a Plate Sensor

Please follow these steps to install required packages:

1. Install Anaconda from `https://www.anaconda.com/`.
2. Once installed, open terminal and type `pip install opencv-python`.
3. In terminal, type `pip install ffmpeg-python`.
4. Download the repository and copy-paste all raw csv files into the `IHT_Working_Folder/T_raw`.
5. Run `image_rectify_crop.py`.
6. Select Source directory: `IHT_Working_Folder/T_raw`
7. Select Destination directory: `IHT_Working_Folder/T_processed`
8. Select CSV file: Select one of the raw images to identify the four corners of the computational domain:
   a. First select the four corners of the surface by clicking on the appropriate spots.
   b. Select some additional points on the edges to identify the new (destination corners).
   c. Hit `Enter`.
   d. Examine the plotted image. If satisfied, press `ESC` to process all the CSV files inside the folder (or reset the process).
9. Run `GUI_IHT.py`.
10. Adjust the default values shown in the opened window.
11. Select the `Source Folder`: "T_Processed"
12. Select the `Destination Folder`: "Incident Radiation"
13. Select the `Apply Inverse Model to Files`: This will apply the IHT on all the files
14. Select `Create Video`: This will export an mp4 of the image sequence to the destination folder.
