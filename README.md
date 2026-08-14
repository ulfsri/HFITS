# HFITS: An Analysis Tool for Calculating Heat Flux to Planar Surfaces using Infrared Thermography

Developed by research engineers at the [Fire Safety Research Institute](https://fsri.org/), part of UL Research Institutes, [HFITS](https://fsri.org/programs/heat-flux-using-infrared-thermography-over-surfaces-hfits) is a software tool that is intended to support experimental measurements of heat flux over planar surfaces using infrared thermography. This technique enables spatially and temporally resolved heat flux measurements at a greater resolution than arrays of traditional point sensors. The target audience is researchers and engineers in thermal engineering disciplines. 

HFITS consists of two main components: pre-processing of infrared thermograms (obtained from heat transfer experiments), and inverse heat transfer analysis (to deduce heat flux over the planar surface in those experiments). The software offers comprehensive functionalities, including support for custom thermogram formats, metadata handling, a graphical interface for selection of regions of interest, the ability to import additional temperature measurements to enhance convective heat transfer estimates, and the exporting of both computed field data and contour videos. Please refer to the software [MANUAL](https://github.com/ulfsri/HFITS/blob/main/MANUAL.pdf) for additional information.

## Note on file names for PNG and video export

The export functions on the 'Inverse Heat Transfer' tab look for fixed file names, and will not find files under any other name:

- The source folder must contain `processed_temperature_array.h5`. This file is created by the 'Image Processing' tab. If the temperature data comes from another source (e.g., a downloaded dataset), the file must be renamed to `processed_temperature_array.h5` before exporting.
- The destination folder must contain `Incident_Radiative_HF.h5`. This file is created by 'Apply Inverse Model', so 'Export PNG' and 'Export Video' only work after that step has been completed.
- The source and destination folders should be kept separate (e.g., `T_proc` and `Q_proc`, as described in the [MANUAL](https://github.com/ulfsri/HFITS/blob/main/MANUAL.pdf)).
- A renamed copy of `Surface_Temperature.h5` should not be used as the temperature input for plotting: it stores temperatures in Kelvin, while the plots use a Celsius scale (0 to 'maximum temperature'), so the plots will look wrong (near-uniform color, hottest regions blank). The 'Image Processing' output should be used instead, which is in the same unit as the input data.
- The CSV thermogram files are not required for PNG/video export when the file types are set to h5py — only the two `.h5` files above are needed.

###

This software has been made available to the community under the GPL-3.0 license. It is under active development, and users are encouraged to contact the software developers with questions and feature requests.

Please refer to the following resources for additional information and examples of the application of HFITS:

- P. Dehghani and M.J. DiDomizio (2024) **HFITS: An analysis tool for calculating heat flux to planar surfaces using infrared thermography**, *SoftwareX*, Volume 28, 101934. [doi:10.1016/j.softx.2024.101934](https://doi.org/10.1016/j.softx.2024.101934).
  - [Article Download (PDF)](https://www.softxjournal.com/action/showPdf?pii=S2352-7110%2824%2900304-2)
- M.J. DiDomizio and J.W. Butta (2024) **Measurement of Heat Transfer and Fire Damage Patterns on Walls for Fire Model Validation**, Technical Report, UL Research Institutes, Fire Safety Research Institute, Columbia, MD. [doi:10.54206/102376/HNKR9109](https://dx.doi.org/10.54206/102376/HNKR9109)
  - [Report Download (PDF)](https://d1gi3fvbl0xj2a.cloudfront.net/2024-07/Measurement%20of%20Heat%20Transfer%20and%20Fire%20Damage%20Patterns%20on%20Walls%20for%20Fire%20Model%20Validation%20240709_0.pdf)
- N. Sauer (2024) **FSRI Experimental Investigation of EV Fires**, Conference Presentation, *SFPE Engineering Solutions Symposium - Progress with Li-Ion Battery Fire Safety: Engineering Solutions to Mobility and Storage Hazards*, Phoenix, AZ.
  - [Conference Information](https://www.sfpe.org/events-education/liveeducation/in-personeducation/liionsymposium)
- P. Dehghani, M. DiDomizio, A. Barowy, and N. Sauer (2024) **Measuring Heat Exposure to the Immediate Surroundings of an Electric Vehicle Fire**, Conference Presentation, *SFPE 2024 Annual Conference and Expo*, Louisville, KY.
  - [Conference Information](https://www.sfpe.org/annual24/home)