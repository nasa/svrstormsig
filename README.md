# svrstormsig
Please read README_goes_aacp_ot_python_model_run_description.pdf for instructions on install process as well as how to run the software.

This software was produced through collaborative work between John W. Cooney, Kristopher M. Bedka, and Charles A. Liles at NASA Langley Research Center.

**Software Use**

This software is intended for research and operational use. Users can contact the software team regarding its use. It is especially recommended that our team be contacted before publication or public presentation. This is the first official release of this software; these products are still undergoing validation, testing, quality control, and debugging. Users are invited to address questions and provide feedback to the contacts below.

**Contact Information**

John Cooney: john.w.cooney@nasa.gov

Kristopher Bedka: kristopher.m.bedka@nasa.gov

**Overview**

Severe weather clouds exhibit distinct patterns in satellite imagery. These patterns, commonly referred to as overshooting cloud tops (OTs) and above anvil cirrus plumes (AACPs), are routinely observed by the human-eye in satellite imagery atop the most intense storms on Earth. With the latest geostationary high spatio-temporal resolution satellite imagers such as, GOES-16 and GOES-17, these severe weather patterns are being observed better than ever before. Advances in deep learning methods allow us to not only detect these signatures but also automate the detection process for real-time applications. Our software code was developed in Python and contains up to 3 model types, U-Net, MultiResU-Net, and AttentionU-Net with varying geostationary satellite channels used as model inputs. The MultiresU-Net model has been found to identify the severe storm signatures better than the other 2 model types and is the default provided in the software release. The models' were trained using a large database of OT and AACP patterns identified from human analysts based on 1-minute GOES-16 ABI imagery. In testing, the models have been shown to successfully detect OTs and AACPs with high precision. The software has also successfully been run for real-time and archived applications. The goal of this software is for users to accurately detect these severe storm signatures for a variety of forecast and research applications.

Version 2 released on 6 March 2024. LARGE model performance enhancement, particularly with AACP detection. Added A LOT of new model input combinations as possibilities, including TROPDIFF, DIRTYIRDIFF, SNOWICE, and CIRRUS. Added new and improved checkpoint files associated with those model runs. Optimal models and thresholds have been updated since the Version 1 release. Changed how we interpolate the GFS data onto the GOES grid. We now follow the methods outlined in Khlopenkov et al. (2021). New version changes how we decide between daytime and nighttime model runs. Previously we used the maximum solar zenith angle within the domain and checked if that exceeded 85°, however, there was an issue of only nighttime models being used for CONUS domains due to how the satellite views the Earth and the data are gridded. Thus, now the software checks to see if more than 5% of the pixels in the domain have solar zenith angles that exceed 85°. If they do, then the software acts as if the domain is nighttime and if not the software acts as if it is daytime within the domain. Fixed major issue with looping over post-processing dates and the month or year changed during a real-time or archived model run. New version speeds up post-processing for OTs. Set variables to None after using them in order to clear cached memory. Catch and hide RunTime warnings that the User does not need to worry about. Software Users prior to 5 March 2024 will need to pull the latest python files from GitHub. This should be very fast. Users MUST also download the new checkpoint files from 'https://science-data.larc.nasa.gov/LaRC-SD-Publications/2023-05-05-001-JWC/data/ML_data.zip'. Once downloaded and unzipped, replace the old model_checkpoints subdirectory with the latest one that was just downloaded. This directory includes all of the improved model detection files as well as checkpoint files for the new input combination


**Citation Request**

If you use this software in your project, please contact our team prior to publication and cite our contribution.
