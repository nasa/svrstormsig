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

Version 2 released on 6 March 2024. LARGE model performance enhancement, particularly with AACP detection. Added A LOT of new model input combinations as possibilities, including TROPDIFF (10.3 μm - GFS Tropopause Temperature), DIRTYIRDIFF (12.3 μm - 10.3 μm), SNOWICE (1.6 μm), and CIRRUS (1.37 μm). Added new and improved checkpoint files associated with those model runs. Optimal models and thresholds have been updated since the Version 1 release. Changed how we interpolate the GFS data onto the GOES grid. We now follow the methods outlined in Khlopenkov et al. (2021). New version changes how we decide between daytime and nighttime model runs. Previously we used the maximum solar zenith angle within the domain and checked if that exceeded 85°, however, there was an issue of only nighttime models being used for CONUS domains due to how the satellite views the Earth and the data are gridded. Thus, now the software checks to see if more than 5% of the pixels in the domain have solar zenith angles that exceed 85°. If they do, then the software acts as if the domain is nighttime and if not the software acts as if it is daytime within the domain. Fixed major issue with looping over post-processing dates and the month or year changed during a real-time or archived model run. New version speeds up post-processing for OTs. Set variables to None after using them in order to clear cached memory. Catch and hide RunTime warnings that the User does not need to worry about. Software Users prior to 5 March 2024 will need to pull the latest python files from GitHub. This should be very fast. Users MUST also download the new checkpoint files from 'https://science-data.larc.nasa.gov/LaRC-SD-Publications/2023-05-05-001-JWC/data/ML_data.zip'. Once downloaded and unzipped, replace the old model_checkpoints subdirectory with the latest one that was just downloaded. This directory includes all of the improved model detection files as well as checkpoint files for the new input combination. If you have issues downloading and unzipping the file from the browser, run the run_download_model_chkpoint_files.py script included with the software. Users have experienced issues downloading the checkpoint files from the browser and this script has been successful.

Version 3 released on 16 April 2025. MTG-1 satellite added as a capability! Lots of additional files related to MTG data, including intermediate files are now available. See section \ref{mtg_setup} for help getting started using MTG data. GOES-19 satellite was also added as a new capability. In addition, there is a model global performance enhancement, particularly with OT detection. New VIS+TROPDIFF OT model added. The previous model performed worse under testing. TROPDIFF is now the default nighttime model and VIS+TROPDIFF is now the default daytime model. This was done for instances with very cold IR brightness temperatures ($<$ 190 K). These cold temperatures saturated the IR products and caused large regions of OT detections rather than the much more precise OT regions. This default works better for global model applications. It may not work best in every single case but we have found it to be more reliable across seasons and regions. The tropopause temperature calculation has also changed. We now use a 20 pixel radius to smooth the tropopause temperature. There was an issue previously with using too small of a radius which resulted in discontinuities in the tropopause. It made tropopause-relative temperature differences appear larger only due to the discontinuity and not due to the cloud height. This fix removed these discontinuities. Due to issues with downloading the tropopause temperatures from ucar, we added the google cloud as a backup location to download GFS data files from.
**Publication**

The technical background and validation for this software is described here:

Cooney, J. W., Bedka, K. M., Liles, C. A., and Homeyer, C. R. (accepted). Automated Detection of Overshooting Tops and Above Anvil Cirrus Plumes Within Geostationary Imagery Using Deep Learning. Artificial Intelligence for the Earth Systems.

**Model Performance**

Example scenes of recent cases that ran the ML OT and AACP detection algorithm can be found in the examples folder provided. Example data masks from severe storm outbreaks used to train, validate, and test model performance are shown below. The black contours outline OTs, the magenta contours outline confident plume labels (warm anomaly), and the white contours outline plume labels.

![masks](https://github.com/nasa/svrstormsig/blob/main/examples/combined_all_case_ex_masks2.jpg)

Model detections for these same scenes are provided below. The OT model used IR+VIS inputs and the AACP model used IR+VIS+DIRTYIRDIFF inputs. These detection models used the optimal input configurations found during testing. This includes IR+VIS during daytime OT model runs, IR during nighttime OT model runs, and IR+DIRTYIRDIFF during day and night AACP model runs. See Cooney et al. (in review) for more details.

![results](https://github.com/nasa/svrstormsig/blob/main/examples/combined_all_case_ex_results2_ir_vis_dirtyirdiff_plume.jpg)

Some additional examples are provided below while additional cases are provided in the examples folder.

![20240507](https://github.com/nasa/svrstormsig/blob/main/examples/20240507T000728Z_C_full_domain_multimap_model_ot_plume_results.png)

Image above shows IR+VIS OT and IR+DIRTYIRDIFF AACP model detections 20 minutes prior to initial report of rope tornado in Oklahoma that damaged a building at 0034 UTC. In Kansas, hail was observed and winds > 50 mph measured in middle of Kansas at 2344 UTC. In addition, a tornado was estimated from radar at 0048 UTC in northeast Kansas. 

![20240508](https://github.com/nasa/svrstormsig/blob/main/examples/20240508T121248Z_C_full_domain_multimap_model_ot_plume_results.png)

![20240508](https://github.com/nasa/svrstormsig/blob/main/examples/20240508T122224Z_C_full_domain_multimap_model_ot_plume_results.png)

The 2 images above shows IR+VIS OT and IR+DIRTYIRDIFF AACP model detections at time of hail and severe wind reports along the Tennessee and Kentucky border. Hail was observed along border at 1205 and 1239 UTC. Hail observed throughout storm within Tennesse as there were multiple reports after 1400 UTC.

![20240516](https://github.com/nasa/svrstormsig/blob/main/examples/20240516T220240Z_M2_full_domain_multimap_model_ot_plume_results.png)

The image above shows IR+VIS OT and IR+DIRTYIRDIFF AACP model detections an hour before a tornado touched down near Houston, Texas. Wind reports and power outages reported at 2213 UTC.

![20240314](https://github.com/nasa/svrstormsig/blob/main/examples/20240314T011440Z_M2_full_domain_multimap_model_ot_plume_results.png)

The image above shows IR OT and IR+DIRTYIRDIFF AACP model detections during a tornado being observed by law enforcement at 0115 UTC in northeast Kansas. Reports of tornado on the ground at 0133 along border of Kansas and Missouri. Quarter size hail reported at time of detections in northeast Kansas as well. In addition quarter size hail covered the ground in northcentral Kansas (reported at 0110 UTC). Quarter size hail observed and time estimated to 0122 in northwest Missouri.

**Data Availability Statement**

The master set of OT and AACP labels used in this study are available at https://science-data.larc.nasa.gov/LaRC-SD-Publications/2023-05-05-001-JWC/data/master_labels.zip. These labels serve as our “truth”, and “masks” to train, validate and test the ML models. The labelers consisted of 2 experienced analysts and 3 lesser-experienced student analysts trained to identify these features. The various labels were combined to arrive at this optimal “master” label database.

There are 4 classes, ‘Overshoot’ (OT), ‘Plume’, ‘Confident Plume’, and ‘Null’. Confident and non-confident AACPs are differentiated to allow training of the model on the features most apparent to an analyst, which should also be most easily detectable by an ML model. AACPs can be either cold or warm, depending on the level of the tropopause relative to the storm top and their distance downstream from the parent OT (Murillo & Homeyer, 2022). Warm anomaly regions of AACPs, directly adjacent to an OT, are considered confident detections because they are easiest to see in the sandwich imagery. In addition, AACPs can span hundreds of kilometers and the peripheries may not be near to the parent OT. Thus, models may confuse similarly textured anvil regions with AACPs. Null labels were strategically positioned within GOES imagery without OT/AACP to attempt to teach the model the difference between ordinary convection and our desired features. 

**Citation Request**

If you use this software in your project, please contact our team prior to publication and cite our contribution.

**Acknowledgements**
The authors would like to acknowledge Rachael Auth and Cornelia Strube for their contributions in testing the software on various cases to help us better understand the model detections and ultimately improve our product offering. In addition, their contributions helped speed up the release of MTG satellite data in the Version 3 release.