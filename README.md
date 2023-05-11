# svrstormsig
Please read README_goes_aacp_ot_python_model_run_description.pdf for instructions on install process as well as how to run the software

This software was produced through collaborative work between John W. Cooney, Kristopher M. Bedka, and Charles A. Liles at NASA Langley Research Center.

**Software Use**

This software is intended for research and operational use. Users can contact the software team regarding its use. It is especially recommended that our team be contacted before publication or public presentation. This is the first official release of this software; these products are still undergoing validation, testing, quality control, and debugging. Users are invited to address questions and provide feedback to the contacts below.

**Contact Information**

John Cooney: john.w.cooney@nasa.gov

Kristopher Bedka: kristopher.m.bedka@nasa.gov

**Overview**

Severe weather clouds exhibit distinct patterns in satellite imagery. These patterns, commonly referred to as overshooting cloud tops (OTs) and above anvil cirrus plumes (AACPs), are routinely observed by the human-eye in satellite imagery atop the most intense storms on Earth. With the latest geostationary high spatio-temporal resolution satellite imagers such as, GOES-16 and GOES-17, these severe weather patterns are being observed better than ever before. Advances in deep learning methods allow us to not only detect these signatures but also automate the detection process for real-time applications. Our software code was developed in Python and contains up to 3 model types, U-Net, MultiResU-Net, and AttentionU-Net with varying geostationary satellite channels used as model inputs. The MultiresU-Net model has been found to identify the severe storm signatures better than the other 2 model types and is the default provided in the software release. The models' were trained using a large database of OT and AACP patterns identified from human analysts based on 1-minute GOES-16 ABI imagery. In testing, the models have been shown to successfully detect OTs and AACPs with high precision. The software has also successfully been run for real-time and archived applications. The goal of this software is for users to accurately detect these severe storm signatures for a variety of forecast and research applications.

**Citation Request**

If you use this software in your project, please contact our team prior to publication and cite our contribution.
