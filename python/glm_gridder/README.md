# Gridder Overview (for glm_gridder.py)
## Data pipeline overview
The data pipeline in this script is set up to take input GLM data files from the input directory `raw_dir`, input it into the gridder to layer and reformat the flash information, and output the processed data to the file stored as the `out_dir` variable. Once the data has been sent to the output file, the data in the output file can be plotted to show the accumulation of the lightning flashes over time. <br/>

The general overview of the data pipeline is as follows: <br/>
GLM data files → input directory  (`raw_dir`)→ gridder → out file (`out_dir`)  → plotting

The name `out_dir` is somewhat of a misnomer given that `out_dir` must be the location of a file such as `out_dir/gridded_data.nc`

For a more detailed breakdown of the data pipeline and how it is set up, look at the `GLM_gridder` notebook. 

## Gridding and Plotting
The gridder is a function from glm tools that takes the data from each file and re-shapes it into different variables in a netCDF file.

Once the grid has been made, we select the `flash_extent densidty` variable from the output netCDF file. This variable is not part of the original netCDF files and is calculated during gridding. This is the variable that will be used as an input into the AACP-identifying U-net. 

## Problems to look into later
- The gridder is really ram-intensive, so it might be worth looking into the package code to see why it takes up so much memory. 
- When the gridded data is written to the output file, the printed output in the terminal is supressed until the plot is closed and the program terminates. This is likely because the output file doesn't get closed after writing, but when I tried fixing that in the script it didn't work. This is probably another source code issue. 