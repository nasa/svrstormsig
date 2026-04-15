import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.widgets import RectangleSelector

def extract_bbox_coordinates():
    coords = [] 
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgreen', alpha=0.3)
    ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)

    # 1. Load country names and store the text objects
    shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    
    country_labels = []
    for country in reader.records():
        name = country.attributes['NAME']
        lon = country.geometry.centroid.x
        lat = country.geometry.centroid.y
        
        txt = ax.text(lon, lat, name, ha='center', va='center', 
                      fontsize=4, color='darkblue', alpha=0.7,
                      transform=ccrs.PlateCarree())
        country_labels.append(txt)

    # 2. Updated Zoom Event Listener (Hide off-screen labels)
    def update_text_size(event_ax):
        # Get the current limits of the zoomed view
        x0, x1 = event_ax.get_xlim()
        y0, y1 = event_ax.get_ylim()
        
        # Ensure correct min/max ordering
        min_x, max_x = min(x0, x1), max(x0, x1)
        min_y, max_y = min(y0, y1), max(y0, y1)
        
        view_width = max_x - min_x
        
        # Determine font size based on zoom level
        if view_width > 200:   size = 2
        elif view_width > 100: size = 4
        elif view_width > 50:  size = 6
        else:                  size = 12
            
        # Loop through text labels and check their position
        for txt in country_labels:
            txt_x, txt_y = txt.get_position()
            
            # If the text's coordinate is inside our current zoom window...
            if (min_x <= txt_x <= max_x) and (min_y <= txt_y <= max_y):
                txt.set_visible(True)      # Turn it on
                txt.set_fontsize(size)     # Update size
            else:
                txt.set_visible(False)     # Turn it off to save resources!

    # Connect the zoom events to our function
    ax.callbacks.connect('xlim_changed', update_text_size)
    ax.callbacks.connect('ylim_changed', update_text_size)

    # 3. Box Drawing Logic
    def onselect(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        coords.clear()
        coords.extend([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
        print(f"Selected Box: {coords}")

    toggle_selector = RectangleSelector(
        ax, onselect, useblit=True, button=[1], interactive=True,
        props=dict(facecolor='red', edgecolor='red', alpha=0.3, fill=True)
    )

    plt.title("1. Use Toolbar to Zoom/Pan  |  2. Turn off Zoom tool  |  3. Draw Box")
    plt.tight_layout() 
    plt.show(block=True) 

    return(coords if coords else None)

# Run it
if __name__ == "__main__":
    my_bbox = extract_bbox_coordinates()
    if my_bbox:
        print(f"\nFinal extracted coordinates to use in script: {my_bbox}")