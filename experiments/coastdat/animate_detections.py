import matplotlib
matplotlib.use('Agg') # non-interactive backend for terminal sessions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from datetime import datetime, timedelta
import sys, os.path, re


def coastDatDetectionAnimation(time_start, time_end, detection, storm_name, ani_file):

    if time_start.year != time_end.year:
        raise ValueError('Beginning and end of the detection must be in the same year.')

    year = time_start.year
    base_time = datetime(year, 1, 1, 1)
    time_start = int((time_start - base_time).total_seconds() / 3600)
    time_end = int((time_end - base_time).total_seconds() / 3600)
    year -= 1957

    # Create figure
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(4)
    ax = [fig.add_subplot(131, title = 'Wind Speed'), fig.add_subplot(132, title = 'Significant Wave Height'), fig.add_subplot(133, title = 'Mean Period')]
    fig.subplots_adjust(left = 0.01, bottom = 0, right = 0.99, top = 1, wspace = 0.2, hspace = None)

    # Load data
    data, vmin, vmax, m = [], [], [], []
    for i, v in enumerate(('ff', 'hs', 'mp')):
        dataset = Dataset('/home/barz/anomaly-detection/CoastDat-raw/{v}/coastDat-1_Waves_{v}_{y:03d}.nc'.format(v = v, y = year))
        
        if i == 0:
            timestep_first = max(0, time_start - 24)
            timestep_last = min(dataset.variables[v].shape[0], time_end + 24)
            time_start = time_start - timestep_first
            time_end = time_end - timestep_first

            lats = dataset.variables['lat'][:]
            lons = dataset.variables['lon'][:]
        
        data.append(dataset.variables[v][timestep_first:(timestep_last+1), :])
        vmin.append(dataset.variables[v][:].min())
        vmax.append(dataset.variables[v][:].max())

        # Create map
        m.append(Basemap(llcrnrlat = lats[0], llcrnrlon = lons[0], urcrnrlat = lats[-1], urcrnrlon = lons[-1], resolution = None, ax = ax[i]))
        m[-1].drawmapboundary(fill_color = '0.3')

    im = [None] * len(ax)
    cb = [None] * len(ax)
    bbox = [None] * len(ax)

    lons, lats = np.meshgrid(lons, lats)

    def animate(t):
    
        for i in range(len(ax)):
            
            # Time
            dateline = str(base_time + timedelta(seconds = (timestep_first + t) * 3600))
            fig.suptitle('{} ({})'.format(dateline, storm_name) if storm_name else dateline)
            
            # Data heatmap
            if im[i] is not None:
                im[i].remove()
                im[i] = None
            im[i] = m[i].pcolormesh(lons, lats, data[i][t,:,:].squeeze(), shading = 'flat', cmap = plt.cm.jet, vmin = vmin[i], vmax = vmax[i], latlon = True)
            
            # Color bar
            if (cb[i] is None) and (im[i] is not None):
                cb[i] = m[i].colorbar(im[i], 'bottom', size = '5%', pad = '2%')
            
            # Detection bounding box
            if t == time_start:
                bbox[i] = m[i].plot(
                    [detection[0][1], detection[0][1], detection[1][1], detection[1][1], detection[0][1]],
                    [detection[0][0], detection[1][0], detection[1][0], detection[0][0], detection[0][0]],
                    color = 'red', linewidth = 3, latlon = True
                )[0]
            if t == time_end:
                bbox[i].remove()
                bbox[i] = None
    
    # Save animation
    animation.FuncAnimation(fig, animate, range(data[0].shape[0]), interval = 100, repeat = False, blit = False).save(ani_file, dpi = 60, bitrate = 1200)
    plt.close(fig)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} <output-file-from-detect_coastdat.py>'.format(sys.argv[0]))
        exit()

    with open(sys.argv[1]) as f:
        detfile = f.read()

    detRE = re.compile('TIMEFRAME:\\s+([0-9-]+ [0-9:]+) - ([0-9-]+ [0-9:]+)\\s+LOCATION:\\s+(-?[0-9.]+) N, (-?[0-9.]+) E - (-?[0-9.]+) N, (-?[0-9.]+) E(\\s+SCORE:\\s+[0-9.-]+\\s+IDENT:\\s+([^(]+)\\s\\()?')

    for i, match in enumerate(detRE.finditer(detfile)):
        print('#{}'.format(i+1))
        time_start = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        time_end = datetime.strptime(match.group(2), '%Y-%m-%d %H:%M:%S')
        loc_start = (float(match.groups(3)), float(match.groups(4)))
        loc_end = (float(match.groups(5)), float(match.groups(6)))
        storm_name = match.group(8)
        try:
            coastDatDetectionAnimation(time_start, time_end, [loc_start, loc_end], storm_name, '{}_{:02d}.gif'.format(os.path.splitext(sys.argv[1])[0], i))
        except Exception as e:
            print('Animating detection {} failed: {}'.format(i+1, e))
