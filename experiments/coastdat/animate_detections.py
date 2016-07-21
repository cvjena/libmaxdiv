from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import sys, os.path, re


class CoastDatAnimation(animation.TimedAnimation):

    def __init__(self, time_start, time_end, detection):

        self.det = detection
        if time_start.year != time_end.year:
            raise ValueError('Beginning and end of the detection must be in the same year.')

        year = time_start.year
        time_start = int((time_start - datetime(year, 1, 1, 1)).total_seconds() / 3600)
        time_end = int((time_end - datetime(year, 1, 1, 1)).total_seconds() / 3600)
        year -= 1958

        self.fig = plt.figure()
        self.fig.set_figwidth(20)
        self.fig.set_figheight(4)
        self.ax = [self.fig.add_subplot(131, title = 'Wind Speed'), self.fig.add_subplot(132, title = 'Significant Wave Height'), self.fig.add_subplot(133, title = 'Mean Period')]
        self.fig.subplots_adjust(left = 0.01, bottom = 0, right = 0.99, top = 1, wspace = 0.2, hspace = None)

        self.data, self.vmin, self.vmax, self.m = [], [], [], []
        for i, v in enumerate(('ff', 'hs', 'mp')):
            dataset = Dataset('/home/barz/anomaly-detection/CoastDat-raw/{v}/coastDat-1_Waves_{v}_{y:03d}.nc'.format(v = v, y = year))
            
            if i == 0:
                timestep_first = max(0, time_start - 24)
                timestep_last = min(dataset.variables[v].shape[0], time_end + 24)
                self.time_start = time_start - timestep_first
                self.time_end = time_end - timestep_first

                lats = dataset.variables['lat'][:]
                lons = dataset.variables['lon'][:]
            
            self.data.append(dataset.variables[v][timestep_first:(timestep_last+1), :])
            self.vmin.append(self.data[i].min())
            self.vmax.append(self.data[i].max())

            self.m.append(Basemap(llcrnrlat = lats[0], llcrnrlon = lons[0], urcrnrlat = lats[-1], urcrnrlon = lons[-1], ax = self.ax[i]))
            self.m[-1].drawmapboundary(fill_color = '0.3')

        self.bbox = [None] * len(self.ax)
        self.im = [None] * len(self.ax)
        self.cb = [None] * len(self.ax)

        self.lons, self.lats = np.meshgrid(lons, lats)

        animation.TimedAnimation.__init__(self, self.fig, interval = 100, repeat = False, blit = True)

    def __del__(self):
        plt.close(self.fig)

    def _draw_frame(self, framedata):
        
        for i in range(len(self.ax)):
            self.im[i] = self.m[i].pcolormesh(self.lons, self.lats, self.data[i][framedata,:].squeeze(), shading = 'flat', cmap = plt.cm.jet, vmin = self.vmin[i], vmax = self.vmax[i], latlon = True)
            if framedata == 0:
                self.cb[i] = self.m[i].colorbar(self.im[i], 'bottom', size = '5%', pad = '2%')
        
            self.bbox[i] = self.m[i].plot([self.det[0][1], self.det[0][1], self.det[1][1], self.det[1][1], self.det[0][1]], [self.det[0][0], self.det[1][0], self.det[1][0], self.det[0][0], self.det[0][0]], color = 'red' if (framedata >= self.time_start) and (framedata <= self.time_end) else 'blue', linewidth = 2, latlon = True)[0]
        
        self._drawn_artists = self.im + self.cb + self.bbox

    def new_frame_seq(self):
        return iter(range(self.data[0].shape[0]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} <output-file-from-detect_coastdat.py>'.format(sys.argv[0]))
        exit()

    with open(sys.argv[1]) as f:
        detfile = f.read()

    detRE = re.compile('TIMEFRAME:\\s+([0-9-]+ [0-9:]+) - ([0-9-]+ [0-9:]+)\\s+LOCATION:\\s+(-?[0-9.]+) N, (-?[0-9.]+) E - (-?[0-9.]+) N, (-?[0-9.]+) E')

    for i, match in enumerate(detRE.finditer(detfile)):
        print('#{}'.format(i+1))
        time_start = datetime.strptime(match.groups()[0], '%Y-%m-%d %H:%M:%S')
        time_end = datetime.strptime(match.groups()[1], '%Y-%m-%d %H:%M:%S')
        loc_start = (float(match.groups()[2]), float(match.groups()[3]))
        loc_end = (float(match.groups()[4]), float(match.groups()[5]))
        try:
            ani = CoastDatAnimation(time_start, time_end, [loc_start, loc_end])
            ani.save('{}_{:02d}.gif'.format(os.path.splitext(sys.argv[1])[0], i), dpi = 60)
            del ani
        except Exception as e:
            print('Animating detection {} failed: {}'.format(i+1, e))
