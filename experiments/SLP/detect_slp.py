""" Runs the MDI algorithm on the Sea Level Pressure dataset. """

import os.path
from utils import *
from maxdiv.libmaxdiv_wrapper import *
import argparse
import time
import json

default_csv = '../coastdat/historic_storms.csv'
default_gridspec = getSLPGridSpec()

parser = argparse.ArgumentParser()
parser.add_argument('--matfile', help='MAT file containing the tensor', default='SLP_ATL.mat')
parser.add_argument('--tensorvar', help='Variable in the MAT containing the tensor', default='pres')
parser.add_argument('--eventscsv', help='CSV file containing historic events', default=default_csv)
parser.add_argument('--yearoffset', help='Grid specification: year offset', default=default_gridspec['year_offs'], type=int)
parser.add_argument('--daystep', help='Grid specification: for k-daily measures set k here', default=default_gridspec['day_step'], type=int)
parser.add_argument('--latoffset', help='Grid specification: latitude offset', default=default_gridspec['lat_offs'], type=float)
parser.add_argument('--latstep', help='Grid specification: latitude step', default=default_gridspec['lat_step'], type=float)
parser.add_argument('--lonstep', help='Grid specification: longitude step', default=default_gridspec['lon_step'], type=float)
parser.add_argument('--lonoffset', help='Grid specification: longitude offset', default=default_gridspec['lon_offs'], type=float)
parser.add_argument('--gridspecfile', help='Load grid specification from a simple text file: <var> = <value>')
parser.add_argument('--timelast', help='Time is the third dimension not the first one', action='store_true')
parser.add_argument('--out', help='Output file', default='detections.json')
args = parser.parse_args()

gridspec = {
    'year_offs': args.yearoffset,
    'lat_offs': args.latoffset,
    'lat_step': args.latstep,
    'lon_step': args.lonstep,
    'lon_offs': args.lonoffset,
    'day_step': args.daystep
}
if not args.gridspecfile is None:
    print ("loading grid specifications from file {} ...".format(args.gridspecfile))
    with open(args.gridspecfile, 'r') as gsf:
        for line in gsf:
            l = line.rstrip().lower()
            variable, value = l.split(' = ')
        if variable=='year_offs':
            gridspec[variable] = int(value)
        else:
            gridspec[variable] = float(value)

if args.eventscsv is None:
    print ("No historic events specified, use {} for the SLP data.".format(default_csv))

params = maxdiv_params_t()
libmaxdiv.maxdiv_init_params(params)
params.min_size[0] = 3
params.min_size[1] = params.min_size[2] = 3
params.max_size[0] = 10
params.kl_mode = enums['MAXDIV_KL_UNBIASED']
#params.proposal_generator = enums['MAXDIV_POINTWISE_PROPOSALS_HOTELLINGST']
params.preproc.embedding.kt = 3
params.preproc.embedding.kx = params.preproc.embedding.ky = 1
params.preproc.normalization = enums['MAXDIV_NORMALIZE_MAX']

if args.eventscsv is None:
    historic_events = []
else:
    historic_events = loadHistoricEvents(args.eventscsv)

print ("loading tensor from {} ...".format(args.matfile))
tensor = loadTensor(args.matfile, args.tensorvar)

# re-order the date if necessary
if args.timelast:
    tensor = np.transpose(tensor, (2, 0, 1, 3, 4))

# tensor dimensions: grid-time grid-lat grid-lon
print ("tensor dimensions: {}".format(tensor.shape))
print ("a detection spanning the whole tensor would be: ")
whole_tensor_span = [0, 0, 0], tensor.shape, 1.0 
printDetection( whole_tensor_span, gridspec )

print ("running maxdiv ...")
start_time = time.time()
detections = maxdiv_exec(tensor, params, 20)
stop_time = time.time()
print ("completed maxdiv in {:.3f} seconds".format(stop_time - start_time))

print ("matching with historic events ...")
readable_detections = []
y_o, d_s = gridspec['year_offs'], gridspec['day_step']
grid_coords = [ gridspec[k] for k in ['lat_offs', 'lat_step', 'lon_offs', 'lon_step' ]]
for detection in detections:
    rd = {}
    range_start, range_end, score = detection
    rd['start'] = str(ind2date(range_start[0], y_o, d_s))
    rd['end'] = str(ind2date(range_start[1], y_o, d_s))
    rd['start_latlon'] = ind2latlon(range_start[1], range_start[2], *grid_coords),
    rd['end_latlon'] = ind2latlon(range_end[1] - 1, range_end[2] - 1, *grid_coords)
    rd['score'] = score
    readable_detections.append(rd)

with open(args.out, 'w') as outf:
    json.dump({'detections_grid': detections, 
       'gridspec': gridspec, 
       'settings': vars(args),
       'detections': readable_detections}, outf, indent=4)
printDetections(detections, gridspec, historic_events)
