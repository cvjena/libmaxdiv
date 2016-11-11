""" Runs the MDI algorithm on the Sea Level Pressure dataset. """

import os.path
from utils import *
from maxdiv.libmaxdiv_wrapper import *
import argparse
import json

default_csv = '../coastdat/historic_storms.csv'
default_gridspec = getSLPGridSpec()

parser = argparse.ArgumentParser()
parser.add_argument('--matfile', help='MAT file containing the tensor', default='SLP_ATL.mat')
parser.add_argument('--tensorvar', help='Variable in the MAT containing the tensor', default='pres')
parser.add_argument('--eventscsv', help='CSV file containing historic events', default=default_csv)
parser.add_argument('--yearoffset', help='Grid specification: year offset', default=default_gridspec['year_offs'], type=int)
parser.add_argument('--latoffset', help='Grid specification: latitude offset', default=default_gridspec['lat_offs'], type=float)
parser.add_argument('--latstep', help='Grid specification: latitude step', default=default_gridspec['lat_step'], type=float)
parser.add_argument('--lonstep', help='Grid specification: longitude step', default=default_gridspec['lon_step'], type=float)
parser.add_argument('--lonoffset', help='Grid specification: longitude offset', default=default_gridspec['lon_offs'], type=float)
parser.add_argument('--gridspecfile', help='Load grid specification from a simple text file: <var> = <value>')
parser.add_argument('--out', help='Output file', default='detections.json')
args = parser.parse_args()

if args.gridspecfile is None:
    gridspec = {
            'year_offs': args.yearoffset,
            'lat_offs': args.latoffset,
            'lat_step': args.latstep,
            'lon_step': args.lonstep,
            'lon_offs': args.lonoffset,
        }
else:
    print ("loading grid specifications from file {} ...".format(args.gridspecfile))
    gridspec = {}
    with open(args.gridspecfile, 'r') as gsf:
        for line in gsf:
            l = line.rstrip().lower()
            variable, value = l.split(' = ')
	    if variable=='year_offs':
	    	gridspec[variable] = int(value)
	    else:
	    	gridspec[variable] = float(value)
		

    # check whether we have all necessary variables
    # also set in the default
    for k in default_gridspec:
        if not k in gridspec:
            sys.exit('Variable {} not specified in {}'.format(k, args.gridspecfile))

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
print ("running maxdiv ...")
detections = maxdiv_exec(tensor, params, 20)
print ("matching with historic events ...")

readable_detections = []
y_o = gridspec['year_offs']
grid_coords = [ gridspec[k] for k in ['lat_offs', 'lat_step', 'lon_offs', 'lon_step' ]]
for detection in detections:
    rd = {}
    range_start, range_end, score = detection
    rd['start'] = str(ind2date(range_start[0], y_o))
    rd['end'] = str(ind2date(range_start[1], y_o))
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
