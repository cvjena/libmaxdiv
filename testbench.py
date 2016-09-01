""" Synthetic Test Bench for MaxDiv """

from maxdiv import maxdiv_util
import sys
import numpy as np
import matplotlib.pylab as plt
try:
    import cPickle as pickle
except ImportError:
    # cPickle has been "hidden" in Python 3 and will be imported automatically by
    # pickle if available.
    import pickle


def sample_gp(X, meany, sigma, n=1, noise=0.001):
    """ sample a function from a Gaussian process with Gaussian kernel """
    K = maxdiv_util.calc_gaussian_kernel(X, sigma) + noise * np.eye(X.shape[1])
    return np.random.multivariate_normal(meany, K, n)

def sample_gp_nonstat(X, meany, sigmas, n=1, noise=0.001):
    """ sample a function from a non-stationary Gaussian process """
    # http://papers.nips.cc/paper/2350-nonstationary-covariance-functions-for-gaussian-process-regression.pdf
    K = maxdiv_util.calc_nonstationary_gaussian_kernel(X, sigmas) + noise * np.eye(X.shape[1])
    return np.random.multivariate_normal(meany, K, n)


def sample_interval(n, minlen, maxlen):
    """ sample the bounds of an interval """
    defect_start = int(np.random.randint(0,n-minlen))
    defect_end = int(np.random.randint(defect_start+minlen,min(defect_start+maxlen,n)))
    defect = np.zeros(n, dtype=bool)
    defect[defect_start:defect_end] = True
    return defect, defect_start, defect_end


def sample_multiple_intervals(n, minlen, maxlen, max_intervals, spacing = 50):
    """ sample the bounds of multiple non-overlapping intervals """
    
    defect = np.zeros(n, dtype=bool)
    regions = []
    
    while (len(regions) < max_intervals) and (maxlen >= minlen):
    
        defect_start = int(np.random.randint(0, n - minlen))
        defect_len = int(np.random.randint(minlen, maxlen + 1))
        defect_end = defect_start + defect_len
        
        if not np.any(defect[max(0, defect_start - spacing):min(n, defect_end + spacing)]):
            defect[defect_start:defect_end] = True
            regions.append((defect_start, defect_end))
            
            max_non_defect_len = 0
            last_defect_end = 0
            for i in range(1, n):
                if defect[i-1] and (not defect[i]):
                    last_defect_end = i
                elif (not defect[i-1]) and defect[i]:
                    non_defect_len = i - last_defect_end - spacing
                    if last_defect_end > 0:
                        non_defect_len -= spacing
                    max_non_defect_len = max(max_non_defect_len, non_defect_len)
            if not defect[n-1]:
                max_non_defect_len = max(max_non_defect_len, n - last_defect_end - spacing)
            maxlen = min(maxlen, max_non_defect_len)
        
    return defect, regions


def rand_sign():
    return (np.random.randint(0, 2) * 2) - 1


def attributes_from_states(gps, numattr, numcorr):
    numstates, n = gps.shape
    states = np.arange(numstates)
    proj = np.zeros((numattr, numstates))
    for i in range(numattr):
        np.random.shuffle(states)
        proj[i, states[:numcorr]] = np.random.randn(numcorr)
    return proj.dot(gps)
    


if __name__ == '__main__':
    
    if (len(sys.argv) < 2) or (sys.argv[1] not in ('large', 'small', 'hd', 'seasonal', 'nominal')):
        print('Usage: {} <type = large|small|hd|seasonal|nominal>'.format(sys.argv[0]))
        exit()
    type = sys.argv[1]

    # ensure reproducable results
    np.random.seed(0)

    if type in ('large', 'small'):
    
        X = np.arange(0,1, 0.004 if type == 'small' else 0.002)
        X = np.reshape(X, [1, len(X)])
        n = X.shape[1]
        numf = 20 if type == 'small' else 100
        sigma = 0.02 if type == 'small' else 0.01

        print ("Generating time series of length {}".format(n))
        defect_maxlen = int(0.2 * n)
        defect_minlen = int(0.05 * n)
        print ("Minimal and maximal length of one extreme {} - {}".format(defect_minlen, defect_maxlen))

        y = {}
        f = {}

        # simple mean shift
        zeroy = np.zeros(X.shape[1])
        y['meanshift'] = []
        gps = sample_gp(X, zeroy, sigma, numf)
        f['meanshift'] = np.reshape(gps, [gps.shape[0], 1, gps.shape[1]])
        for i in range(numf):
            defect, _, _ = sample_interval(n, defect_minlen, defect_maxlen)
            y['meanshift'].append(defect)
            f['meanshift'][i,0,defect] += rand_sign() * (np.random.rand()*1.0 + 3.0)

        y['meanshift_hard'] = []
        gps = sample_gp(X, zeroy, sigma, numf)
        f['meanshift_hard'] = np.reshape(gps, [gps.shape[0], 1, gps.shape[1]])
        for i in range(numf):
            defect, _, _ = sample_interval(n, defect_minlen, defect_maxlen)
            y['meanshift_hard'].append(defect)
            f['meanshift_hard'][i,0,defect] += rand_sign() * (np.random.rand()*0.5 + 0.5)

        y['meanshift_multvar'] = []
        gps = sample_gp(X, zeroy, sigma, numf*5)
        f['meanshift_multvar'] = np.reshape(gps, [numf, 5, gps.shape[1]])
        for i in range(numf):
            defect, _, _ = sample_interval(n, defect_minlen, defect_maxlen)
            y['meanshift_multvar'].append(defect)
            f['meanshift_multvar'][i,0,defect] += rand_sign() * (np.random.rand()*1.0 + 3.0)

        y['amplitude_change'] = []
        f['amplitude_change'] = np.zeros([numf, 1, n])
        for i in range(numf):
            defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
            y['amplitude_change'].append(defect)
            func = sample_gp(X, zeroy, sigma, 1)
            sigmaw = (b-a)/4.0
            mu = (a+b)/2.0
            gauss = np.array([ np.exp(-(xp-mu)**2/(2*sigmaw*sigmaw)) for xp in range(n) ])
            gauss[gauss>0.2] = 0.2
            func = func * (2.0*gauss/np.max(gauss)+1)
            f['amplitude_change'][i, 0] = func
         
        y['amplitude_change_multvar'] = []
        f['amplitude_change_multvar'] = np.zeros([numf, 5, n])
        for i in range(numf):
            defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
            y['amplitude_change_multvar'].append(defect)
            func = sample_gp(X, zeroy, sigma, 5)
            sigmaw = (b-a)/4.0
            mu = (a+b)/2.0
            gauss = np.array([ np.exp(-(xp-mu)**2/(2*sigmaw*sigmaw)) for xp in range(n) ])
            gauss[gauss>0.2] = 0.2
            func[0,:] = func[0,:] * (2.0*gauss/np.max(gauss)+1)
            f['amplitude_change_multvar'][i, :] = func

        y['frequency_change'] = []
        f['frequency_change'] = np.zeros([numf, 1, n])
        for i in range(numf):
            defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
            y['frequency_change'].append(defect)
            func = sample_gp_nonstat(X, zeroy, (1-defect)*0.01+0.0001, 1)
            f['frequency_change'][i, 0] = func

        y['frequency_change_multvar'] = []
        f['frequency_change_multvar'] = np.zeros([numf, 5, n])
        for i in range(numf):
            defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
            y['frequency_change_multvar'].append(defect)
            func_defect = sample_gp_nonstat(X, zeroy, (1-defect)*0.01+0.0001, 1)
            func_ok = sample_gp(X, zeroy, sigma, 4)
            f['frequency_change_multvar'][i] = np.vstack([func_defect, func_ok])
        
        
        # Anomalies generated by a "different mechanism"
        fade_len = 10
        fading = np.linspace(0, 1, fade_len, endpoint = False)
        
        gps_nominal = sample_gp(X, zeroy, sigma, numf, 0).reshape([numf, 1, X.shape[1]])
        gps_anomalous = sample_gp(X, zeroy, sigma, numf, 0).reshape([numf, 1, X.shape[1]])
        interpolation_mask = np.zeros((numf, 1, X.shape[1]))
        y['mixed'] = []
        for i in range(numf):
            while True:
                defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
                if (a >= 50) and (b <= n - 50):
                    break
            interpolation_mask[i, :, (a - fade_len/2):(b + fade_len/2)] = np.concatenate([fading, np.ones(b - a - fade_len), fading[::-1]])
            y['mixed'].append(defect)
        f['mixed'] = gps_nominal + (gps_anomalous - gps_nominal) * interpolation_mask + 0.1 * np.random.randn(*gps_nominal.shape)
        
        gps_nominal = sample_gp(X, zeroy, sigma, numf * 5, 0).reshape([numf, 5, X.shape[1]])
        gps_anomalous = sample_gp(X, zeroy, sigma, numf * 5, 0).reshape([numf, 5, X.shape[1]])
        interpolation_mask = np.zeros((numf, 5, X.shape[1]))
        y['mixed_multvar'] = []
        for i in range(numf):
            while True:
                defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
                if (a >= 50) and (b <= n - 50):
                    break
            interpolation_mask[i, :, (a - fade_len/2):(b + fade_len/2)] = np.concatenate([fading, np.ones(b - a - fade_len), fading[::-1]])
            y['mixed_multvar'].append(defect)
        f['mixed_multvar'] = gps_nominal + (gps_anomalous - gps_nominal) * interpolation_mask + 0.1 * np.random.randn(*gps_nominal.shape)


        # Multiple extremes
        X = np.arange(0,1,0.001)
        X = np.reshape(X, [1, len(X)])
        n = X.shape[1]
        zeroy = np.zeros(X.shape[1])
        maxint = 5
        sigma = 0.01

        print ("Generating time series of length {} with multiple extremes".format(n))
        defect_maxlen = int(0.05*n)
        defect_minlen = int(0.02*n)
        print ("Minimal and maximal length of one extreme {} - {}".format(defect_minlen, defect_maxlen))

        y['meanshift5'] = []
        gps = sample_gp(X, zeroy, sigma, numf)
        f['meanshift5'] = np.reshape(gps, [gps.shape[0], 1, gps.shape[1]])
        for i in range(numf):
            defect, regions = sample_multiple_intervals(n, defect_minlen, defect_maxlen, maxint)
            y['meanshift5'].append(defect)
            for a, b in regions:
                f['meanshift5'][i,0,a:b] += rand_sign() * (np.random.rand()*1.0 + 3.0)

        y['meanshift5_hard'] = []
        gps = sample_gp(X, zeroy, sigma, numf)
        f['meanshift5_hard'] = np.reshape(gps, [gps.shape[0], 1, gps.shape[1]])
        for i in range(numf):
            defect, regions = sample_multiple_intervals(n, defect_minlen, defect_maxlen, maxint)
            y['meanshift5_hard'].append(defect)
            for a, b in regions:
                f['meanshift5_hard'][i,0,a:b] += rand_sign() * (np.random.rand()*0.5 + 0.5)
        
        with open('testcube_small.pickle' if type == 'small' else 'testcube.pickle', 'wb') as fout:
            pickle.dump({'f': f, 'y': y}, fout)
    
    elif type == 'hd':
        
        X = np.arange(0,1, 0.001)
        X = np.reshape(X, [1, len(X)])
        n = X.shape[1]
        zeroy = np.zeros(n)
        numf = 100
        numattr = 100
        numstates = 10
        numcorr = 3
        sigma = 0.01

        print ("Generating time series of length {} with {} attributes reflecting {} hidden states".format(n, numattr, numstates))
        defect_maxlen = int(0.1 * n)
        defect_minlen = int(0.02 * n)
        print ("Minimal and maximal length of one extreme {} - {}".format(defect_minlen, defect_maxlen))

        y = {}
        f = {}

        y['meanshift_hd'] = []
        f['meanshift_hd'] = np.ndarray((numf, numattr, n))
        gps = np.reshape(sample_gp(X, zeroy, sigma, numf*numstates), [numf, numstates, n])
        for i in range(numf):
            defect, _, _ = sample_interval(n, defect_minlen, defect_maxlen)
            y['meanshift_hd'].append(defect)
            gps[i,0,defect] += rand_sign() * (np.random.rand()*1.0 + 3.0)
            f['meanshift_hd'][i] = attributes_from_states(gps[i], numattr, numcorr)
         
        y['amplitude_change_hd'] = []
        f['amplitude_change_hd'] = np.ndarray((numf, numattr, n))
        for i in range(numf):
            defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
            y['amplitude_change_hd'].append(defect)
            func = sample_gp(X, zeroy, sigma, numstates)
            sigmaw = (b-a)/4.0
            mu = (a+b)/2.0
            gauss = np.array([ np.exp(-(xp-mu)**2/(2*sigmaw*sigmaw)) for xp in range(n) ])
            gauss[gauss>0.2] = 0.2
            func[0,:] = func[0,:] * (2.0*gauss/np.max(gauss)+1)
            f['amplitude_change_hd'][i] = attributes_from_states(func, numattr, numcorr)

        y['frequency_change_hd'] = []
        f['frequency_change_hd'] = np.ndarray([numf, numattr, n])
        for i in range(numf):
            defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
            y['frequency_change_hd'].append(defect)
            func_defect = sample_gp_nonstat(X, zeroy, (1-defect)*0.01+0.0001, 1)
            func_ok = sample_gp(X, zeroy, sigma, numstates - 1)
            f['frequency_change_hd'][i] = attributes_from_states(np.vstack([func_defect, func_ok]), numattr, numcorr)
        
        fade_len = 10
        fading = np.linspace(0, 1, fade_len, endpoint = False)
        gps_nominal = sample_gp(X, zeroy, sigma, numf * numstates, 0).reshape([numf, numstates, n])
        gps_anomalous = sample_gp(X, zeroy, sigma, numf * numstates, 0).reshape([numf, numstates, n])
        y['mixed_hd'] = []
        f['mixed_hd'] = np.ndarray([numf, numattr, n])
        for i in range(numf):
            while True:
                defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
                if (a >= 50) and (b <= n - 50):
                    break
            interpolation_mask = np.zeros((numstates, X.shape[1]))
            interpolation_mask[:, (a - fade_len/2):(b + fade_len/2)] = np.concatenate([fading, np.ones(b - a - fade_len), fading[::-1]])
            y['mixed_hd'].append(defect)
            f['mixed_hd'][i] = attributes_from_states(gps_nominal[i] + (gps_anomalous[i] - gps_nominal[i]) * interpolation_mask + 0.1 * np.random.randn(numstates, n), numattr, numcorr)
        
        with open('testcube_hd.pickle', 'wb') as fout:
            pickle.dump({'f': f, 'y': y}, fout)
    
    elif type == 'seasonal':
    
        X = np.arange(0,1, 0.002)
        X = np.reshape(X, [1, len(X)])
        n = X.shape[1]
        numf = 100
        sigma = 0.1

        print ("Generating time series of length {}".format(n))
        defect_maxlen = int(0.2 * n)
        defect_minlen = int(0.05 * n)
        print ("Minimal and maximal length of one extreme {} - {}".format(defect_minlen, defect_maxlen))

        y = {}
        f = {}
        
        # Diurnal seasonality with amplitude change anomalies
        y['diurnal'] = []
        f['diurnal'] = np.zeros([numf, 1, n])
        for i in range(numf):
            season_amp = np.random.uniform(0.5, 3.0) * np.ones(n)
            phase_shift = np.random.rand()
            seasonality = np.sin(np.pi * (np.arange(0, n) / 24.0 + phase_shift)) ** 2
            
            defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
            y['diurnal'].append(defect)
            sigmaw = (b-a)/4.0
            mu = (a+b)/2.0
            gauss = np.array([ np.exp(-(xp-mu)**2/(2*sigmaw*sigmaw)) for xp in range(n) ])
            gauss[gauss>0.2] = 0.2
            amp_change = np.random.uniform(2.0, 4.0)
            seasonality *= season_amp + amp_change * gauss/np.max(gauss)
            
            f['diurnal'][i, 0] = sample_gp(X, seasonality, sigma, 1, 0.01)
        
        # Multivariate Diurnal seasonality with amplitude change anomalies
        y['diurnal_multvar'] = []
        f['diurnal_multvar'] = np.zeros([numf, 5, n])
        for i in range(numf):
            for j in range(5):
                season_amp = np.random.uniform(0.5, 3.0) * np.ones(n)
                phase_shift = np.random.rand()
                seasonality = np.sin(np.pi * (np.arange(0, n) / 24.0 + phase_shift)) ** 2
                
                if j == 0:
                    defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
                    y['diurnal_multvar'].append(defect)
                    sigmaw = (b-a)/4.0
                    mu = (a+b)/2.0
                    gauss = np.array([ np.exp(-(xp-mu)**2/(2*sigmaw*sigmaw)) for xp in range(n) ])
                    gauss[gauss>0.2] = 0.2
                    amp_change = np.random.uniform(2.0, 4.0)
                    season_amp += amp_change * gauss / np.max(gauss)
                
                f['diurnal_multvar'][i, j] = sample_gp(X, seasonality * season_amp, sigma, 1, 0.01)
        
        # Diurnal and weekly seasonality with amplitude change anomalies
        X = np.arange(0,1, 0.001)
        X = np.reshape(X, [1, len(X)])
        n = X.shape[1]
        y['diurnal_weekly'] = []
        f['diurnal_weekly'] = np.zeros([numf, 1, n])
        for i in range(numf):
            season_amp = np.random.uniform(0.5, 3.0) * np.ones(n)
            phase_shift = np.random.rand()
            diurnal_seasonality = np.sin(np.pi * (np.arange(0, n) / 24.0 + phase_shift)) ** 2
            phase_shift = 2.0 * np.random.rand()
            weekly_seasonality = np.random.uniform(0.5, 3.0) * np.sin(np.pi * (np.arange(0, n) / (7.0 * 24.0) + phase_shift))
            
            defect, a, b = sample_interval(n, defect_minlen, defect_maxlen)
            y['diurnal_weekly'].append(defect)
            sigmaw = (b-a)/4.0
            mu = (a+b)/2.0
            gauss = np.array([ np.exp(-(xp-mu)**2/(2*sigmaw*sigmaw)) for xp in range(n) ])
            gauss[gauss>0.2] = 0.2
            amp_change = np.random.uniform(2.0, 4.0)
            diurnal_seasonality += amp_change * gauss / np.max(gauss)
            
            f['diurnal_weekly'][i, 0] = sample_gp(X, diurnal_seasonality + weekly_seasonality, sigma, 1, 0.01)
        
        with open('testcube_seasonal.pickle', 'wb') as fout:
            pickle.dump({'f': f, 'y': y}, fout)
    
    elif type == 'nominal':
    
        # Some completely normal time series with more noise
        X = np.arange(0,1,0.001)
        X = np.reshape(X, [1, len(X)])
        n = X.shape[1]
        numf = 100
        zeroy = np.zeros(X.shape[1])
        sigma = 0.1
        
        normal_funcs = {}
        gps = sample_gp(X, zeroy, sigma, numf)
        normal_funcs['normal_gp'] = np.reshape(gps, [gps.shape[0], 1, gps.shape[1]])
        gps = sample_gp(X, zeroy, sigma, 5 * numf)
        normal_funcs['normal_gp_multvar'] = np.reshape(gps, [numf, 5, gps.shape[1]])

        with open('testcube_normal.pickle', 'wb') as fout:
            pickle.dump(normal_funcs, fout)