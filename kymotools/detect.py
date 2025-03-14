from scipy.optimize import curve_fit
from scipy.optimize import linear_sum_assignment
from kymotools.kymo import Peak
from kymotools.kymo import Track
from tqdm import tqdm

import math
import numpy as np

NO_LINK = 100000000

class Detector():
    def __init__(self,half_t_w = 2, peak_det_thresh = 3.5, max_dist = 6, max_frame_gap = 10, min_track_length = 50, min_track_density=0, track_heritage_weight=100, n_max = 8, a_lb = 0, a_ub = 10000, c_lb = 1.2, c_ub = 3, c_def = 2, starting_window=None):
        self._half_t_w = half_t_w
        self._peak_det_thresh = peak_det_thresh
        self._max_dist = max_dist
        self._max_frame_gap = max_frame_gap
        self._min_track_length = min_track_length
        self._min_track_density = min_track_density
        self._track_heritage_weight = track_heritage_weight
        self._n_max = n_max
        self._a_lb = a_lb
        self._a_ub = a_ub
        self._c_lb = c_lb
        self._c_ub = c_ub
        self._c_def = c_def
        self._starting_window = starting_window
        
    def detect(self,image):
        peaks = {}
        tracks = {}

        for frame in tqdm(range(image.shape[1])):
            vals = get_raw_profile(image,frame,self._half_t_w)
            frame_peaks = self.fit_peaks(image,vals,frame)

            # Determining the maximum peak ID and adding 1
            max_peak_id = 0
            if len(peaks) != 0:
                max_peak_id = max(peaks.keys())
            peak_id = max_peak_id + 1
                
            # Adding new peaks into main collection of peaks
            for frame_peak in frame_peaks.values():
                frame_peak.ID = peak_id
                peaks[peak_id] = frame_peak
                peak_id = peak_id + 1

            # Tracking
            (costs,peak_ids,track_ids) = _calculate_cost_matrix(peaks,tracks,frame, self._max_dist, self._max_frame_gap, self._track_heritage_weight)
            peak_idx,track_idx = linear_sum_assignment(costs)
            _apply_tracks(peaks,tracks,costs,peak_ids,track_ids,peak_idx,track_idx)
            _assign_unlinked_tracks(peaks,tracks)

        _track_length_filter(tracks,peaks,self._min_track_length)
        _track_density_filter(tracks,peaks,self._min_track_density)
        
        if self._starting_window is not None:
            _ignore_missing_at_start(tracks,peaks,self._starting_window)

        return tracks

    def fit_peaks(self,vals,frame=0):
        frame_peaks = {}
        
        x = np.arange(len(vals))
            
        temp_peaks = []
        scores = []
        for n_peaks in range(self._n_max):        
            # Creating initial parameters for fitting
            fits = self._initialise_guesses(x,vals,n_peaks)

            if fits is None:
                break

            (p0,p_lb,p_ub) = fits            

            try:
                popt = curve_fit(multi_gauss_1D, x, vals, p0, bounds=(p_lb, p_ub))[0]
                g = multi_gauss_1D(x,*popt)
                temp_peaks.append(popt)
                scores.append(sum(abs(vals-g)))
                
            except:
                continue

        if len(scores) == 0:
            return frame_peaks
            
        idx = scores.index(min(scores))
        best_peaks = temp_peaks[idx]
        
        for i in range(0, len(best_peaks),3):
            max_peak_id = 0
            if len(frame_peaks) != 0:
                max_peak_id = max(frame_peaks.keys())
            peak_id = max_peak_id + 1
            
            peak = Peak(peak_id, frame, best_peaks[i], best_peaks[i+1], best_peaks[i+2])
            frame_peaks[peak_id] = peak   

        fit_vals = multi_gauss_1D(x, *popt)
        rmse = np.sqrt(np.mean((vals - fit_vals) ** 2))
            
        return frame_peaks, rmse

    def get_parameters(self): 
        params = {}
        params['half_t_w'] = self._half_t_w
        params['peak_det_thresh'] = self._peak_det_thresh
        params['max_dist'] = self._max_dist
        params['max_frame_gap'] = self._max_frame_gap
        params['min_track_length'] = self._min_track_length
        params['min_track_density'] = self._min_track_density
        params['track_heritage_weight'] = self._track_heritage_weight
        params['n_max'] = self._n_max
        params['a_lb'] = self._a_lb
        params['a_ub'] = self._a_ub
        params['c_lb'] = self._c_lb
        params['c_ub'] = self._c_ub
        params['c_def'] = self._c_def
        params['starting_window'] = self._starting_window

        return params

    def _initialise_guesses(self,x,vals,n_peaks):
        p0 = []
        p_lb = []
        p_ub = []
        vals_temp = vals

        for peak in range(n_peaks+1):
            if np.max(vals_temp) < self._peak_det_thresh:
                return None
            
            b = vals_temp.argmax()
            try:
                b_est = ((b-1)*vals_temp[b-1]+b*vals_temp[b]+(b+1)*vals_temp[b+1])/(vals_temp[b-1]+vals_temp[b]+vals_temp[b+1])
            except:
                b_est = b
            
            max_val = np.max(vals_temp)
            g = gauss_1D(x, np.max(vals_temp), b_est, self._c_def)
            vals_temp = vals_temp - g

            p0.append(max_val)
            p0.append(b_est)
            p0.append(self._c_def)

            p_lb.append(self._a_lb)
            p_lb.append(0)
            p_lb.append(self._c_lb)

            p_ub.append(self._a_ub)
            p_ub.append(len(x))
            p_ub.append(self._c_ub)

        return (p0,p_lb,p_ub)
       

def get_raw_profile(image, frame, half_t_w):
    if half_t_w < 1:
        vals = image[:,frame]
    else:
        widevals = image[:, max(0,frame-half_t_w):min(frame+half_t_w,image.shape[1]-1)]
        vals = np.median(widevals, 1)

    return vals

# Gaussian functions
def gauss_1D(x, a, b, c):
    return a*np.exp(-((x-b)*(x-b))/(2*c*c))


def multi_gauss_1D(x, *p):
    g = np.zeros(len(x))
    for i in range(0, len(p), 3):
        g = g + gauss_1D(x, p[i], p[i+1], p[i+2])

    # Checking the peaks aren't too close together
    for i in range(0,len(p),3):
        for j in range(0,len(p),3):
            if i == j:
                continue
             
            if abs(p[i+1]-p[j+1]) < 2:   
                g = g + np.inf
                
    return g

def _assign_unlinked_tracks(peaks,tracks):
    for peak in peaks.values():
        if peak.track is None:
            max_track_id = 0
            if len(tracks) != 0:
                max_track_id = max(tracks.keys())
            
            track = Track(max_track_id+1)
            track.add_peak(peak)
            tracks[max_track_id + 1] = track

def _calculate_cost_matrix(peaks, tracks, frame, max_dist, max_frame_gap, track_heritage_weight):
    peak_ids = []
    track_ids = []

    # Counting the number of unassigned peaks and tracks within the linking window
    n_peaks = 0
    for peak_id in peaks.keys():
        peak = peaks[peak_id]
        if peak.track is None:
            peak_ids.append(peak_id)
            n_peaks = n_peaks + 1

    n_tracks = 0
    for track_id in tracks.keys():
        track = tracks[track_id]
        if max(track.peaks.keys()) >= frame-max_frame_gap:            
            track_ids.append(track_id)
            n_tracks = n_tracks + 1

    # Initialising cost matrix with same size as n_peaks and n_tracks
    costs = np.zeros((n_peaks,n_tracks))

    # Iterating back over peaks and tracks, calculating costs
    n_peaks = 0    
    for peak in peaks.values():
        n_tracks = 0

        if peak.track is not None:
            continue

        for track in tracks.values():
            if max(track.peaks.keys()) < frame-max_frame_gap:
                continue

            # Getting most recent peak in track and calculating cost
            prev_peak = track.peaks[max(track.peaks.keys())]

            dist = abs(prev_peak.b-peak.b)
            heritage = track_heritage_weight*(math.exp((frame-len(track.peaks))/frame)-1)
            # heritage = 100*max(0,max_frame_gap-len(track.peaks))/min(frame,max_frame_gap)
            height_diff = abs(prev_peak.a-peak.a)
            height_diff = 0
            if dist <= max_dist:
                costs[n_peaks, n_tracks] = dist + heritage + height_diff
            else:
                costs[n_peaks, n_tracks] = NO_LINK # Bug in linear_sum_assignment means np.inf doesn't work

            n_tracks = n_tracks + 1
        
        n_peaks = n_peaks + 1
    
    return (costs,peak_ids,track_ids)

def _apply_tracks(peaks, tracks, costs, peak_ids, track_ids, peak_idx, track_idx):
    for i in range(len(peak_idx)):
        peak_id = peak_ids[peak_idx[i]]
        peak = peaks[peak_id]

        track_id = track_ids[track_idx[i]]
        track = tracks[track_id]

        if not costs[peak_idx[i],track_idx[i]] == NO_LINK:
            track.add_peak(peak)

def _track_length_filter(tracks, peaks, min_length):
    to_remove_ids = [id for id, track in tracks.items() if len(track.peaks) < min_length]
    for to_remove_id in to_remove_ids:
        track = tracks[to_remove_id]
        # Removing peaks assigned to this track
        for peak in track.peaks.values():
            del peaks[peak.ID]

        # Removing this track
        del tracks[to_remove_id]    

def _track_density_filter(tracks, peaks, min_density):
    to_remove_ids = [id for id, track in tracks.items() if (len(track.peaks)/(max(track.peaks.keys())-min(track.peaks.keys())+1)) < min_density]
    for to_remove_id in to_remove_ids:
        track = tracks[to_remove_id]
        # Removing peaks assigned to this track
        for peak in track.peaks.values():
            del peaks[peak.ID]

        # Removing this track
        del tracks[to_remove_id]      

def _ignore_missing_at_start(tracks, peaks, starting_window):
    to_remove_ids = []
    for track in tracks.values():
        min_t = 100000000

        for peak in track.peaks.values():
            min_t = min(peak.t,min_t)
        
        if min_t > starting_window:
            to_remove_ids.append(track.ID)

    for to_remove_id in to_remove_ids:
        track = tracks[to_remove_id]
        # Removing peaks assigned to this track
        for peak in track.peaks.values():
            del peaks[peak.ID]

        # Removing this track
        del tracks[to_remove_id]    