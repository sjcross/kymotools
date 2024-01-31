import numpy as np

class Peak:
    def __init__(self, ID, t, a, b, c):
        self.ID = ID    # ID number
        self.t = t      # Timepoint
        self.a = a      # Amplitude
        self.b = b      # X-position
        self.c = c      # Sigma
        self.track = None # Assigned track

class Track:
    def __init__(self, ID):
        self.ID = ID
        self.peaks = {}
        self.intensity = {}
        self.filtered = {}
        self.step_trace = {}
        self.steps = {}
        
    def add_peak(self, peak):
        self.peaks[peak.t] = peak
        peak.track = self

    def measure_intensity(self, image, half_x_w=0, end_pad=100):
        for t in self.peaks.keys():
            peak = self.peaks.get(t)
            x = round(peak.b)
            self.intensity[t] = peak.a

        # Adding measurement points to the end
        t_start = max(max(self.peaks.keys())+1, 0)
        t_end = min(max(self.peaks.keys())+end_pad, image.shape[1])
        for t in range(t_start,t_end):
            self.intensity[t] = image[x-half_x_w:x+half_x_w,t].mean()
            
    def apply_temporal_filter(self,half_t_w=1):
        filtered = {}
        for t in self.intensity.keys():
            diff = abs(np.array(list(self.intensity.keys()))-t)
            filtered[t] = np.median(np.array(list(self.intensity.values()))[np.where(diff<=half_t_w)])

        self.intensity = filtered

    def get_timepoints(self):
        return [peak.t for peak in self.peaks.values()]
    
    def get_amplitudes(self):
        return [peak.a for peak in self.peaks.values()]

    def get_positions(self):
        return [peak.b for peak in self.peaks.values()]
    
    def get_sigmas(self):
        return [peak.c for peak in self.peaks.values()]
    
    def calculate_stationary_probability(self, kymo_size, link_dist=3):
        counts = np.zeros(kymo_size)

        for curr_timepoint, curr_peak in sorted(list(self.peaks.items()), reverse=True):
            curr_position = curr_peak.b

            # Comparing to all other peaks in this track
            for prev_timepoint, prev_peak in sorted(list(self.peaks.items()), reverse=True):
                prev_position = prev_peak.b

                # Only process peaks from earlir timepoints
                if prev_timepoint >= curr_timepoint:
                    continue

                timepoint_separation = curr_timepoint - prev_timepoint
                position_separation = abs(curr_position-prev_position)

                if position_separation < link_dist:
                    counts[round(curr_position),timepoint_separation] = counts[round(curr_position),timepoint_separation] + 1
                else:
                    # Once the peak has moved out of the linking range, stop recording history
                    break

        for f in range(kymo_size[1]):
            for x in range(kymo_size[0]):
                if counts[x,f] > 0:
                    counts[x,f] = counts[x,f] / (max(self.peaks.keys())-min(self.peaks.keys())-f-1)
                    # counts[x,f] = counts[x,f] / (image.shape[1]-f-1)

        return counts
    