from collections import OrderedDict
from scipy.optimize import curve_fit

import csv
import matplotlib.pyplot as plt
import numpy as np

class MSD:
    def __init__(self,track,spatial_scale=1,spatial_units="pixels",time_scale=1,time_units="frames"):
        # Track for which this is to be calculated
        self.track = track

        # Calibration
        self.spatial_scale = spatial_scale
        self.spatial_units = spatial_units
        self.time_scale = time_scale
        self.time_units = time_units

        # Measurements
        self.msd = None
        self.d_coeff = None
        self.d_coeff_intercept = None
        self.d_coeff_range = None

        # Calcualte MSD
        self.measure_msd()

    def measure_msd(self):
        # Each peak represents a timepoint, so iterating over each peak pair,
        # adding their value to the time difference.  For this, storing a count 
        # per time difference is necessary.
        msd = {}

        for peak_1 in self.track.peaks.values():
            for peak_2 in self.track.peaks.values():
                dt_f = peak_2.t - peak_1.t
                t_1 = peak_1.t*self.time_scale
                t_2 = peak_2.t*self.time_scale
                x_1 = peak_1.b*self.spatial_scale
                x_2 = peak_2.b*self.spatial_scale

                dt = t_2-t_1

                if dt_f <= 0:
                    continue

                if dt_f not in msd:
                    msd[dt_f] = [dt,0,0]

                msd[dt_f][1] = msd[dt_f][1] + (x_2-x_1)*(x_2-x_1)
                msd[dt_f][2] = msd[dt_f][2] + 1

        # Calculating the average (we shouldn't have a dt of 0)
        for dt_f in msd.keys():
            msd[dt_f][1] = msd[dt_f][1]/msd[dt_f][2]

        self.msd = OrderedDict(sorted(msd.items())) # MSD stored as a dict with frame intervals as keys

        return self.msd
    
    def measure_diffusion_coefficient(self, n=10):
        # Fitting straight line to first n points of MSD curve
        x = []
        y = []

        i = 0
        for [dt,val,count] in self.msd.values():
            x.append(dt)
            y.append(val)
            if i > n:
                break
            i = i + 1
            
        def f(x, A, B):
            return A*x + B
        
        popt = curve_fit(f, x, y)

        self.d_coeff = popt[0][0]
        self.d_coeff_intercept = popt[0][1]
        self.d_coeff_range = n

        return (self.d_coeff, self.d_coeff_intercept)
    
    def plot(self, show=True, show_fit_if_available=True):
        fig = plt.figure()
        
        x = []
        y = []

        for [dt,val,count] in self.msd.values():
            x.append(dt)
            y.append(val)

        plt.plot(x,y)
        
        if show_fit_if_available and self.d_coeff is not None:
            xx = np.array(x[0:self.d_coeff_range])        
            yy = xx*self.d_coeff + self.d_coeff_intercept

            plt.plot(xx,yy)

        ax = plt.gca()
        ax.set_xlabel('Time interval ('+self.time_units+')')
        ax.set_ylabel('MSD ('+self.spatial_units+'²)')

        if show:
            plt.show()

        return fig

    def write_to_csv(self, filepath):
        with open(filepath+".csv", 'w', newline='') as file:
            writer = csv.writer(file)

            # Adding header row
            row = ['dt (frames)','dt ('+self.time_units+')','MSD ('+self.spatial_units+'²)','Count']

            if self.d_coeff is not None:
                row.append('Fit ('+self.spatial_units+'²)')

            writer.writerow(row)

            i = 0
            for dt,msd in self.msd.items():
                row = []
                row.append(str(dt))
                row.append(str(msd[0]))
                row.append(str(msd[1]))
                row.append(str(msd[2]))

                if self.d_coeff is not None and i <= self.d_coeff_range:
                    row.append(dt*self.d_coeff + self.d_coeff_intercept)

                writer.writerow(row)
                i = i + 1
