from cmath import sqrt
from lumicks import pylake
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import scipy

TIME = "Time"
FORCE = "Force"
DIST = "Distance"

class Trace:
    def __init__(self, time, force, distance):
        self.data = pd.DataFrame()
        self.data[TIME] = time
        self.data[FORCE] = force
        self.data[DIST] = distance

    def extract_time_window(self, time_start, time_end):
        idx_start = np.where(abs(self.data[TIME] - time_start) == min(abs(self.data[TIME] - time_start)))[0][0]
        idx_end = np.where(abs(self.data[TIME] - time_end) == min(abs(self.data[TIME] - time_end)))[0][0]

        sub_data = self.data[idx_start:idx_end]

        return Trace(time=sub_data[TIME],force=sub_data[FORCE],distance=sub_data[DIST])

    def estimate_contour_length(self,T,P):
        a = scipy.constants.k*T/P
        ff = self.data[FORCE]
        zz = self.data[DIST]

        Lc = np.empty(shape=(len(ff)))

        for (i,(f,z)) in enumerate(zip(ff,zz)):
            Lc[i] = ((54*a**3*z**3 - 27*a**2*f*z**3 + 3*sqrt(3)*sqrt(135*a**4*f**2*z**6 - 108*a**3*f**3*z**6 + 144*a**2*f**4*z**6 - 64*a*f**5*z**6) + 72*a*f**2*z**3 - 16*f**3*z**3)**(1/3)/(6*2**(1/3)*f) - (-36*a**2*z**2 + 12*a*f*z**2 - 16*f**2*z**2)/(12*2**(2/3)*f*(54*a**3*z**3 - 27*a**2*f*z**3 + 3*sqrt(3)*sqrt(135*a**4*f**2*z**6 - 108*a**3*f**3*z**6 + 144*a**2*f**4*z**6 - 64*a*f**5*z**6) + 72*a*f**2*z**3 - 16*f**3*z**3)**(1/3)) + (3*a*z + 4*f*z)/(6*f)).real
        
        return Lc
    
    def find_force_event(self):
        x = self.data[TIME]
        vals = self.data[FORCE]

        # try:
        #     res = curve_fit(__getSaw__, x, vals, p0, bounds=(p_lb, p_ub))[0]
        #     g = multi_gauss_1D(x,*res)
        #     temp_peaks.append(res)
        #     scores.append(sum(abs(vals-g)))
        # except:
        #     return
    
    def get_time(self):
        return self.data[TIME]
    
    def get_force(self):
        return self.data[FORCE]
    
    def get_distance(self):
        return self.data[DIST]
    
def __getSaw__(n,x_start,x_peak,x_end,amplitude,baseline_a,baseline_b,baseline_c):
    # Generating baseline
    polynomial = np.polynomial.Polynomial(coef=(baseline_a,baseline_b,baseline_c),domain=(0,n-1),window=(0,n-1))
    saw = polynomial.linspace(n)

    # Adding saw peak
    m_asc = amplitude/(x_peak-x_start)
    for i in range(x_start,x_peak):
        saw[1][i] = saw[1][i] + (i-x_start)*m_asc

    m_dsc = amplitude/(x_end-x_peak)
    for i in range(x_peak,x_end):
        saw[1][i] = saw[1][i] + (x_end-i)*m_dsc
        
    return saw

fpath = "/Users/sc13967/Documents/People/Alex Hughes-Games/20230120-135230 Kymograph 4 H119A trans, F.h5"
h5_file = pylake.File(fpath)

time_ns = h5_file['Force LF']['Trap 2'].timestamps
time_s = (time_ns - time_ns[0])*1E-9
force_N = h5_file['Force LF']['Trap 2'].data*1E-12
dist_m = h5_file['Distance']['Distance 1'].data*1E-6

trace = Trace(time=time_s,force=force_N,distance=dist_m)
