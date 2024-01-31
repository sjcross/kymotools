from cmath import sqrt
from lumicks import pylake
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

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

        return Trace(time=sub_data[TIME].values,force=sub_data[FORCE].values,distance=sub_data[DIST].values)

    def estimate_contour_length(self,T,P):
        a = scipy.constants.k*T/P
        ff = self.data[FORCE]
        zz = self.data[DIST]

        Lc = np.empty(shape=(len(ff)))

        for (i,(f,z)) in enumerate(zip(ff,zz)):
            Lc[i] = ((54*a**3*z**3 - 27*a**2*f*z**3 + 3*sqrt(3)*sqrt(135*a**4*f**2*z**6 - 108*a**3*f**3*z**6 + 144*a**2*f**4*z**6 - 64*a*f**5*z**6) + 72*a*f**2*z**3 - 16*f**3*z**3)**(1/3)/(6*2**(1/3)*f) - (-36*a**2*z**2 + 12*a*f*z**2 - 16*f**2*z**2)/(12*2**(2/3)*f*(54*a**3*z**3 - 27*a**2*f*z**3 + 3*sqrt(3)*sqrt(135*a**4*f**2*z**6 - 108*a**3*f**3*z**6 + 144*a**2*f**4*z**6 - 64*a*f**5*z**6) + 72*a*f**2*z**3 - 16*f**3*z**3)**(1/3)) + (3*a*z + 4*f*z)/(6*f)).real
        
        return Lc
    
    def find_force_event(self):
        x = self.data[TIME].values
        vals = self.data[FORCE].values*1E12
        
        try:
            x_max = np.where(vals==max(vals))[0][0]
            p0 = (0.001,len(x)*0.25,x_max,min(vals))
            
            return curve_fit(__get_saw__, x, vals,p0)[0]
            
        except Exception as e:
            print(e)
            return None
    
    def get_time(self):
        return self.data[TIME]
    
    def get_force(self):
        return self.data[FORCE]
    
    def get_distance(self):
        return self.data[DIST]

def __get_saw__(x,a,x_min,x_max,y_min):
    n = len(x)
    curve = np.zeros(n)

    for i in range(0,n):
        if i < x_min:
            curve[i] = y_min
        elif i >= x_min and i < x_max:
            curve[i] = a*(i-x_min)*(i-x_min)+y_min
        else:
            curve[i] = y_min

    return curve

# fpath = "/Users/sc13967/Documents/People/Alex Hughes-Games/20230120-135230 Kymograph 4 H119A trans, F.h5"
# h5_file = pylake.File(fpath)

# time_ns = h5_file['Force LF']['Trap 2'].timestamps
# time_s = (time_ns - time_ns[0])*1E-9
# force_pN = h5_file['Force LF']['Trap 2'].data
# dist_m = h5_file['Distance']['Distance 1'].data*1E-6

# trace = Trace(time=time_s,force=force_pN,distance=dist_m)
# trace_window = trace.extract_time_window(time_start=40,time_end=60)
# res = trace_window.find_force_event()

# f_a = res[0]
# f_x_min= res[1]
# f_x_max = res[2]
# f_y_min = res[3]

# g = __getSaw__(trace_window.get_time(),f_a,f_x_min,f_x_max,f_y_min)

# vals = trace_window.get_force()
# plt.plot(vals)
# plt.plot(g)
# plt.show()


