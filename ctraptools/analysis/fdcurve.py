from cmath import sqrt

import numpy as np
import pandas as pd
import scipy

FORCE = "Force"
DIST = "Distance"

class FDCurve:
    def __init__(self, force, distance):
        self.data = pd.DataFrame()
        self.data[FORCE] = force
        self.data[DIST] = distance

    def estimate_contour_length(self,T,P):
        a = scipy.constants.k*T/P
        ff = self.data[FORCE]
        zz = self.data[DIST]

        Lc = np.empty(shape=(len(ff)))

        for (i,(f,z)) in enumerate(zip(ff,zz)):
            Lc[i] = ((54*a**3*z**3 - 27*a**2*f*z**3 + 3*sqrt(3)*sqrt(135*a**4*f**2*z**6 - 108*a**3*f**3*z**6 + 144*a**2*f**4*z**6 - 64*a*f**5*z**6) + 72*a*f**2*z**3 - 16*f**3*z**3)**(1/3)/(6*2**(1/3)*f) - (-36*a**2*z**2 + 12*a*f*z**2 - 16*f**2*z**2)/(12*2**(2/3)*f*(54*a**3*z**3 - 27*a**2*f*z**3 + 3*sqrt(3)*sqrt(135*a**4*f**2*z**6 - 108*a**3*f**3*z**6 + 144*a**2*f**4*z**6 - 64*a*f**5*z**6) + 72*a*f**2*z**3 - 16*f**3*z**3)**(1/3)) + (3*a*z + 4*f*z)/(6*f)).real
        
        return Lc
    