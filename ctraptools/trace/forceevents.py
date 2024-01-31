from cmath import sqrt

import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt

FORCE = "Force"
TIME = "Time"

class ForceEvents:
    def __init__(self, force, time):
        self.data = pd.DataFrame()
        self.data[FORCE] = force
        self.data[TIME] = time

    # def fitPeak():

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

saw = __getSaw__(30,4,22,26,2.4,20,-0.5,0.02)
