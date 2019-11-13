# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:31:09 2019

@author: logiusti
"""

import random
import time
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from astropy.convolution import Gaussian1DKernel, convolve
class Simulator:
    def __init__(self):
        pass

    def generate_random_date(self, start, end):
        def str_time_prop(start, end, format, prop):
            """Get a time at a proportion of a range of two formatted times.

            start and end should be strings specifying times formated in the
            given format (strftime-style), giving an interval [start, end].
            prop specifies how a proportion of the interval to be taken after
            start.  The returned time will be in the specified format.
            """

            stime = time.mktime(time.strptime(start, format))
            etime = time.mktime(time.strptime(end, format))

            ptime = stime + prop * (etime - stime)

            return time.strftime(format, time.localtime(ptime))


        def random_date(start, end, prop):
            return str_time_prop(start, end, '%Y-%m-%d %H:%M:%S', prop)

        return random_date(start, end, random.random())

    def generate_random_date_sequence(self, start, end, num):
        dates = [self.generate_random_date(start, end) for _ in range(num)]
        return pd.Series(dates).sort_values().reset_index(drop=True)

    def generate_temps(self,num):
        t = np.linspace(0, 10,num)
        cos = 20+7.5*(np.cos(2*np.pi*t/2))
        # Parameters of the mixture components
        norm_params = np.array([[0, 1],
                                [1, 1],
                                [5, 1]])
        #n_components = norm_params.shape[0]
        # Weight of each component, in this case all of them are 1/3
        weights = [0.949, 0.0499] #np.ones(n_components, dtype=np.float64) / 3.0
        weights.append(1-sum(weights))
        # A stream of indices from which to choose the component
        mixture_idx = np.random.choice(len(weights), size=num, replace=True, p=weights)
        # y is the mixture sample
        y = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                           dtype=np.float64)

        return cos+y

    def seasonal_decomposition(self, x, period):
        """Extracts the seasonal components of the signal x, according to period"""
        num_period = len(x) // period
        assert(num_period > 0)

        x_trunc = x[:num_period*period].reshape((num_period, period))
        x_season = np.mean(x_trunc, axis=0)
        x_season = np.concatenate((np.tile(x_season, num_period), x_season[:len(x) % period]))
        return x_season

    def automatic_seasonality_remover(self, x, k_components=10, verbose=False):
        """Extracts the most likely seasonal component via FFT"""
        f_x = np.fft.rfft(x - np.mean(x))
        f_x = np.real(f_x * f_x.conj())
        periods = len(x) / f_x.argsort()[-k_components:][::-1]
        periods = np.rint(periods).astype(int)
        min_error = None
        best_period = 0
        best_season = np.zeros(len(x))

        for period in periods:
            if period == len(x): continue
            x_season = self.seasonal_decomposition(x, period)
            error = np.average((x - x_season)**2)
            if verbose:
                print("Testing period: {}. Error: {}".format(period, error))

            if min_error is None or error < min_error:
                min_error = error
                best_season = x_season
                best_period = period

        if verbose:
            print("Best fit period: {}".format(best_period))
        return best_season


plt.style.use('seaborn')
s = Simulator()
t = s.generate_random_date_sequence("2013-12-12 14:52:35", "2019-09-12 12:56:04", 1500)
T = s.generate_temps(1500)
df = pd.DataFrame()
df['Time'] = t[:-3]
#df['Value'] = T


filtered_temperature = T
dTemperature = np.gradient(filtered_temperature, edge_order=2)[:-3]
energy_of_dTemperature = np.cumsum(dTemperature**2) #how much is changed the system over time
signed_total_variation = np.cumsum(dTemperature**3) #how much is changed the system over time considering it's behavour
dEnergy = np.gradient(energy_of_dTemperature, edge_order=2) #the speed in which the system is chagning
dSTV = np.gradient(signed_total_variation, edge_order=2)

df['T']    = filtered_temperature[:-3]
df['dT']   = dTemperature
df['EdT']  = energy_of_dTemperature
df['STV']  = signed_total_variation
df['EdE']  = dEnergy
df['dSTV'] = dSTV
df.index = pd.to_datetime(t[:-3], format="%Y.%m.%d %H:%M:%S.%f")


plt.figure(figsize=(18,10))
df.plot(subplots=True,  layout=(2,3), sharex=True, sharey=False, legend=False)
[ax.legend(loc=1) for ax in plt.gcf().axes]
plt.tight_layout()
#plt.savefig(r'err1.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
plt.show()

"""

df['season'] = s.automatic_seasonality_remover(df['Temperature'].values)
df['Noise'] = df['Temperature'] - df['season']


gauss_kernel = Gaussian1DKernel(df['Temperature'].std()**2)
df['Filt'] =  convolve(df['Temperature'], gauss_kernel)

gauss_kernel = Gaussian1DKernel(df['Filt'].std()**2)
df['Filts'] =  convolve(df['season'], gauss_kernel)


plt.figure(figsize=(28,10))
df.plot(subplots=True,  layout=(2,3), sharex=True, sharey=False, legend=False)
[ax.legend(loc=1) for ax in plt.gcf().axes]
plt.tight_layout()
#plt.savefig(r'err1.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
plt.show()

"""