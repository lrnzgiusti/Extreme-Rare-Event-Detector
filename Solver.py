# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:08:11 2019

@author: logiusti
"""
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from astropy.convolution import Gaussian1DKernel,convolve
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import ConvexHull, distance_matrix
import Loader
from Utility import timeit
import json
import os

class Solver:
    r"""

    Class wich provies all the functionalities to do math on our data

    """
    def __init__(self):
        self.loader = Loader.Loader(r"C:\Users\logiusti\Lorenzo\Data\ups")
        self.loader.set_all_public_variables()

    def get_all_pof(self, df):

        def sigmoid(x):
            """
            parametrizzare
            """
            return 1/(1+np.exp(-x+10))

        def get_one_pof(p0, p, eta, clicks):
            """
            parametrizzare
            """
            distance = 1-(1/(1+np.linalg.norm(p0-p, 1)))
            pof_eta_load = sigmoid(0.75*eta**.5 + 0.6*clicks**.5)
            pof = distance*pof_eta_load**.5
            return pof

        def get_p0_name():
            # test points
            pts = df[[2, 3, 4]].to_numpy()

            # two points which are fruthest apart will occur as vertices of the convex hull
            candidates = pts[ConvexHull(pts).vertices]

            # get distances between each pair of candidate points
            dist_mat = distance_matrix(candidates, candidates)

            # get indices of candidates that are furthest apart
            i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

            #get the data into the df according to the most distance points
            tmp_df = df[(df[[2, 3, 4]].to_numpy() == candidates[j]) |
                        (df[[2, 3, 4]].to_numpy() == candidates[i])]

            #return the one who has lower clicks and lower age
            return tmp_df.assign(f=tmp_df['eta']**2 * tmp_df['clicks']**2)\
                         .sort_values('f')\
                         .drop('f', axis=1)\
                         .iloc[0]['UPS']


        v = []
        p0 = df.loc[df['UPS'] == get_p0_name()][[2, 3, 4]].to_numpy()
        for _, row in df.iterrows():
            p = np.array([row[2], row[3], row[4]])
            v.append(get_one_pof(p0, p, row['eta'], row['clicks']))
        return pd.Series(v)





    def retrive_coefficients(self, f):
        r"""
        Compute the Generalized Fourier Coefficients for a functon f.

        Now we compute the best 30 coefficient which describes our function,
        in a future version the number of coefficients will be a result of an optimization problem.

        We project the function onto a cosine basis

        Parameters
        -------
        f : The function we wanna retrive the coefficients in a numpy.array form

        Returns
        -------
        best_coeffs : numpy.array
            array containing the best coefficients for describe the input function.

        Examples
        --------
        >>> from Solver import retrive_coefficients
        >>> retrive_coefficients(f)
        array([ 4950.        , -2879.00578948,  -382.32454756,  -182.59023075,
                -127.56162164,  -104.91636767,   -93.45366287,   -86.86121299,
                 -82.72537469,   -79.96165859,   -78.02413703,   -76.61368594,
                 -75.5552643 ,   -74.74090667,   -74.10108704,   -73.5893936 ,
                 -73.17388555,   -72.83199846,   -72.54742568,   -72.30814614,
                 -72.10514101,   -71.93153736,   -71.78202383,   -71.65244394,
                 -71.5395078 ,   -71.4405845 ,    70.00357134,    70.00357134,
                  70.00357134,    70.00357134])
        """

        def cos_basis(x, j):
            return np.array([1]*len(x)) if j == 0 else np.array(np.sqrt(2)*np.cos(x*np.pi*j))

        cos_domain = np.linspace(0, 1, len(f)) #cosine basis is orthonormal only in [0,1]
        jmax = 100

        coeffs = np.array([0]*(jmax), dtype=np.float64)

        for i in range(0, jmax):
            coeffs[i] = np.inner(f, cos_basis(cos_domain, i))


        def search(A, B):
            r"""
            Retrive the indeces of the elements of A in B
            """
            v = np.zeros(len(A), dtype=np.int8)
            k = 0
            for i in range(len(A)):
                for j in range(len(B)):
                    if A[i] == B[j] and k < len(v):
                        v[k] = j
                        k += 1
            return v

        def best_idxs(coeffs, jmax):

            test_coeff = np.abs(coeffs)
            test_coeff[::-1].sort()

            idx = search(test_coeff, np.abs(coeffs))
            """
            res = np.zeros(jmax, dtype=np.float64)

            j = 0
            for i in range(len(idx)):
                n = len(test_coeff)
                #this determines the minimum number of coefficient i should pick
                trim = round(len(test_coeff)/2)
                min_idx = n - trim - 1
                sigmasq = (n/trim)*sum((test_coeff[min_idx:])**2)
                #risk using the first j coefficients
                res[j] = (i*(sigmasq/n) + sum((test_coeff[(i+1):n]**2-sigmasq/n)))
                j += 1
            """
            optimal_num = 50#np.where(res == min(res))[0][0]

            idx = search(test_coeff[:optimal_num], np.abs(coeffs))

            return idx, optimal_num

        idx, jmax = best_idxs(coeffs, jmax)

        best_coeffs = coeffs[idx]

        return best_coeffs

    @timeit
    def retrive_derivation_of_temperature(self, metric, trim=0, smooth=True):
        ups_to_derivation = {}
        ups_to_temperature = self.loader.ups_to_temperature
        for ups in ups_to_temperature:
            df_tmp = ups_to_temperature[ups]
            if df_tmp.Time.is_monotonic and not df_tmp.empty:

                temperature = df_tmp.Temperature.to_numpy()

                if smooth:
                    stdev = np.std(temperature)*3

                    gauss_kernel = Gaussian1DKernel(stdev)
                    smoothed_temperature = convolve(temperature, gauss_kernel)
                    temperature = smoothed_temperature#[:-3] #last points are bad due to the nature of convolution

                dT = np.gradient(temperature, edge_order=2)[:-3] #same here
                energy_of_dTemperature = np.cumsum(dT**2) #how much is changed the system over time
                signed_total_variation = np.cumsum(dT**3) #how much is changed the system over time considering it's behavour
                if len(signed_total_variation) < 15:
                    continue
                dSTV = np.gradient(signed_total_variation, edge_order=2)
                dEnergy = np.gradient(energy_of_dTemperature, edge_order=2) #the speed in which the system is chagning
                ups_to_derivation[ups] = eval(metric) #you'll get the metric you want
        return ups_to_derivation


    def metric_to_best_trim(self):
        r"""
            For each metric, you'll get back the right value of trim level.
            You get the best according to calinski_harabasz_score with eta as y.

            Parameters
            ----------


            Returns
            -------
            metric_to_best_trim_level: dict
                dictionary containing the ups position as key,
                the desired function of the temperature as value

            Examples
            --------
            >>> from wrapper import get_best_trim_params
            >>> get_best_trim_params()
            {
                 'ESS520_SLASH_E91':                 Time           Temperature
                                     0   2015-02-20 07:53:11.205    22.400000
                                     1   2015-03-02 10:46:26.936    23.100000,

                'ESS608_SLASH_X83':                  Time           Temperature
                                     0   2015-01-21 09:09:45.056    23.299999
                                     1   2015-01-21 10:53:25.250    23.299999
            }

        """


        def single_metric_best_trim(self, start, stop, metric):
            best_score = 0
            best_trim = 0
            cols = [2,3,4]
            for trim in range(start, stop, -1):
                df = pd.DataFrame()
                ups_to_derivation = self.retrive_derivation_of_temperature(trim, metric)
                for ups in ups_to_derivation:
                    sequence = ups_to_derivation[ups]
                    eta =  self.loader.ups_to_eta[ups]
                    coefficients = self.retrive_coefficients(sequence)
                    df = df.append(pd.DataFrame([ups, eta, *coefficients]).T)

                df.reset_index(inplace=True, drop=True)
                df.rename(columns = {0:'UPS', 1:'eta'}, inplace=True)

                X = df[cols].to_numpy()
                y = df['eta'].to_numpy().astype(int)
                score = metrics.calinski_harabasz_score(X,y)

                if score > best_score:
                    best_score = score
                    best_trim = trim

            return (best_score, best_trim)

        derivations = ['temperature', 'dT', 'energy_of_dTemperature', 'signed_total_variation', 'dSTV', 'dEnergy']
        metric_to_best_trim = {}

        if os.path.isfile(r"./data/metric_to_best_trim.json"):
            with open(r"./data/metric_to_best_trim.json", "r") as input_file:
                metric_to_best_trim.update(json.load(input_file))
                not_covered = set(derivations).difference(set(metric_to_best_trim.keys()))
                if len(not_covered) == 0:
                    return metric_to_best_trim
                derivations = not_covered

        for metric in derivations:
            pool = ThreadPool(processes=7)
            async_result = pool.starmap_async(single_metric_best_trim,
                                  [(self, 200, 160, metric),  (self, 160, 120, metric),
                                   (self, 120, 90, metric),  (self, 90, 60, metric),
                                   (self, 60, 40, metric), (self, 40, 20, metric),
                                   (self, 20, 0, metric)]) # tuple of args for foo (3,2),  (2,1),(1,0)

            pool.close()
            pool.join()
            d = {k[0] : k[1] for k in async_result.get()}
            metric_to_best_trim[metric] = d[max(d)]
        json.dump(metric_to_best_trim, open(r"./data/metric_to_best_trim.json", 'w'))
        return metric_to_best_trim


    def save_filtered_metrics(self, metric_to_best_trim):
        for metric in metric_to_best_trim:
            df = pd.DataFrame()
            ups_to_derivation = self.retrive_derivation_of_temperature(metric_to_best_trim[metric], metric)
            for ups in ups_to_derivation:
                sequence = ups_to_derivation[ups]
                eta =  self.loader.ups_to_eta[ups]
                coefficients = self.retrive_coefficients(sequence)
                df = df.append(pd.DataFrame([ups, eta, *coefficients]).T)

            df.reset_index(inplace=True, drop=True)
            df.rename(columns = {0:'UPS', 1:'eta'}, inplace=True)

            with open(r"./data/filtered_"+metric+".pickle", "wb") as output_file:
                pickle.dump(df, output_file)


    def ups_to_max_min(self):
        metric_to_best_trim = {}

        if os.path.isfile(r"./data/ups_to_max_min.pickle"):
            with open(r"./data/ups_to_max_min.pickle", "rb") as input_file:
                return pickle.load(input_file)

        with open(r"./data/metric_to_best_trim.json", "r") as input_file:
                metric_to_best_trim.update(json.load(input_file))
                if len(metric_to_best_trim) == 0:
                    metric_to_best_trim = self.metric_to_best_trim()

        ups_to_max_min = {}
        for metric in metric_to_best_trim:
            ups_to_derivation = self.retrive_derivation_of_temperature(metric_to_best_trim[metric], metric, smooth=False)
            ups_to_max_min[metric]= {ups : (min(ups_to_derivation[ups]), max(ups_to_derivation[ups])) for ups in ups_to_derivation}

        with open(r"./data/ups_to_max_min.pickle", "wb") as output_file:
            pickle.dump(ups_to_max_min, output_file)

        return ups_to_max_min

