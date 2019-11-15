# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:16:48 2019

@author: logiusti
"""


import os
import numpy as np
import pandas as pd
from Utility import timeit
import pickle
from astropy.convolution import Gaussian1DKernel, convolve

class Loader:
    r"""
    Class which provides all the functionalities to load the data

    """
    def __init__(self, ups_data_path=r"C:\Users\logiusti\Lorenzo\Data\ups"):
        self.__ups_data_path = ups_data_path
        self.__ups_to_commission_date = self.load_commission_dates(r"C:\Users\logiusti\Lorenzo\Data\Grapheable\commission_date_df.csv")
        self.__ups_to_temperature = self.load_temperatures()
        self.__ups_name_list = self.retrive_ups_name_list()
        self.__ups_to_eta = self.get_all_eta()
        self.__ups_to_clicks = self.count_clicks()
        self.__ups_to_overheat = self.count_overheats()


    def set_all_public_variables(self):
        self.ups_to_eta = self.__ups_to_eta
        self.ups_to_temperature = self.__ups_to_temperature

    @timeit
    def count_clicks(self):
        r"""
        Load all the clicks of the ups.

        For click we mean a single load/unload process.

        Returns the dictionary with the ups as key and the associated clicks as value

        in self.__ups_to_clicks there's a copy of the result

        Returns
        -------
        ups_to_clicks : dict
            dictionary containing the ups position as key,
            the data of its temperature as value

        Examples
        --------
        >>> from wrapper import count_clicks
        >>> count_clicks()
        {
             'EAS11_SLASH_2HB': 60,
             'EAS11_SLASH_8H': 94,
             'EAS11_SLASH_A3': 29,
             'EAS11_SLASH_A7': 2,
             'EAS1_SLASH_2HB': 71,
             'EAS1_SLASH_8H': 89,
             'EAS212_SLASH_MS1': 75
        }
        """
        ups_to_clicks = {}

        pwd = os.getcwd()

        if os.path.isfile(r"./data/ups_to_clicks.pickle"):
            with open(r"./data/ups_to_clicks.pickle", "rb") as input_file:
                return pickle.load(input_file)

        os.chdir(self.__ups_data_path)
        for ups in os.listdir():
            df_tmp = pd.read_csv(ups+"/"+ups+"_IBat.csv")
            df_tmp['Time'] = pd.to_datetime(df_tmp['Time'], format="%Y-%m-%d %H:%M:%S.%f")
            df_tmp.drop(df_tmp[(df_tmp['Value'] >= (np.median(df_tmp['Value'])-3*np.std(df_tmp['Value']))) &
                               (df_tmp['Value'] <= (np.median(df_tmp['Value'])+3*np.std(df_tmp['Value'])))].index,
                        inplace=True)

            df_tmp.sort_values(by=['Time'], inplace=True)
            df_tmp.reset_index(drop=True, inplace=True)
            if df_tmp.Time.is_monotonic and not df_tmp.empty:
                #creates timeslots of +- 20 minutes, everything goes into the right bucket
                df_tmp['Ts'] = (df_tmp.Time.diff().fillna(pd.Timedelta(seconds=0))/np.timedelta64(20, 'm'))\
                                .gt(1).cumsum().add(1).astype(str)
                ups_to_clicks[ups] = int(df_tmp['Ts'].to_numpy()[-1])



        os.chdir(pwd)
        with open(r"./data/ups_to_clicks.pickle", "wb") as output_file:
            pickle.dump(ups_to_clicks, output_file)
        return ups_to_clicks

    @timeit
    def count_overheats(self):
        r"""
        Count the times in which the temperatures of each ups has exceed the 40C limit

        Returns the dictinary with Position (functional position of the ups) as key and
        and the numbr of excessive overheats as value.

        in self.__ups_to_overheats there's a copy of the result

        Parameters
        ----------


        Returns
        -------
        ups_to_overheats : dict
            dictionary containing the ups position as key,
            the data of its overheats as value

        Examples
        --------
        >>> from wrapper import count_overheats
        >>> count_overheats()
        {
             'ESS520_SLASH_E91':   6,
            'ESS608_SLASH_X83':   14
        }
        """
        ups_to_overheat = {}
        if os.path.isfile(r"./data/ups_to_overheats.pickle"):
            with open(r"./data/ups_to_overheats.pickle", "rb") as input_file:
                return pickle.load(input_file)

        for ups in self.__ups_to_temperature:
            ups_to_overheat[ups] = self.__ups_to_temperature[ups].Temperature.ge(35).sum()

        with open(r"./data/ups_to_overheats.pickle", "wb") as output_file:
            pickle.dump(ups_to_overheat, output_file)
        return ups_to_overheat

    @timeit
    def load_temperatures(self):
        r"""
        Load all the temperatures associated to the ups.

        Returns the dictinary with Position (functional position of the ups) as key and
        and the dataframe of temperatures associated to the position as value.

        in self.__ups_to_temperature there's a copy of the result

        Parameters
        ----------
        path : string
            The path where the ups folders are.

        Returns
        -------
        ups_to_temperature : dict
            dictionary containing the ups position as key,
            the data of its temperature as value

        Examples
        --------
        >>> from wrapper import load_temperatures
        >>> load_temperatures()
        {
             'ESS520_SLASH_E91':                 Time           Temperature
                                 0   2015-02-20 07:53:11.205    22.400000
                                 1   2015-03-02 10:46:26.936    23.100000,

            'ESS608_SLASH_X83':                  Time           Temperature
                                 0   2015-01-21 09:09:45.056    23.299999
                                 1   2015-01-21 10:53:25.250    23.299999
        }
        """
        ups_to_temperature = {}
        pwd = os.getcwd()
        if os.path.isfile(r"./data/ups_to_temperature.pickle"):
            with open(r"./data/ups_to_temperature.pickle", "rb") as input_file:
                return pickle.load(input_file)
        os.chdir(self.__ups_data_path)

        for ups in os.listdir():
            df_tmp = pd.read_csv(ups+"/"+ups+"_TBat.csv")

            #convert the string associated to the timestamp into python time
            df_tmp['Time'] = pd.to_datetime(df_tmp['Time'], format="%Y-%m-%d %H:%M:%S.%f")

            #remove ouliers statically
            df_tmp.drop(df_tmp[(df_tmp['Value'] <= 7.5) |
                               (df_tmp['Value'] >= 60)].index, inplace=True)

            #earlier events comes first
            df_tmp.sort_values(by=['Time'], inplace=True)
            df_tmp.reset_index(drop=True, inplace=True)
            df_tmp.rename(columns={'Value':'Temperature'}, inplace=True)
            ups_to_temperature[ups] = df_tmp

        os.chdir(pwd)
        with open(r"./data/ups_to_temperature.pickle", "wb") as output_file:
            pickle.dump(ups_to_temperature, output_file)
        return ups_to_temperature


    @timeit
    def load_temperature_functionals(self, remove_seasonals=True):
        r"""
        Load all the functionals related to the temperature for each ups.

        Returns a dictionary of dictionaries, in wich,
        the first key is the ups functional position.
        Given an ups, the sub-dictionary has the functional name as key and
        the associated timeseries as value.
        The temperature is first filtered by a gaussian kernel.

        in self.__ups_to_functionals there's a copy of the result

        Parameters
        ----------
        filter_std : int
            The standard deviation of the Gaussian kernel for filter the temperature.

        Returns
        -------
        ups_to_functionals : dict
            dictionary containing the ups position as key,
            metric name-to-timeseries as value:
                the metric name works as key of another sub-ditionary with the timeseries as value

        Examples
        --------
        >>> from wrapper import load_temperature_functionals
        >>> load_temperature_functionals()
        {
             'ESS520_SLASH_E91':
                 'Temperature':
                                     Time                       Temperature
                                 0   2015-02-20 07:53:11.205    22.400000,
                                 1   2015-03-02 10:46:26.936    23.100000,
                    'dT':
                                Time                            dT
                                 0   2015-02-20 07:53:11.205    2.400000,
                                 1   2015-03-02 10:46:26.936    -1.100000,

            'ESS608_SLASH_X83':
                'Temperature':
                                     Time                       Temperature
                                 0   2014-04-15 11:03:27.304    25.400000,
                                 1   2014-04-22 10:35:49.307    23.100000,
                    'dT':
                                Time                            dT
                                 0   2014-04-15 11:03:27.304    0.66165188
                                 1   2014-04-22 10:35:49.307    0.100000,
        }
        """

        def seasonal_decomposition(x, period):
            """Extracts the seasonal components of the signal x, according to period"""
            num_period = len(x) // period
            assert(num_period > 0)

            x_trunc = x[:num_period*period].reshape((num_period, period))
            x_season = np.mean(x_trunc, axis=0)
            x_season = np.concatenate((np.tile(x_season, num_period), x_season[:len(x) % period]))
            return x_season

        def automatic_seasonality_remover(x, k_components=10, verbose=False):
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
                x_season = seasonal_decomposition(x, period)
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

        def temporalize(X, lookback):
            output_X = []
            for i in range(len(X)-lookback-1):
                t = []
                for j in range(1,lookback+1):
                    # Gather past records upto the lookback period
                    t.append(X[[(i+j+1)]])
                output_X.append(t)
            return output_X

        '''
        try:
            ups_to_temperature = self.__ups_to_temperature
        except NameError:
            ups_to_temperature = self.load_temperatures()
        '''
        ups_to_temperature = self.load_temperatures()
        ups_to_functionals = {}

        noisy_temperature = {}
        clean_temperature = {}
        for ups in ups_to_temperature:
            if len(ups_to_temperature[ups]['Temperature']) <= 11:
                continue
            if remove_seasonals:
                ups_to_temperature[ups]['season'] = automatic_seasonality_remover(ups_to_temperature[ups]['Temperature'].values)
                ups_to_temperature[ups]['Temperature'] = ups_to_temperature[ups]['Temperature'] - ups_to_temperature[ups]['season']



            """
            Noisy temperature
            """
            filtered_temperature = ups_to_temperature[ups].Temperature


            gauss_kernel = Gaussian1DKernel(filtered_temperature.std()**2)
            filtered_temperature = convolve(filtered_temperature, gauss_kernel)

            dTemperature = np.gradient(filtered_temperature, edge_order=2)[:-2]
            energy_of_dTemperature = np.cumsum(dTemperature**2) #how much is changed the system over time
            signed_total_variation = np.cumsum(dTemperature**3) #how much is changed the system over time considering it's behavour
            dEnergy = np.gradient(energy_of_dTemperature, edge_order=2) #the speed in which the system is chagning
            dSTV = np.gradient(signed_total_variation, edge_order=2)

            noisy_temperature[ups] = pd.DataFrame()
            noisy_temperature[ups]['T']    = filtered_temperature[:-2]
            noisy_temperature[ups]['dT']   = dTemperature
            noisy_temperature[ups]['EdT']  = energy_of_dTemperature
            noisy_temperature[ups]['STV']  = signed_total_variation
            noisy_temperature[ups]['EdE']  = dEnergy
            noisy_temperature[ups]['dSTV'] = dSTV
            noisy_temperature[ups].index = pd.to_datetime(ups_to_temperature[ups].Time[:-2], format="%Y.%m.%d %H:%M:%S.%f")

            """
            Clean temperature
            """
            filtered_temperature = ups_to_temperature[ups].season

            gauss_kernel = Gaussian1DKernel(filtered_temperature.std()**2)
            filtered_temperature = convolve(filtered_temperature, gauss_kernel)

            dTemperature = np.gradient(filtered_temperature, edge_order=2)[:-2]
            energy_of_dTemperature = np.cumsum(dTemperature**2) #how much is changed the system over time
            signed_total_variation = np.cumsum(dTemperature**3) #how much is changed the system over time considering it's behavour
            dEnergy = np.gradient(energy_of_dTemperature, edge_order=2) #the speed in which the system is chagning
            dSTV = np.gradient(signed_total_variation, edge_order=2)

            clean_temperature[ups] = pd.DataFrame()
            clean_temperature[ups]['T']    = filtered_temperature[:-2]
            clean_temperature[ups]['dT']   = dTemperature
            clean_temperature[ups]['EdT']  = energy_of_dTemperature
            clean_temperature[ups]['STV']  = signed_total_variation
            clean_temperature[ups]['EdE']  = dEnergy
            clean_temperature[ups]['dSTV'] = dSTV
            clean_temperature[ups].index = pd.to_datetime(ups_to_temperature[ups].Time[:-2], format="%Y.%m.%d %H:%M:%S.%f")


            ups_temperature = ups_to_temperature[ups].Temperature
            gauss_kernel = Gaussian1DKernel(ups_temperature.std()**2)
            smoothed_data_gauss = convolve(ups_temperature, gauss_kernel)
            filtered_temperature = smoothed_data_gauss
            dTemperature = np.gradient(filtered_temperature, edge_order=2)[:-2]
            energy_of_dTemperature = np.cumsum(dTemperature**2) #how much is changed the system over time
            signed_total_variation = np.cumsum(dTemperature**3) #how much is changed the system over time considering it's behavour
            dEnergy = np.gradient(energy_of_dTemperature, edge_order=2) #the speed in which the system is chagning
            dSTV = np.gradient(signed_total_variation, edge_order=2)

            ups_to_functionals[ups] = pd.DataFrame()
            ups_to_functionals[ups]['T']    = filtered_temperature[:-2]
            ups_to_functionals[ups]['dT']   = dTemperature
            ups_to_functionals[ups]['EdT']  = energy_of_dTemperature
            ups_to_functionals[ups]['STV']  = signed_total_variation
            ups_to_functionals[ups]['EdE']  = dEnergy
            ups_to_functionals[ups]['dSTV'] = dSTV
            ups_to_functionals[ups].index = pd.to_datetime(ups_to_temperature[ups].Time[:-2], format="%Y.%m.%d %H:%M:%S.%f")


        return ups_to_functionals



    @timeit
    def load_commission_dates(self, path):
        r"""
        Load the file with all the commission date associated to the ups

        Returns the dataframe with two columns:
            Position (functional position of the ups), Commission date

        in self.__ups_to_commission_date there's a copy of the result

        Parameters
        ----------
        path : string
            The path where the csv file relies.

        Returns
        -------
        cd_df : pandasDataframe()
            Dataframe containing the commission date associated to the ups functional position

        Examples
        --------
        >>> from wrapper import load_commission_dates
        >>> load_commission_dates(r"C:\Users\logiusti\Lorenzo\Data\Grapheable\commission_date_df.csv")
                     Position           Commission Date
                0    EAS1_SLASH_2HB      2014-02-27
                1    EAS1_SLASH_8H       2013-12-18
                2    EAS11_SLASH_2HB     2014-02-27
                3    EAS11_SLASH_8H      2013-12-18
                4    EAS11_SLASH_A1      2018-09-13
        """

        if os.path.isfile(r"./data/ups_to_commission_date.pickle"):
            with open(r"./data/ups_to_commission_date.pickle", "rb") as input_file:
                return pickle.load(input_file)

        cd_df = pd.read_csv(r""+path)
        cd_df['Position'] = cd_df['Position'].apply(lambda x: x.replace("/", "_SLASH_").replace("*", "_STAR_"))
        cd_df['Commission Date'] = pd.to_datetime(cd_df['Commission Date'], format="%m/%d/%Y")


        with open(r"./data/ups_to_commission_date.pickle", "wb") as output_file:
            pickle.dump(cd_df, output_file)

        return cd_df


    @timeit
    def retrive_ups_name_list(self):
        r"""
        Get the list of the functional positions we have to analyze.

        Returns a numpy array containing the list of our ups,
        it will search the list from one of the local dataframes


        this name has only the ups for which the temperature has non-zero length

        in self.__ups_name_list there's a copy of the result

        Returns
        -------
        ups_name_list : list
                list of ups names

       Examples
        --------
        >>> from wrapper import retrive_ups_name_list
        >>> retrive_ups_name_list()
            array(['EAS1_SLASH_2HB', 'EAS1_SLASH_8H', 'EAS11_SLASH_2HB',
                   'EAS11_SLASH_8H', 'EAS11_SLASH_A1', 'EAS11_SLASH_A2',
                   'EAS11_SLASH_A4', 'EAS11_SLASH_A5', 'EAS11_SLASH_A6'], dtype=object)
        """
        ups_name_list = self.__ups_to_temperature.keys()
        return ups_name_list

    @timeit
    def get_all_eta(self):
        r"""
        Create a dictionary which associate to each ups its eta in months

        Returns a a dictionary, the key is the ups name, the value is its eta

        if the value is None means that the ups has no data

        in self.__ups_to_eta there's a copy of the result

        Returns
        -------
            ups_to_eta : dict
                Eta in months of the ups passed as parameter

       Examples
        --------
        >>> from wrapper import get_one_eta
        >>> get_all_eta()
            {
             'EAS11_SLASH_2HB': 66.82,
             'EAS11_SLASH_8H': 69.05,
             'EAS11_SLASH_A1': None,
             'EAS11_SLASH_A2': None,
             'EAS11_SLASH_A3': 30.44,
             'EAS11_SLASH_A4': None,
             'EAS11_SLASH_A5': None
            }
        """
        def get_one_eta(ups):
            r"""
            Get the eta' in months of one ups.

            Returns a float number, the decimal part is the fraction of days in a month.

            Parameters
            ----------
            ups : the name of the ups (its functional position)

            Returns
            -------
                eta : float
                    Eta in months of the ups passed as parameter

           Examples
            --------
            >>> from wrapper import get_one_eta
            >>> get_one_eta("EAS11_SLASH_8H")
                69.05
            """
            cd_df = self.__ups_to_commission_date
            df = self.__ups_to_temperature[ups]
            if len(df) > 0:
                timedelta = (df.iloc[-1].Time - cd_df.loc[cd_df['Position'] == ups, "Commission Date"].iloc[0])/np.timedelta64(1, 'M')
                eta = round(timedelta, 2)
            else:
                return None
            return eta

        if os.path.isfile(r"./data/ups_to_eta.pickle"):
            with open(r"./data/ups_to_eta.pickle", "rb") as input_file:
                return pickle.load(input_file)
        ups_to_eta = {ups : get_one_eta(ups) for ups in self.__ups_name_list}


        with open(r"./data/ups_to_eta.pickle", "wb") as output_file:
            pickle.dump(ups_to_eta, output_file)

        return ups_to_eta
