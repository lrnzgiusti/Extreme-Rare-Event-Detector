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
#from astropy.convolution import Gaussian1DKernel, convolve

#from sklearn import preprocessing

#import tensorflow as tf
#from tensorflow.keras.layers import Input, Dropout, Lambda, LeakyReLU
#from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional
#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.losses import mse
#from tensorflow.keras.callbacks import EarlyStopping


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
        norm_params = np.array([[2, 1],
                                [5, 1],
                                [15, 1]])
        #n_components = norm_params.shape[0]
        # Weight of each component, in this case all of them are 1/3
        weights = [0.90, 0.099] #np.ones(n_components, dtype=np.float64) / 3.0
        weights.append(1-sum(weights))
        # A stream of indices from which to choose the component
        mixture_idx = np.random.choice(len(weights), size=num, replace=True, p=weights)
        # y is the mixture sample
        y = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                           dtype=np.float64)

        noisy_ts = cos+y
        for i in range(len(noisy_ts)):
            if (noisy_ts[i] >= noisy_ts[i-1]+7) and (i >= len(noisy_ts)*0.85):
                for j in range(i+1, len(noisy_ts)):
                    noisy_ts[j] = noisy_ts[j-1]+np.random.normal()
                return noisy_ts, 1
        return noisy_ts, 0

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



def temporalize(X, lookback):
    output_X = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)]])
        output_X.append(t)
    return output_X


def flatten(X):
    '''
    Flatten a 3D array.

    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.

    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)












plt.style.use('seaborn')

import random, string, os

kkk = 0
for i in range(500):
    x = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))
    s = Simulator()
    t = s.generate_random_date_sequence("2013-12-12 14:52:35", "2019-09-12 12:56:04", 1500)
    T, anomalous = s.generate_temps(1500)

    kkk += anomalous

    df = pd.DataFrame()
    df['Value']    = T
    df['Time']    = t
    os.mkdir(r"C:\Users\logiusti\Lorenzo\Data\virtual\\"+x+"_"+str(anomalous)+"/")
    df.to_csv(r"C:\Users\logiusti\Lorenzo\Data\virtual\\"+x+"_"+str(anomalous)+"/"+x+"_"+str(anomalous)+"_TBat.csv", index=False)





o = 0
i = 0
for virt in os.listdir(r"C:\Users\logiusti\Lorenzo\Data\virtual"):
    if virt.split("_")[1] == '1':
        i += 1
    else:
        o += 1





r"""

s = Simulator()
t = s.generate_random_date_sequence("2013-12-12 14:52:35", "2019-09-12 12:56:04", 1500)
T = s.generate_temps(1500)


df = pd.DataFrame(index=t)
df['T']    = T
df.plot()

T = T - s.automatic_seasonality_remover(T)

lookback = 5

gauss_1D_kernel = Gaussian1DKernel(.7*np.std(T))
T= convolve(T,gauss_1D_kernel)

filtered_temperature = T
dTemperature = np.gradient(filtered_temperature, edge_order=1)
energy_of_dTemperature = np.cumsum(dTemperature**2) #how much is changed the system over time
signed_total_variation = np.cumsum(dTemperature**3) #how much is changed the system over time considering it's behavour
dEnergy = np.gradient(energy_of_dTemperature, edge_order=1) #the speed in which the system is chagning
dSTV = np.gradient(signed_total_variation, edge_order=1)

df = pd.DataFrame()
df['T']    =  filtered_temperature
df['dT']   = dTemperature
df['EdT']  = energy_of_dTemperature
df['STV']  = signed_total_variation
df['EdE']  = dEnergy
df['dSTV'] = dSTV
df.index = pd.to_datetime(t, format="%Y.%m.%d %H:%M:%S.%f")

scaler = preprocessing.MinMaxScaler()

test_size = np.ceil(len(df)*.15).astype(int)

dataset_train = df[:-test_size]
dataset_test = df[-test_size:]

scaler = preprocessing.MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(dataset_train),
                       columns=dataset_train.columns,
                       index=dataset_train.index)#.sample(frac=1)# Random shuffle training data

X_test = pd.DataFrame(scaler.transform(dataset_test),
                      columns=dataset_test.columns,
                      index=dataset_test.index)

df = X_train.append(X_test, sort=False)

frame = df.to_numpy()
features = 6

tmp =  np.apply_along_axis(temporalize, 0, frame[:,0], lookback=lookback)
for i in range(1,features):
 tmp = np.append(tmp, np.apply_along_axis(temporalize, 0, frame[:,i], lookback=lookback),axis=2)


train = tmp[:-test_size]
test = tmp[-test_size:]




model = Sequential()

model.add(Bidirectional(LSTM(16, activation='relu',
                             kernel_initializer='lecun_normal',
                             input_shape=(lookback,features), return_sequences=True)))

model.add(Bidirectional(LSTM(5, activation='relu',
                             kernel_initializer='lecun_normal',
                             return_sequences=False)))
model.add(RepeatVector(5))
model.add(Bidirectional(LSTM(5, activation='relu',
                             kernel_initializer='lecun_normal',
                             return_sequences=True)))
model.add(Bidirectional(LSTM(16, activation='relu',
                             kernel_initializer='lecun_normal',
                             return_sequences=True)))
model.add(TimeDistributed(Dense(features)))

adam = tf.keras.optimizers.Adam(learning_rate=0.003, amsgrad=True)
model.compile(optimizer=adam, loss='mae')
#model.summary()
# fit model
es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, verbose=0, patience=35)
history = model.fit(train, train , epochs=50, batch_size=64, verbose=1, steps_per_epoch=None, validation_split=0.05, callbacks=[es])


plt.plot(history.history['loss'],
                     'b',
                     label='Training loss')
plt.plot(history.history['val_loss'],
         'r',
         label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
ymax = max(max(history.history['loss']), max(history.history['val_loss']))+0.5
ymin = min(min(history.history['loss']), min(history.history['val_loss']))-0.5
plt.ylim([ymin, ymax])

X_pred = model.predict(train).reshape(train.shape[0], lookback, train.shape[-1])

scored_train = pd.DataFrame(index= t[lookback+1:lookback+train.shape[0]+1])
scored_train['Loss_mae'] = np.mean(flatten(np.abs(X_pred-train)), axis = 1)



threshold = scored_train['Loss_mae'].quantile(.95) + 1*(scored_train['Loss_mae'].quantile(.95) - scored_train['Loss_mae'].quantile(.05))
scored_train['Threshold'] = threshold
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']

X_test_pred = model.predict(test).reshape(test.shape[0], lookback, test.shape[-1])
scored_test = pd.DataFrame(index=t[lookback+train.shape[0]+1:])
scored_test['Loss_mae'] = np.mean(flatten(np.abs(X_test_pred-test)), axis=1)
scored_test['Threshold'] = threshold
scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']

scored = pd.concat([scored_train, scored_test], sort=False)


ymax = 100*max(scored['Loss_mae'])
ymin = 0.001*min(scored['Loss_mae'])
fig = scored.plot(logy=True, figsize=(15, 9), ylim=[ymin, ymax], color=['blue', 'red'])

s = Simulator()
df = pd.read_csv('simulated_data_paper.csv')
#df['T'] = df['T'] - s.automatic_seasonality_remover(df['T'].to_numpy())

#gauss_1D_kernel = Gaussian1DKernel(.7*np.std(df['T']))
#df['T'] = convolve(df['T'],gauss_1D_kernel)







plt.figure(figsize=(18,10))
#df.plot(subplots=True,  layout=(2,3), sharex=True, sharey=False, legend=False)
#[ax.legend(loc=1) for ax in plt.gcf().axes]
df['T'].plot()
plt.tight_layout()
plt.savefig(r'temp.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
plt.show()



df['season'] = s.automatic_seasonality_remover(df['T'].values)
df['Noise'] = df['T'] - df['season']


gauss_kernel = Gaussian1DKernel(df['Noise'].std()**2)
df['Filt'] =  convolve(df['Noie'], gauss_kernel)



plt.figure(figsize=(28,10))
df.plot(subplots=True,  layout=(2,3), sharex=True, sharey=False, legend=False)
[ax.legend(loc=1) for ax in plt.gcf().axes]
plt.tight_layout()
plt.savefig(r'tempz.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
plt.show()

"""
