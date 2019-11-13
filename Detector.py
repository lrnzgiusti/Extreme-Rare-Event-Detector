# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:55:50 2019

@author: logiusti
"""

import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

# Common imports
import os
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import IsolationForest


import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Lambda, LeakyReLU
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import EarlyStopping

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


seed(10)
tf.random.set_seed(10)

import Loader
class Detector:
    r'''
    Anomaly detection (or outlier detection) is the identification of rare items,
    events or observations which raise suspicions by differing significantly from the majority of the data.
    Typically, anomalous data can be connected to some kind of problem or rare event such as:
        bank fraud, medical problems, structural defects, malfunctioning equipment etc.
    This connection makes it very interesting to be able to pick out which data points can be considered anomalies,
    as identifying these events are typically very interesting from a business perspective.

    In the case of two-dimensional data (X and Y), or at most three-dimensional data (X, Y and Z),
    it becomes quite easy to visually identify anomalies through data points located outside the typical distribution.
    However, often is not possible to identify the outlier directly from investigating one variable at the time:
        It is the combination of the X and Y (and Z if exist) variable that allows us to easily identify the anomaly.
    This complicates the matter substantially when we scale up from two variables to 10–100s of variables,
    which is often the case in practical applications of anomaly detection.

   '''

    def __init__(self):
        self.loader = Loader.Loader()


    def load_and_scale(self, data):
        r'''
        Before setting up the models,
        we need to define train/test data.
        To do this, we perform a simple split where we train on the 90% of the dataset
        (which should represent normal operating conditions),
        and test on the remaining parts of the dataset leading up to the eventual failure.
        '''
        test_size = np.ceil(len(data)*.15).astype(int)

        dataset_train = data[:-test_size]
        dataset_test = data[-test_size:]

        r'''
        Scale the input variables of the model.

        Standardize features by removing the mean and scaling to unit variance.
        Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
        Mean and standard deviation are then stored to be used on later data using the transform method.
        Standardization of a dataset is a common requirement for many machine learning estimators:
            they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
        For instance many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the L1 and L2 regularizers of linear models) assume that all features are centered around 0 and have variance in the same order.
        If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
        '''
        scaler = preprocessing.MinMaxScaler()

        X_train = pd.DataFrame(scaler.fit_transform(dataset_train),
                               columns=dataset_train.columns,
                               index=dataset_train.index).sample(frac=1)# Random shuffle training data

        X_test = pd.DataFrame(scaler.transform(dataset_test),
                              columns=dataset_test.columns,
                              index=dataset_test.index)

        return (X_train, X_test)

    def scheduler(self):
        ups_to_functionals = self.loader.load_temperature_functionals()
        i = 0
        for ups in ups_to_functionals:
            i += 1
            if i % 20 == 0:
                print(i)
            if ups == 'EBS11_SLASH_38'or ups not in ['ESS329_SLASH_7E', 'EBS2C06_SLASH_BL1', 'ESS328_SLASH_5E', 'ESSXX_SLASH_ZZ', 'ESSXX_SLASH_XX']:#, 'EBS12_SLASH_33'  ,'EBS1_SLASH_56', 'EBS22_SLASH_A6', 'ESS103_SLASH_6R', 'ESS303_SLASH_2X', 'ESSXX_SLASH_YY']:
                continue
            if not os.path.isdir(r'./data/anom/'+ups):
                os.mkdir(r'./data/anom/'+ups)

            #X_train, X_test = self.load_and_scale(ups_to_functionals[ups])
            #md = self.mahalanobis_distances(X_train, X_test, ups)

            X_train, X_test = self.load_and_scale(ups_to_functionals[ups])
            #autoenc = self.nonlinear_autoencoder_detect(X_train, X_test, ups)
            LSTM_autoenc = self.nonlinear_LSTM_autoencoder_detect(X_train, X_test, ups)

            fig, axes = plt.subplots(nrows=3)

            #md.plot(logy=True, figsize=(30, 18), ylim=[1e-3, 1e3], color=['green', 'red'], title='Mahalanobis Anomaly' ,ax=axes[0])
            #autoenc.plot(logy=True, figsize=(30, 18), ylim=[1e-3, 1e3], color=['green', 'red'], title='Autoencoder Anomaly' , ax=axes[1])
            LSTM_autoenc.plot(logy=True, figsize=(30, 18), ylim=[1e-3, 1e3], color=['green', 'red'],  title='LSTM Autoencoder Anomaly',  ax=axes[2])
            plt.savefig(r'./data/anom/'+ups+'/merged.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')

            plt.close('all')



    def mahalanobis_distances(self, X_train, X_test, ups):
        r'''
        The Mahalanobis distance is widely used in cluster analysis and classification techniques.
        In order to use the Mahalanobis distance to classify a test point as belonging to one of N classes,
        one first estimates the covariance matrix of each class,
        usually based on samples known to belong to each class.
        In our case, as we are only interested in classifying “normal” vs “anomaly”,
        we use training data that only contains normal operating conditions to calculate the covariance matrix.
        Then, given a test sample, we compute the Mahalanobis distance to the “normal” class,
        and classifies the test point as an “anomaly” if the distance is above a certain threshold.
        '''

        def is_pos_def(A):
            r'''
            Check if matrix is positive definite.
            '''
            if np.allclose(A, A.T):
                try:
                    np.linalg.cholesky(A)
                    return True
                except np.linalg.LinAlgError:
                    return False
            else:
                return False

        def cov_matrix_f(data):
            r'''
            Calculate the covariance matrix.
            '''
            covariance_matrix = np.cov(data, rowvar=False)
            if is_pos_def(covariance_matrix):
                inv_covariance_matrix = np.linalg.inv(covariance_matrix)
                if is_pos_def(inv_covariance_matrix):
                    return covariance_matrix, inv_covariance_matrix
                else:
                    print("Error: Inverse of Covariance Matrix is not positive definite!")
            else:
                print("Error: Covariance Matrix is not positive definite!")


        def MahalanobisDist(inv_cov_matrix, mean_distr, data):
            r'''
            Calculate the Mahalanobis distance.
            '''
            inv_covariance_matrix = inv_cov_matrix
            vars_mean = mean_distr
            diff = data - vars_mean
            md = []
            for i in range(len(diff)):
                md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
            return md


        def MD_detectOutliers(dist, extreme=False):
            r'''
            Detecting outliers.
            '''
            k = 3. if extreme else 2.
            threshold = np.mean(dist) * k
            outliers = []
            for i in range(len(dist)):
                if dist[i] >= threshold:
                    outliers.append(i)  # index of the outlier
            return np.array(outliers)

        def MD_threshold(dist, extreme=False):
            r'''
            Calculate threshold value for classifying datapoint as anomaly.
            '''
            k = 3. if extreme else 2.
            threshold = np.mean(dist) * k
            return threshold

        r'''
        As dealing with high dimensional sensor data is often challenging,
        there are several techniques to reduce the number of variables (dimensionality reduction).
        One of the main techniques is principal component analysis (PCA),
        which performs a linear mapping of the data to a lower-dimensional space
        in such a way that the variance of the data in the low-dimensional representation is maximized.
        In practice, the covariance matrix of the data is constructed and the eigenvectors of this matrix are computed.
        The eigenvectors that correspond to the largest eigenvalues (the principal components) can now be used to reconstruct a large fraction of the variance of the original data.
        The original feature space has now been reduced (with some data loss, but hopefully retaining the most important variance) to the space spanned by a few eigenvectors.

        We use Kernel Principal component analysis (KPCA) since we are almost sure that the patterns are NON-Linear.

        Non-linear dimensionality reduction through the use of kernels.
        '''

        pca = KernelPCA(n_components=2, kernel='poly', degree=3, coef0=0, fit_inverse_transform=False,
                        remove_zero_eig=True, n_jobs=-1, eigen_solver='auto')

        X_train_PCA = pca.fit_transform(X_train)
        X_train_PCA = pd.DataFrame(X_train_PCA)
        X_train_PCA.index = X_train.index

        X_test_PCA = pca.transform(X_test)
        X_test_PCA = pd.DataFrame(X_test_PCA)
        X_test_PCA.index = X_test.index

        # Define train/test set from the two main principal components.
        data_train = np.array(X_train_PCA.values)
        data_test = np.array(X_test_PCA.values)

        # Calculate the covariance matrix and its inverse, based on data in the training set.
        cov_matrix, inv_cov_matrix = cov_matrix_f(data_train)

        r'''
        We also calculate the mean value for the input variables in the training set,
        as this is used later to calculate the Mahalanobis distance to datapoints in the test set
        '''
        mean_distr = data_train.mean(axis=0)

        r'''
        Using the covariance matrix and its inverse,
        we can calculate the Mahalanobis distance for the training data defining “normal conditions”,
        and find the threshold value to flag datapoints as an anomaly.
        One can then calculate the Mahalanobis distance for the datapoints in the test set
        and compare that with the anomaly threshold.
        '''

        dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test)
        dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train)
        threshold = MD_threshold(dist_train, extreme=True)

        r'''
        Threshold value for flagging an anomaly:

        The square of the Mahalanobis distance to the centroid of the distribution should follow a χ2 distribution
        if the assumption of normal distributed input variables is fulfilled.
        This is also the assumption behind the above calculation of the “threshold value” for flagging an anomaly.
        As this assumption is not necessarily fulfilled in our case,
        it is beneficial to visualize the distribution of the Mahalanobis distance to set a good threshold value for flagging anomalies.

        We start by visualizing the square of the Mahalanobis distance, which should then ideally follow a χ2 distribution.
        '''

        plt.figure()
        sns.distplot(np.square(dist_train),
                     bins=75,
                     kde=False);
        plt.xlim([0.0, 15])
        plt.xlabel('Square of the Mahalanobis distance')
        plt.savefig(r'./data/anom/'+ups+'/MahalanobisDistHistSquared.jpeg',
                    quality=95, optimize=True, progressive=True, format='jpeg')


        # Then visualize the Mahalanobis distance itself
        plt.figure()
        sns.distplot(dist_train,
                     bins=75,
                     kde=True,
                     color='green')
        plt.xlim([0.0, 5])
        plt.xlabel('Mahalanobis dist')
        plt.savefig(r'./data/anom/'+ups+'/MahalanobisDistHist.jpeg',
                    quality=95, optimize=True, progressive=True, format='jpeg')


        r'''
        From the above distributions,
        the calculated threshold value is defined as 3 standard deviations from the center of the distribution.
        '''

        r'''
        We can then save the Mahalanobis distance,
        as well as the threshold value and “anomaly flag” variable for both train and test data in a dataframe.
        '''
        anomaly_train = pd.DataFrame()
        anomaly_train['Mob dist'] = dist_train
        anomaly_train['Thresh'] = threshold
        # If Mob dist above threshold: Flag as anomaly
        anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
        anomaly_train.index = X_train_PCA.index
        anomaly = pd.DataFrame()
        anomaly['Mob dist'] = dist_test
        anomaly['Thresh'] = threshold
        # If Mob dist above threshold: Flag as anomaly
        anomaly['Anomaly'] = anomaly['Mob dist'] > anomaly['Thresh']
        anomaly.index = X_test_PCA.index

        r'''
        Based on the calculated statistics, any distance above the threshold value will be flagged as an anomaly.
        We can now merge the data in a single dataframe and save it as a .csv file
        '''

        anomaly_alldata = pd.concat([anomaly_train, anomaly], sort=False)
        anomaly_alldata.sort_index(inplace=True)
        anomaly_alldata.to_csv(r'./data/anom/'+ups+'/Anomaly_distance.csv')


        r'''
        Verifying PCA model on test data:
        We can now plot the calculated anomaly metric (Mob dist),
        and check when it crosses the anomaly threshold (note the logarithmic y-axis).
        '''


        ymax = 100*max(anomaly['Mob dist'])
        ymin = 0.001*min(anomaly['Mob dist'])
        fig = anomaly_alldata.plot(logy=True, figsize=(30, 18), ylim=[ymin, ymax], color=['green', 'red'])
        fig.get_figure().savefig(r'./data/anom/'+ups+'/Anomaly_distance.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
        plt.close('all')
        return anomaly_alldata


    def nonlinear_autoencoder_detect(self, X_train, X_test, ups):
        r'''
        The basic idea here is to use an autoencoder neural network to “compress” the sensor readings to a low dimensional representation,
        which captures the correlations and interactions between the various variables.
        (Essentially the same principle as the PCA model, but here we also allow for non-linearities among the input variables).

        We use a 3 layer neural network:
            First layer has 6 nodes, middle layer has 2 nodes, and third layer has 6 nodes.
        We use the mean square error as loss function, and train the model using the SGD optimizer with 0.8 as momentum value.
        '''

        train = X_train.to_numpy()

        corrupted_train = train.copy()


        random_rows_to_corrupt = np.random.randint(0, corrupted_train.shape[0], size=(int(corrupted_train.size * 0.4),))
        random_cols_to_corrupt = np.random.randint(0, corrupted_train.shape[1], size=(int(corrupted_train.size * 0.4),))

        corrupted_train[random_rows_to_corrupt, random_cols_to_corrupt] = 0


        act_func = LeakyReLU(alpha=0.01)

        # Input layer:
        model = Sequential()
        # First hidden layer, connected to input vector X.
        model.add(Dense(6, activation=act_func,
                        kernel_initializer='glorot_uniform',
                        input_shape=(X_train.shape[1],)))



        model.add(Dense(2, activation=act_func,
                        kernel_initializer='glorot_uniform'))


        model.add(Dense(6, activation=act_func,
                        kernel_initializer='glorot_uniform'))

        model.add(Dense(X_train.shape[1],
                        kernel_initializer='glorot_uniform'))


        adam = tf.keras.optimizers.Adam(amsgrad=True)
        model.compile(loss='mae', optimizer=adam)

        # Train model for 200 epochs, batch size of 32:
        NUM_EPOCHS = 350
        BATCH_SIZE = 64

        r'''
        To keep track of the accuracy during training,
        we use 5% of the training data for validation after each epoch (validation_split = 0.05).
        '''

        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, verbose=0, patience=25)
        history = model.fit(corrupted_train + np.random.normal(), np.array(X_train),
                          batch_size=BATCH_SIZE,
                          epochs=NUM_EPOCHS,
                          validation_split=0.05,
                          verbose=0,
                          callbacks=[es])

        # Visualize training/validation loss
        plt.plot(history.history['loss'], 'b', label='Training loss')
        plt.plot(history.history['val_loss'], 'r', label='Validation loss')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('Loss, [mse]')
        ylim = max(max(history.history['loss']), max(history.history['val_loss']))
        plt.ylim([0, ylim])
        plt.savefig(r'./data/anom/'+ups+'/Autoencoder_loss.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
        plt.close()

        r'''
        Distribution of loss function in the training set:

        By plotting the distribution of the calculated loss in the training set,
        one can use this to identify a suitable threshold value for identifying an anomaly.
        In doing this, one can make sure that this threshold is set above the “noise level”,
        and that any flagged anomalies should be statistically significant above the noise background.
        '''
        X_pred = model.predict(np.array(X_train))
        X_pred = pd.DataFrame(X_pred,
                              columns=X_train.columns)
        X_pred.index = X_train.index

        scored = pd.DataFrame(index=X_train.index)
        scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis=1)
        plt.figure()
        sns.distplot(scored['Loss_mae'],
                     bins=50,
                     kde=True,
                     color='blue')
        plt.xlim([0.0, max(scored['Loss_mae'])])
        plt.savefig(r'./data/anom/'+ups+'/Autoencoder_loss_dist.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
        plt.close()

        r'''
        From the above loss distribution,
        let us try a threshold based on one of the last quantiles for flagging an anomaly.
        We can then calculate the loss in the test set, to check when the output crosses the anomaly threshold.
        '''

        threshold = scored['Loss_mae'].quantile(.95) + 1*(scored['Loss_mae'].quantile(.95) - scored['Loss_mae'].quantile(.05))
        X_pred = model.predict(np.array(X_test))
        X_pred = pd.DataFrame(X_pred,
                              columns=X_test.columns)
        X_pred.index = X_test.index

        scored = pd.DataFrame(index=X_test.index)
        scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis=1)
        scored['Threshold'] = threshold
        scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']

        fig = scored.plot(logy=True, figsize=(30, 18), ylim=[1e-3, 1e3], color=['blue', 'red'])
        fig.get_figure().savefig(r'./data/anom/'+ups+'/Autoencoder_test_threshold.jpeg',
                                 quality=95, optimize=True, progressive=True, format='jpeg')

        plt.close('all')


        r'''
        We then calculate the same metrics also for the training set, and merge all data in a single dataframe.
        '''
        X_pred = model.predict(np.array(X_test))
        X_pred = pd.DataFrame(X_pred,
                              columns=X_test.columns)
        X_pred.index = X_test.index

        scored_test = pd.DataFrame(index=X_test.index)
        scored_test['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis=1)
        scored_test['Threshold'] = threshold
        scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']



        X_pred_train = model.predict(np.array(X_train))
        X_pred_train = pd.DataFrame(X_pred_train,
                                    columns=X_train.columns)
        X_pred_train.index = X_train.index

        scored_train = pd.DataFrame(index=X_train.index)
        scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis=1)
        scored_train['Threshold'] = threshold
        scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']

        scored = pd.concat([scored_train, scored_test], sort=False)

        r'''
        Isolation forest prediction based on anomality detected by the autoencoder
        '''
        contamination = (scored.Anomaly == True).sum()/len(scored)

        clf = IsolationForest(behaviour='new', n_estimators=500, max_samples='auto', contamination=contamination, \
                                max_features=1.0, bootstrap=True, n_jobs=-1, random_state=42, verbose=0)

        clf.fit(scored.Loss_mae.to_numpy().reshape(-1, 1))

        pred = clf.predict(scored.Loss_mae.to_numpy().reshape(-1, 1))
        dct = {-1:threshold, 1:np.nan}
        pred = [dct[k] for k in pred]

        r'''
        Results from Autoencoder model:
        Having calculated the loss distribution and the anomaly threshold, we can visualize the model output in the time leading up to the bearing failure:
        '''

        ymax = 100*max(scored['Loss_mae'])
        ymin = 0.001*min(scored['Loss_mae'])
        scored.plot(logy=True, figsize=(30, 18), ylim=[ymin, ymax], color=['green', 'red'])
        isolation_forest = plt.scatter(scored.index, pred, s=150, c='b', marker=".")
        plt.legend((isolation_forest,), ('Isolation Forest',), scatterpoints=1, loc='upper right', ncol=1, fontsize=8)
        plt.savefig(r'./data/anom/'+ups+'/Autoencoder_full_threshold.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
        plt.close('all')


        scored.sort_index(inplace=True)
        scored.to_csv(r'./data/anom/'+ups+'/Dense_autoencoder_distance.csv')

        return scored


    def nonlinear_LSTM_autoencoder_detect(self, X_train, X_test, ups):

            #act_func = 'relu'
            train = X_train.to_numpy()

            corrupted_train = train.copy()


            random_rows_to_corrupt = np.random.randint(0, corrupted_train.shape[0], size=(int(corrupted_train.size * 0.4),))
            random_cols_to_corrupt = np.random.randint(0, corrupted_train.shape[1], size=(int(corrupted_train.size * 0.4),))

            corrupted_train[random_rows_to_corrupt, random_cols_to_corrupt] = 0

            test = X_test.to_numpy()
            timesteps = 1
            n_features = 6
            train = train.reshape(train.shape[0], timesteps, n_features)
            corrupted_train = corrupted_train.reshape(corrupted_train.shape[0], timesteps, n_features)


            model = Sequential()

            model.add(Bidirectional(LSTM(5, activation='selu',
                                         kernel_initializer='lecun_normal',
                                         input_shape=(timesteps,n_features), return_sequences=True)))
            model.add(Bidirectional(LSTM(3, activation='selu',
                                         kernel_initializer='lecun_normal',
                                         return_sequences=False)))
            model.add(RepeatVector(timesteps))
            model.add(Bidirectional(LSTM(3, activation='selu',
                                         kernel_initializer='lecun_normal',
                                         return_sequences=True)))
            model.add(Bidirectional(LSTM(5, activation='selu',
                                         kernel_initializer='lecun_normal',
                                         return_sequences=True)))
            model.add(TimeDistributed(Dense(n_features)))

            adam = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
            model.compile(optimizer=adam, loss='mse')
            #model.summary()
            # fit model
            es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, verbose=0, patience=35)
            history = model.fit(corrupted_train + np.random.normal() , train , epochs=750, batch_size=64, verbose=0, steps_per_epoch=None, validation_split=0.05, callbacks=[es])

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
            plt.savefig(r'./data/anom/'+ups+'/LSTM_Autoencoder_loss.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')

            X_pred = model.predict(train).reshape(train.shape[0], train.shape[-1])
            X_pred = pd.DataFrame(X_pred,
                                  columns=X_train.columns)
            X_pred.index = X_train.index

            scored_train = pd.DataFrame(index=X_train.index)
            scored_train['Loss_mae'] = np.mean(np.abs(X_pred-train.reshape(train.shape[0],train.shape[-1])), axis = 1)
            plt.figure()
            sns.distplot(scored_train['Loss_mae'],
                         bins = 100,
                         kde= True,
                        color = 'blue');
            plt.xlim([0.0, max(scored_train['Loss_mae'])])

            plt.savefig(r'./data/anom/'+ups+'/LSTM_Autoencoder_loss_dist.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
            plt.close()


            threshold = scored_train['Loss_mae'].quantile(.95) + 1*(scored_train['Loss_mae'].quantile(.95) - scored_train['Loss_mae'].quantile(.05))
            scored_train['Threshold'] = threshold
            scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']

            X_test_pred = model.predict(test.reshape(test.shape[0], timesteps, test.shape[-1])).reshape(test.shape[0], test.shape[-1])
            X_test_pred = pd.DataFrame(X_test_pred,
                                  columns=X_test.columns)
            X_test_pred.index = X_test.index

            scored_test = pd.DataFrame(index=X_test.index)
            scored_test['Loss_mae'] = np.mean(np.abs(X_test_pred-test), axis=1)
            scored_test['Threshold'] = threshold
            scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']

            scored = pd.concat([scored_train, scored_test], sort=False)


            ymax = 100*max(scored['Loss_mae'])
            ymin = 0.001*min(scored['Loss_mae'])
            fig = scored.plot(logy=True, figsize=(30, 18), ylim=[ymin, ymax], color=['blue', 'red'])
            fig.get_figure().savefig(r'./data/anom/'+ups+'/LSTM_Autoencoder_test_threshold.jpeg',
                                     quality=95, optimize=True, progressive=True, format='jpeg')

            plt.close('all')

            scored.sort_index(inplace=True)
            scored.to_csv(r'./data/anom/'+ups+'/LSTM_autoencoder_distance.csv')

            return scored


    def variational_autoencoder(self):

        ups_to_functionals = self.loader.load_temperature_functionals()

        for ups in ups_to_functionals:
            if ups == 'EBS11_SLASH_38'  or len(ups_to_functionals[ups]) < 50:
                continue
            if not os.path.isdir(r'./data/anom/'+ups):
                os.mkdir(r'./data/anom/'+ups)

            r'''
            Before setting up the models,
            we need to define train/test data.
            To do this, we perform a simple split where we train on the 90% of the dataset
            (which should represent normal operating conditions),
            and test on the remaining parts of the dataset leading up to the eventual failure.
            '''
            data = ups_to_functionals[ups]
            test_size = np.ceil(len(data)*.1).astype(int)

            dataset_train = data[:-test_size]
            dataset_test = data[-test_size:]

            r'''
            Scale the input variables of the model.

            Standardize features by removing the mean and scaling to unit variance.
            Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
            Mean and standard deviation are then stored to be used on later data using the transform method.
            Standardization of a dataset is a common requirement for many machine learning estimators:
                they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
            For instance many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the L1 and L2 regularizers of linear models) assume that all features are centered around 0 and have variance in the same order.
            If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
            '''
            scaler = preprocessing.StandardScaler()

            X_train = pd.DataFrame(scaler.fit_transform(dataset_train),
                                   columns=dataset_train.columns,
                                   index=dataset_train.index).sample(frac=1)# Random shuffle training data

            X_test = pd.DataFrame(scaler.transform(dataset_test),
                                  columns=dataset_test.columns,
                                  index=dataset_test.index)


            #act_func = 'relu'
            train = X_train.to_numpy()
            test = X_test.to_numpy()
            intermediate_dim = 8
            batch_size = 64
            latent_dim = 2
            epochs = 150

            def sampling(args):
                """Reparameterization trick by sampling from an isotropic unit Gaussian.

                # Arguments
                    args (tensor): mean and log of variance of Q(z|X)

                # Returns
                    z (tensor): sampled latent vector
                """

                z_mean, z_log_var = args
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                # by default, random_normal has mean = 0 and std = 1.0
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon

            # VAE model = encoder + decoder
            # build encoder model
            inputs = Input(shape=(train.shape[1],), name='encoder_input')
            x = Dense(intermediate_dim, activation='relu')(inputs)
            z_mean = Dense(latent_dim, name='z_mean')(x)
            z_log_var = Dense(latent_dim, name='z_log_var')(x)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

            # instantiate encoder model
            encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

            # build decoder model
            latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
            x = Dense(intermediate_dim, activation='relu')(latent_inputs)
            outputs = Dense(train.shape[1], activation='relu')(x)

            # instantiate decoder model
            decoder = Model(latent_inputs, outputs, name='decoder')

            # instantiate VAE model
            outputs = decoder(encoder(inputs)[2])
            vae = Model(inputs, outputs, name='vae_mlp')


            reconstruction_loss = mse(inputs, outputs)
            reconstruction_loss *= train.shape[1]
            kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
            kl_loss = tf.reduce_sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
            vae.add_loss(vae_loss)
            vae.compile(optimizer='adam')
            history = vae.fit(train,
                    epochs=epochs,
                    batch_size=batch_size,
                  validation_split=0.05,
                  verbose=0)


            # Visualize training/validation loss
            plt.plot(history.history['loss'], 'b', label='Training loss')
            plt.plot(history.history['val_loss'], 'r', label='Validation loss')
            plt.legend(loc='upper right')
            plt.xlabel('Epochs')
            plt.ylabel('Loss, [mse]')
            ylim = max(max(history.history['loss']), max(history.history['val_loss']))
            plt.ylim([0, ylim])
            plt.savefig(r'./data/anom/'+ups+'/VAE_loss.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
            plt.close()

            r'''
            Distribution of loss function in the training set:

            By plotting the distribution of the calculated loss in the training set,
            one can use this to identify a suitable threshold value for identifying an anomaly.
            In doing this, one can make sure that this threshold is set above the “noise level”,
            and that any flagged anomalies should be statistically significant above the noise background.
            '''
            X_pred = vae.predict(np.array(X_train))
            X_pred = pd.DataFrame(X_pred,
                                  columns=X_train.columns)
            X_pred.index = X_train.index

            scored = pd.DataFrame(index=X_train.index)
            scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis=1)
            plt.figure()
            sns.distplot(scored['Loss_mae'],
                         bins=50,
                         kde=True,
                         color='blue')
            plt.xlim([0.0, max(scored['Loss_mae'])])
            plt.savefig(r'./data/anom/'+ups+'/VAE_loss_dist.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
            plt.close()

            r'''
            From the above loss distribution,
            let us try a threshold based on one of the last quantiles for flagging an anomaly.
            We can then calculate the loss in the test set, to check when the output crosses the anomaly threshold.
            '''

            threshold = scored['Loss_mae'].quantile(.95) + 1*(scored['Loss_mae'].quantile(.95) - scored['Loss_mae'].quantile(.05))

            X_pred = vae.predict(np.array(X_test))
            X_pred = pd.DataFrame(X_pred,
                                  columns=X_test.columns)
            X_pred.index = X_test.index

            scored = pd.DataFrame(index=X_test.index)
            scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis=1)
            scored['Threshold'] = threshold
            scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']



            ymax = 100*max(scored['Loss_mae'])
            ymin = 0.001*min(scored['Loss_mae'])
            fig = scored.plot(logy=True, figsize=(30, 18), ylim=[ymin, ymax], color=['blue', 'red'])
            fig.get_figure().savefig(r'./data/anom/'+ups+'/VAE_test_threshold.jpeg',
                                     quality=95, optimize=True, progressive=True, format='jpeg')

            plt.close('all')


            r'''
            We then calculate the same metrics also for the training set, and merge all data in a single dataframe.
            '''
            X_pred = vae.predict(np.array(X_test))
            X_pred = pd.DataFrame(X_pred,
                                  columns=X_test.columns)
            X_pred.index = X_test.index

            scored_test = pd.DataFrame(index=X_test.index)
            scored_test['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis=1)
            scored_test['Threshold'] = threshold
            scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']



            X_pred_train = vae.predict(np.array(X_train))
            X_pred_train = pd.DataFrame(X_pred_train,
                                        columns=X_train.columns)
            X_pred_train.index = X_train.index

            scored_train = pd.DataFrame(index=X_train.index)
            scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis=1)
            scored_train['Threshold'] = threshold
            scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']

            scored = pd.concat([scored_train, scored_test], sort=False)

            r'''
            Isolation forest prediction based on anomality detected by the autoencoder
            '''
            contamination = (scored.Anomaly == True).sum()/len(scored)

            clf = IsolationForest(behaviour='new', n_estimators=500, max_samples='auto', contamination=contamination, \
                                    max_features=1.0, bootstrap=True, n_jobs=-1, random_state=42, verbose=0)

            clf.fit(scored.Loss_mae.to_numpy().reshape(-1, 1))

            pred = clf.predict(scored.Loss_mae.to_numpy().reshape(-1, 1))
            dct = {-1:threshold, 1:np.nan}
            pred = [dct[k] for k in pred]

            r'''
            Results from Autoencoder model:
            Having calculated the loss distribution and the anomaly threshold, we can visualize the model output in the time leading up to the bearing failure:
            '''
            scored.plot(logy=True, figsize=(30, 18), ylim=[1e-3, 1e3], color=['green', 'red'])
            isolation_forest = plt.scatter(scored.index, pred, s=150, c='b', marker=".")
            plt.legend((isolation_forest,), ('Isolation Forest',), scatterpoints=1, loc='upper right', ncol=1, fontsize=8)
            plt.savefig(r'./data/anom/'+ups+'/VAE_full_threshold.jpeg', quality=95, optimize=True, progressive=True, format='jpeg')
            plt.close('all')

    def false_positive_detect(self, crit_seqs_path=r"C:\Users\logiusti\Lorenzo\PyWorkspace\scripts\Wrapper\data\critical_dates",
                              anomalities_path=r'C:\Users\logiusti\Lorenzo\PyWorkspace\scripts\Wrapper\data\anom'):


        ups_to_false_positive = dict()
        def pick_the_peak(df):
            r'''
            We pick the time in which the loss is maximum in a t+-dt period.
            '''
            return df.loc[df['Loss_mae'] == max(df['Loss_mae'])]

        os.chdir(anomalities_path)
        for ups in os.listdir():
            ups_to_false_positive[ups] = []
            # read the critical sequences timestamps of each ups
            try:
                crit_seq = pd.to_datetime(pd.Series(pd.read_csv(crit_seqs_path+"\\"+ups+'_crit_seq_dates.csv', header=None)[0])).sort_values().reset_index(drop=True)
            except FileNotFoundError:
                crit_seq = pd.Series()
            #read the anomaly alterts, we choose the LSTM denoising autoencore since it seems to have better performances.
            anomalities = pd.read_csv("./"+ups+'/LSTM_autoencoder_distance.csv')
            anomalities = anomalities.loc[anomalities['Anomaly'] == True] #use only values flagged as anomaly
            anomalities['Time'] = pd.to_datetime(anomalities['Time'], format="%Y-%m-%d %H:%M:%S.%f")

            if anomalities.Time.empty:
                ups_to_false_positive[ups] = []
                continue
            #tim off, first 10% of the period should not be considered in the false positive count
            dayoff = ((anomalities.Time.iloc[-1] - anomalities.Time.iloc[0]).days)*.1
            #cut-off the DF
            anomalities = anomalities.loc[anomalities['Time']  >= (pd.to_datetime(anomalities.Time.iloc[0], format="%Y-%m-%d") + pd.DateOffset(days=dayoff))]


            #we want to group an entire period in which we've spotted an anomaly (7 days as upper bound)
            anomalities['groups'] = (anomalities.Time.diff().fillna(pd.Timedelta(seconds=0))/np.timedelta64(7, 'D'))\
                                            .gt(1).cumsum().add(1).astype(str)

            anomalities.set_index('groups', drop=True, inplace=True)
            danger = anomalities.groupby(level=0, group_keys=False).apply(pick_the_peak) #take the max of that period
            danger['Time'] = danger['Time'].dt.date

            #if nothing happens in the time(max[])+4days
            for idx,row in danger.iterrows():
                start = pd.to_datetime(row.Time, format="%Y-%m-%d")
                end = start + pd.DateOffset(days=4)
                if crit_seq.loc[(crit_seq >= start ) & (crit_seq <= end) ].empty and row['Loss_mae'] >= 1.5*row['Threshold']:
                    print("Falso Positivo", ups, "\n", row)
                    ups_to_false_positive[ups].append(row['Time'])
        return ups_to_false_positive

r'''

Summary:
Both modeling approaches give similar results,
where they are able to flag the upcoming bearing malfunction well in advance of the actual failure.
The main difference is essentially how to define a suitable threshold value for flagging anomalies,
to avoid to many false positives during normal operating conditions
'''

Detector().scheduler()
#iaia = Detector().false_positive_detect()

#Detector().nonlinear_autoencoder_detect()
#Detector().nonlinear_LSTM_autoencoder_detect()
#Detector().variational_autoencoder()