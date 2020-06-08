# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:50:49 2020

@author: Liz
"""

#Kaggle Project： Classification for Audio Samples
import os
import glob
import time
import numpy as np
import librosa
import librosa.display
import random
import IPython.display as ipd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


random.seed(814)
os.chdir("C:\\Users\\labbi\\OneDrive\\Liz\\Brandeis\\Data mining\\HW2\\data\\urbansound8k")

#to know more about audio
audio_path = 'fold1\\9031-3-1-0.wav'
ipd.Audio(audio_path)

# Extract the audio data (x) and the sample rate (sr).
x, sr = librosa.load(audio_path)
# Plot the sample
plt.figure(figsize=(12, 5))
librosa.display.waveplot(x, sr=sr)
plt.show()

# spectogram
#display Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
plt.colorbar()

# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()

#feature extraction
#zero_crossing: is the rate of sign-changes along a signal, 
#i.e., the rate at which the signal changes from positive to negative or back
zero_crossings = sum(librosa.zero_crossings(x, pad=False))
print(sum(zero_crossings))

#Spectral Centroid: It indicates where the ”centre of mass” for a sound is located 
#and is calculated as the weighted mean of the frequencies present in the sound.
spectral_centroids = librosa.feature.spectral_centroid(X, sr=sr)[0]
spectral_centroids.shape
len(spectral_centroids)
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')

#Spectral Rolloff:Spectral rolloff is the frequency below which a specified percentage of the total spectral energy
spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

#MFCC — Mel-Frequency Cepstral Coefficients
#one of the most important method to extract a feature of an audio signal and is used majorly whenever working on audio signals. 
mfccs = librosa.feature.mfcc(x, sr=sr)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')


#http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    spectral_centroids = len(librosa.feature.spectral_centroid(X, sr=sample_rate)[0])
    zero_crossings = sum(librosa.zero_crossings(X, pad=False))
    return mfccs,chroma,mel,contrast,tonnetz,spectral_centroids,zero_crossings

def load_audio(filelist, file_ext):
    features, labels = np.empty((0,195)), np.empty(0)
    for file in filelist:
        for fn in glob.glob(os.path.join(file, file_ext)):
            try:
                mfccs, chroma, mel, contrast,tonnetz,spectral_centroids,zero_crossings = extract_feature(fn)
            except Exception:
                print(''.join("Error encountered while parsing file: "+file+fn))
                continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz,spectral_centroids,zero_crossings])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('\\')[1].split('-')[1])
            labels = np.array(labels, dtype = np.int)
            print(''.join("complete: "+file+fn))
    return features, labels


file_list = [''.join('fold'+str(i)) for i in range(1,11)]
x_train, y_train = load_audio(filelist=file_list, file_ext="*.wav")

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)

from numpy import savetxt
# save to csv file
savetxt('x_train.csv', x_train_scaled, delimiter=',')
savetxt('y_train.csv', y_train, delimiter=',')
# load numpy array from csv file
from numpy import loadtxt
# load array
x_train_scaled = loadtxt('x_train.csv', delimiter=',')
y_train = loadtxt('y_train.csv', delimiter=',')

#KNN
from sklearn.neighbors import KNeighborsClassifier
grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 17, 21, 23, 25],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=10, n_jobs=-1)
model.fit(x_train_scaled, y_train)
model.best_estimator_
model.cv_results_

KNN = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', metric = 'euclidean')
KNN_cv_results = cross_validate(KNN,x_train_scaled, y_train, cv=10,return_train_score = True)
print(np.mean(KNN_cv_results['train_score']))
print(np.mean(KNN_cv_results['test_score']))


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
start = time.time()
gnb = GaussianNB()
cv_results = cross_validate(gnb,x_train_scaled, y_train, cv=10,return_train_score = True)
print(np.mean(cv_results['train_score']))
print(np.mean(cv_results['test_score']))
end = time.time()
print(end - start)

#SVM
start = time.time()
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
from sklearn.svm import SVC
grid_params = dict(gamma=gamma_range, C=C_range)
model = GridSearchCV(SVC(), grid_params, cv=10, n_jobs=-1)
model.fit(x_train_scaled, y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))
means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']
params = model.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
end = time.time()
print(end - start)

#SVM heatmap
import pandas as pd
pvt = pd.pivot_table(pd.DataFrame(model.cv_results_),
values='mean_test_score', index='param_C', columns='param_gamma')
import seaborn as sns
ax = sns.heatmap(pvt)

rbf_svm = SVC(C=10,gamma=0.01)
rbf_svm_cv_results = cross_validate(rbf_svm,x_train_scaled, y_train, cv=10,return_train_score = True)
print(np.mean(rbf_svm_cv_results['train_score']))
print(np.mean(rbf_svm_cv_results['test_score']))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
criterion = ['gini', 'entropy']
splitter=['best', 'random']
max_features = [None, 'auto', 'sqrt', 'log2']
min_samples_split = [2,5,10,15,20,50,100,150,200,300]
max_leaf_nodes = [10,30,50,100,200,500,1000]
grid_params = dict(criterion=criterion, splitter=splitter,max_features=max_features,min_samples_split=min_samples_split,
                   max_leaf_nodes=max_leaf_nodes)
model = GridSearchCV(DecisionTreeClassifier(), grid_params, cv=10, n_jobs=-1)
model.fit(x_train_scaled, y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))
means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']
params = model.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

start = time.time()
dtc = DecisionTreeClassifier(criterion='entropy', max_features=None, max_leaf_nodes=100, min_samples_split=5, splitter='best')
dtc_cv_results = cross_validate(dtc,x_train_scaled, y_train, cv=10,return_train_score = True)
print(np.mean(dtc_cv_results['train_score']))
print(np.mean(dtc_cv_results['test_score']))
end = time.time()
print(end - start)

#Perceptron
from sklearn.linear_model import Perceptron
max_iter = [50,100,200,500,1000,2000]
shuffle = [True, False]
eta0 = list(np.arange(0, 105, 5))
eta0.pop(0)
grid_params = dict(max_iter=max_iter,shuffle=shuffle,eta0=eta0)
start = time.time()
Perceptrpn_model = GridSearchCV(Perceptron(random_state=814), grid_params, cv=10, n_jobs=-1)
Perceptrpn_model.fit(x_train_scaled, y_train)
print("Best: %f using %s" % (Perceptrpn_model.best_score_, Perceptrpn_model.best_params_))
means = Perceptrpn_model.cv_results_['mean_test_score']
stds = Perceptrpn_model.cv_results_['std_test_score']
params = Perceptrpn_model.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
end = time.time()
print(end - start)

perceptron_model = Perceptron(random_state=814,max_iter=50,eta0=5,shuffle=True)
print(perceptron_model)
perceptron_cv_result = cross_validate(perceptron_model,x_train_scaled, y_train, cv=10,return_train_score = True)
print(np.mean(perceptron_cv_result['train_score']))
print(np.mean(perceptron_cv_result['test_score']))

#Neurol Network
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
def create_model(neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=195, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(4)))
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
grid_result = grid.fit(x_train_scaled, y_train)

#random forest
from sklearn.ensemble import RandomForestClassifier
n_estimators = [5,10,20,50,100,150,200,300,500]
criterion = ['gini','entropy']
max_depth = [3,5,10,15,20,25,30,50,80]
min_samples_split = [2,5,10,15,20,50,100,150,200,300]
max_features = [None, 'auto', 'sqrt', 'log2']
grid_params = dict(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,
                   max_features=max_features)
rfc_model = GridSearchCV(RandomForestClassifier(random_state=814), grid_params, cv=10, n_jobs=-1)
rfc_model.fit(x_train_scaled, y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))
rfc_cv_model=RandomForestClassifier(random_state=814,max_depth=50,criterion = 'gini',n_estimators=300,min_samples_split=2,max_features=None)
rfc_cv_result = cross_validate(rfc_cv_model,x_train_scaled, y_train, cv=10,return_train_score = True)
print(np.mean(rfc_cv_result['train_score']))
print(np.mean(rfc_cv_result['test_score']))