# -*- coding: utf-8 -*-
"""SVR_lat.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15kr1HojMYwGawpbwhDZDOPxlgGBijqpY
"""

# frameworks vari
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# toy samples
import sklearn.datasets as datasets

# sciKitLearn
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import sklearn.metrics as metrics

# path handler
from pathlib import Path 

# per il salvataggio del modello su file
import pickle as pk

from google.colab import drive
drive.mount('/content/drive')

training_set_path = '/content/drive/MyDrive/UJI_indoor_loc/UJI_indoor_loc/'
test_set_path = '/content/drive/MyDrive/UJI_indoor_loc/UJI_indoor_loc/'
save_path = '/content/drive/MyDrive/UJI_indoor_loc/UJI_indoor_loc/'

# load dataset
# load dataset
Ds_tr = pd.read_csv( save_path + 'myTrainingSet.csv' ).to_numpy( )
Ds_tt = pd.read_csv( save_path + 'myTestSet.csv' ).to_numpy( )

# separazione del dataset in due parti
X_tr = Ds_tr[:, 0:520]
y_tr = Ds_tr[:, [520, 521]]
X_tt = Ds_tt[:, 0:520]
y_tt = Ds_tt[:, [520, 521]]

# normalizzazione
scaleX = preprocessing.MinMaxScaler( feature_range=(0, 1) )
scaleX.fit( np.row_stack((X_tr, X_tt)) )

X_tr = scaleX.transform( X_tr )
X_tt = scaleX.transform( X_tt )

with open( save_path + 'scaler.sav', "wb" ) as fil:
  pk.dump( scaleX, fil )

svr_param = {
	'C'       : np.logspace( -4, 3, 15 ),
	'gamma'   : np.logspace( -4, 3, 15 ),
	'epsilon' : [0, 0.01]
}

# Model Selection per SVR -- latitudine
H_lat = model_selection.GridSearchCV( 
	estimator  = svm.SVR( kernel='rbf' ),
	param_grid = svr_param,
	scoring    = 'neg_mean_absolute_error',
	cv         = 2,
	verbose    = 2
).fit( X_tr, y_tr[:, 1] )

# parametri del modello
print( "--- Best Params for Latitude ---" )
print( "C : ", H_lat.best_params_['C'] )
# print( "kernel : ", H_lat.best_params_['kernel'] )
print( "gamma : ", H_lat.best_params_['gamma'] )
print( "epsilon : ", H_lat.best_params_['epsilon'] )
print( "---" )

# salvataggio dei coefficienti per la latitudine
# pd.DataFrame( H_lat ).to_csv( save_path + "params_train_lat.csv" )

# SVR su latitudine
lm_lat = svm.SVR( 
  C = H_lat.best_params_['C'],
	#kernel = H_long.best_params_['kernel'],
	kernel = 'rbf',
	gamma = H_lat.best_params_['gamma'],
	epsilon = H_lat.best_params_['epsilon'] 
).fit( X_tr, y_tr[:, 1] )
print( "--- Score Latitudine --- " )
print( "Score sul training set: ", lm_lat.score( X_tr, y_tr[:, 1] ) )
# print( "Score sul test set: ", lm_lat.score( X_tt, y_tt[:, 1] ) )
print( "---" )

with open( save_path + 'LM_lat_data.sav', 'wb' ) as fil:
  pk.dump( lm_lat, fil )