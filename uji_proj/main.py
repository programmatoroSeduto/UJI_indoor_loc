#! /usr/bin/python3

# main frameworks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import time

# sciKitLearn
import sklearn.svm as svm
# import sklearn.metrics as metrics

# UJI files
from ujiProject import paths as uji_paths
from ujiProject.DataAnalysis import *
from ujiProject.DataPreparation import *
from ujiProject.SVRLearning import *
from ujiProject.time_utils import time_utils

# per il salvataggio del modello su file
import pickle as pk

# per calcolare il tempo di esecuzione 
import time 

if __name__ == "__main__":
	# timer
	tmu = time_utils( )

	# data loading
	tmu.start( )
	print( "loading file: ", uji_paths.path_training_set )
	ds_tr = data_loader( uji_paths.path_training_set )
	'''
	ds_tr = data_loader( empty_class=True )
	ds_tr.load_data( True, "../results/tr_Ds.csv", 
		"../results/tr_idx.csv", "../results/tr_idxc.csv", 
		"../results/tr_Xn.csv", "../results/tr_Xcn.csv", 
		"../results/tr_Y.csv", "../results/tr_Yc.csv", 
		"../results/tr_scaler.sav" )
	'''
	print( "loading file: ", uji_paths.path_validation_set )
	ds_tt = data_loader( uji_paths.path_validation_set )
	# ds_tt = data_loader( empty_class=True )
	print( "loading OK" )
	tmu.stop( )

	# find the scaler for the normalization
	tmu.start( )
	print( "Fitting the scaler..." )
	scaler = preprocessing.MinMaxScaler( )
	scaler.fit( np.vstack( ( ds_tr.ds[:, 0:520], ds_tt.ds[:, 0:520] ) ) )
	ds_tr.scaler = scaler
	ds_tt.scaler = scaler
	print( "scaler fitting OK" )
	tmu.stop( )

	# data split and shuffle
	tmu.start( )
	print( "selecting and shuffling data sets ..." )
	ds_tr.resample_data( 0.70, copy=True )
	ds_tt.resample_data( 0.25, copy=True )
	print( "data split OK" )
	tmu.stop( )

	# data normalization
	tmu.start( )
	print( "Data normalization" )
	ds_tr.normalize_data( )
	ds_tt.normalize_data( )
	print( "Data normalization OK" )
	tmu.stop( )

	# save the datasets before starting the work
	tmu.start( )
	print( "writing data on files..." )
	ds_tr.save_data( uji_paths.results_path, "tr_", index=False, add_timestamp=False )
	ds_tt.save_data( uji_paths.results_path, "tt_", index=False, add_timestamp=False )
	print( "dump classes..." )
	ds_tr.save_data_on_file( uji_paths.results_path + "/ds_tr.sav" )
	ds_tt.save_data_on_file( uji_paths.results_path + "/ds_tt.sav" )
	print( "files OK" )
	tmu.stop( )
	
	# model search
	tmu.start( )
	print( "MODEL SEARCH" )
	C = np.array( [1, 2] )
	gamma = np.array( [3, 4] )
	epsilon = np.array( [5] )
	mo = multiout_grid_search( )
	mo.search( ds_tr.X, ds_tr.Y, C, gamma, epsilon, 3 )
	print( "model search OK" )
	print( "params: ", mo.params )
	tmu.stop( )

	# training step
	tmu.start( )
	print( "final training ..." )
	svr_lm = multithread_SVR_learning( )
	lm = svr_lm.train( ds_tr.X, ds_tr.Y, mo.params )
	lm_long = lm[0]
	lm_lat = lm[1]
	tmu.stop( )
	print( "final training OK" )

	print( "saving learners..." )
	tmu.start( )
	print( "--- longitude : ", uji_paths.results_path + '/LM_long_data.sav' )
	with open( uji_paths.results_path + '/LM_long_data.sav', 'wb' ) as fil:
		pk.dump( lm_long, fil )
	tmu.stop( )
	tmu.start( )
	print( "--- latitude : ", uji_paths.results_path + '/LM_lat_data.sav' )
	with open( uji_paths.results_path + '/LM_lat_data.sav', 'wb' ) as fil:
		pk.dump( lm_lat, fil )
	tmu.stop( )
	print( "learners saved OK" )

	