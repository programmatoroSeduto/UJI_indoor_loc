#! /usr/bin/python3

# main frameworks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# sciKitLearn
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import sklearn.svm as svm
import sklearn.metrics as metrics

# UJI files
from ujiProject import paths as uji_paths
from ujiProject.DataAnalysis import *
from ujiProject.DataPreparation import *
from ujiProject.SVRLearning import *

# per il salvataggio del modello su file
import pickle as pk

if __name__ == "__main__":
	# data loading
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

	# find the scaler for the normalization
	print( "Fitting the scaler..." )
	scaler = preprocessing.MinMaxScaler( )
	scaler.fit( np.vstack( ( ds_tr.ds[:, 0:520], ds_tt.ds[:, 0:520] ) ) )
	ds_tr.scaler = scaler
	ds_tt.scaler = scaler
	print( "scaler fitting OK" )

	# data split and shuffle
	print( "selecting and shuffling data sets ..." )
	ds_tr.resample_data( 0.70, copy=True )
	ds_tt.resample_data( 0.25, copy=True )
	print( "data split OK" )

	# data normalization
	print( "Data normalization" )
	ds_tr.normalize_data( )
	ds_tt.normalize_data( )
	print( "Data normalization OK" )

	# save the datasets before starting the work
	print( "writing data on files..." )
	ds_tr.save_data( uji_paths.results_path, "tr_", index=False, add_timestamp=False )
	ds_tt.save_data( uji_paths.results_path, "tt_", index=False, add_timestamp=False )
	print( "dump classes..." )
	ds_tr.save_data_on_file( uji_paths.results_path + "/ds_tr.sav" )
	ds_tt.save_data_on_file( uji_paths.results_path + "/ds_tt.sav" )
	print( "files OK" )