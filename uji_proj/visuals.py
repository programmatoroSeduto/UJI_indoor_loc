#! /usr/bin/python3

# main frameworks
import numpy as np
import pandas as pd

# sciKitLearn
import sklearn.svm as svm

# UJI files
from ujiProject import paths as uji_paths
from ujiProject.DataAnalysis import *
from ujiProject.DataPreparation import *
from ujiProject.time_utils import time_utils

# per il salvataggio del modello su file
import pickle as pk

'''
    print( "[visuals] ", f" ..." )
    tmu.start( )

    tmu.stop( False )
    print( "[visuals] ", f" ... OK", f" in {np.round(tmu.value, 1)}s" )
'''

if __name__ == "__main__":
    tmu = time_utils( )

    # === LOADING PHASE

    # data loading
    print( "[visuals] ", "loading data from files..." )
    tmu.start( )
    ds_tr = data_loader( empty_class=True )
    ds_tr.load_data( True, "../results/tr_ds.csv", 
		"../results/tr_idx.csv", "../results/tr_idxc.csv", 
		"../results/tr_X.csv", "../results/tr_Xc.csv", 
		"../results/tr_Y.csv", "../results/tr_Yc.csv", 
		"../results/tr_scaler.sav" )
    ds_tt = data_loader( empty_class=True )
    ds_tt.load_data( True, "../results/tt_ds.csv", 
		"../results/tt_idx.csv", "../results/tt_idxc.csv", 
		"../results/tt_X.csv", "../results/tt_Xc.csv", 
		"../results/tt_Y.csv", "../results/tt_Yc.csv", 
		"../results/tt_scaler.sav" )
    tmu.stop( False )
    print( "[visuals] ", "loading data from files... OK", f" in {np.round(tmu.value, 1)}s" )

    # learners loading
    print( "[visuals] ", f"loading learners ..." )
    tmu.start( )
    lm = [ None, None ]
    with open( '../results/LM_long_data.sav', 'rb' ) as fil:
        lm[0] = pk.load( fil )
    with open( '../results/LM_lat_data.sav', 'rb' ) as fil:
        lm[1] = pk.load( fil )
    tmu.stop( False )
    print( "[visuals] ", f"loading learners ... OK", f" in {np.round(tmu.value, 1)}s" )

    # predictions
    print( "[visuals] ", f"generating predictions ..." )
    tmu.start( )
    y_tt_pred_long = lm[0].predict(ds_tt.X)
    y_tt_pred_lat = lm[1].predict(ds_tt.X)
    tmu.stop( False )
    print( "[visuals] ", f"generating predictions ... OK", f" in {np.round(tmu.value, 1)}s" )



    # === GRAPHS GENERATIONS PHASE

    