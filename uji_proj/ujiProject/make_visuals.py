
# main frameworks
import numpy as np
import pandas as pd

# sciKitLearn
import sklearn.svm as svm

# UJI files
from ujiProject import paths as uji_paths
import ujiProject.DataAnalysis as vs
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

def make_visuals( ):
    tmu = time_utils( )

    # === LOADING PHASE
    print( "=== LOADING PHASE ===" )

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
    
    print( "--- DATA ---" )
   
    print( " Historical input values from training set  ", ds_tr.X.shape[0] )
    print( " Historical output values from training set ", ds_tr.Y.shape[0] )
    print( f" Data taken for training ={np.round((ds_tr.X.shape[0] / ds_tr.ds_rows ), 1)*100}%" )
   
    print( " Historical input values from test set  ", ds_tt.X.shape[0] )
    print( " Historical output values from test set ", ds_tt.Y.shape[0] )
    print( f" Data taken for validation ={np.round((ds_tt.X.shape[0] / ds_tt.ds_rows ), 1)*100}%" )
    print( "--- DATA ---" )

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
    ym_long = lm[0].predict(ds_tt.X)
    ym_lat = lm[1].predict(ds_tt.X)
    tmu.stop( False )
    print( "[visuals] ", f"generating predictions ... OK", f" in {np.round(tmu.value, 1)}s" )

    # revert normalization
    print( "[visuals] ", f"reverting normalization ..." )
    tmu.start( )
    if ds_tr._normalized_data:
        ds_tr.normalize_data( inverse_norm=True )
    if ds_tt._normalized_data:
        ds_tt.normalize_data( inverse_norm=True )
    tmu.stop( False )
    print( "[visuals] ", f"reverting normalization ... OK", f" in {np.round(tmu.value, 1)}s" )


    # === GRAPHS GENERATIONS PHASE
    print( "=== GRAPHS GENERATIONS PHASE ===" )
    scale = 15
    print( "using scale: ", scale )
    vs.set_fig_scale( scale )

    # simple map
    print( "[visuals] ", f"simple data maps ..." )
    tmu.start( )
    vs.plot_simple_map( ds_tr.Y[:, 0], ds_tr.Y[:, 1], 
      marker='xb',
      title="Simple Map -- training set", 
      outpath="../visuals/tr_map.png" )
    vs.plot_simple_map( ds_tt.Y[:, 0], ds_tt.Y[:, 1], 
      marker='xr',
      title="Simple Map -- validation set", 
      outpath="../visuals/tt_map.png" )
    tmu.stop( False )
    print( "[visuals] ", f"simple data maps ... OK", f" in {np.round(tmu.value, 1)}s" )

    # SVR coefficients
    print( "[visuals] ", f"plot SVR coefficients ..." )
    tmu.start( )
    # vs.set_fig_res( 15, 7 )
    vs.alpha_coefficients_SVR( ds_tr.X.shape[0], lm, outpath="../visuals/SVR_coef.png" )
    # vs.set_fig_scale( scale )
    tmu.stop( False )
    print( "[visuals] ", f"plot SVR coefficients ... OK", f" in {np.round(tmu.value, 1)}s" )

    # real output Vs prediction
    print( "[visuals] ", f"real output Vs prediction ..." )
    tmu.start( )
    vs.set_fig_res( 15, 7 )
    vs.original_vs_machine_out( ds_tt.Y, ym_long, ym_lat, 
      outpath="../visuals/original_vs_machine.png" )
    vs.set_fig_scale( scale )
    tmu.stop( False )
    print( "[visuals] ", f"real output Vs prediction ... OK", f" in {np.round(tmu.value, 1)}s" )

    # scatter plots
    print( "[visuals] ", f"scatter plot ..." )
    tmu.start( )
    vs.set_fig_res( 15, 7 )
    vs.scatter_plot_SVR( ds_tt.Y, ym_long, ym_lat, 
      marker_line="--b", marker_points="xr",
      outpath="../visuals/scatter.png" )
    vs.set_fig_scale( scale )
    tmu.stop( False )
    print( "[visuals] ", f"scatter plot ... OK", f" in {np.round(tmu.value, 1)}s" )

    # training set coverage (see slides)
    print( "[visuals] ", f"training set coverage ..." )
    tmu.start( )
    pc = [ [6, 3, np.Inf], [13, 10, 90], [13, 10, 80], [13, 10, 70] ]
    for p in pc:
      s = ( f"../visuals/tr_coverage_th_{p[0]}_tol_{p[1]}.png" if p[2] == np.Inf
              else f"../visuals/tr_coverage_th_{p[0]}_tol_{p[1]}_mindb_{p[2]}.png" )
      vs.area_coverage_simple( ds_tr.X, ds_tr.Y, 
        threshold=p[0], tolerance=p[1], min_db=p[2],
        outpath=s )
    tmu.stop( False )
    print( "[visuals] ", f"training set coverage ... OK", f" in {np.round(tmu.value, 1)}s" )

    # precision 
    print( "[visuals] ", f"plot precision ..." )
    tmu.start( )
    precl = [50, 25, 10, 5]
    for prec in precl:
      vs.plot_precision( ds_tt.Y, ym_long, ym_lat, 
        max_distance=prec, outpath=f"../visuals/prec_{prec}m_good.png",
        print_worst_cases=False, print_good=True, print_bad=False,
        subtitle="only good samples" )
      vs.plot_precision( ds_tt.Y, ym_long, ym_lat, 
        max_distance=prec, outpath=f"../visuals/prec_{prec}m_good_bad.png",
        print_worst_cases=False, print_good=True, print_bad=True,
        subtitle="" )
      vs.plot_precision( ds_tt.Y, ym_long, ym_lat, 
        max_distance=prec, outpath=f"../visuals/prec_{prec}m_drift.png",
        print_worst_cases=True, print_good=False, print_bad=False,
        subtitle="worst cases only" )
      vs.plot_precision( ds_tt.Y, ym_long, ym_lat, 
        max_distance=prec, outpath=f"../visuals/prec_{prec}m.png",
        print_worst_cases=True, print_good=True, print_bad=True,
        subtitle="with worst cases" )
    tmu.stop( False )
    print( "[visuals] ", f"plot precision... OK", f" in {np.round(tmu.value, 1)}s" )