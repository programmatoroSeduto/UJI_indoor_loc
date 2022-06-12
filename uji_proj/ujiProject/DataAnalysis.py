#! /usr/bin/python3

import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 7)

import matplotlib.pyplot as plt
import pandas as pd

# importing classes 

from ujiProject.locPrecision import *
from ujiProject.wapCoverage import *

# Commented out IPython magic to ensure Python compatibility.
# %pylab inline


'''
# learning machines
LM_long = None
LM_lat = None

with open( path + 'LM_long_data.sav', 'rb' ) as fil:
  LM_long = pk.load( fil )

with open( path + 'LM_lat_data.sav', 'rb' ) as fil:
  LM_lat = pk.load( fil )

## scaler
scaler = None
with open( path + 'input_scaler.sav', 'rb' ) as fil:
  scaler = pk.load( fil )

# training set e test set
Ds_tr = pd.read_csv( path + 'myTrainingSet.csv' ).to_numpy( )
Ds_tt = pd.read_csv( path + 'myTestSet.csv' ).to_numpy( )

X_tr = scaler.transform( Ds_tr[:, 0:520] )
y_tr = Ds_tr[:, [520, 521]]
X_tt = scaler.transform( Ds_tt[:, 0:520] )
y_tt = Ds_tt[:, [520, 521]]

print( "Training Set Samples: ", Ds_tr.shape[0] )
print( "Validation Set Samples: ", Ds_tt.shape[0] )
'''


''' PREDICTION
ym_long = LM_long.predict( X_tt )
ym_lat = LM_lat.predict( X_tt )
'''



## PRIMA FUNZIONE DA IPLEMENTARE

def alpha_coefficients_SVR (X_tr, lm= [], outpath='alpha_coeff_svr.png'):
    '''alhpa coefficient plot 
    
    Function to be called for plotting longitude and latitude
    data.

    Args: 
        lm (list): 
            this list contains the two learning machines
    '''
    

    # MATPOTLIB figure and axes
    fig,ax = plt.subplots(nrows=2)

    # build the figure 
    
    ax[0].grid( True )
    ax[0].set_title( f"Longitude ({lm[0].dual_coef_.shape[1]} samples out of {X_tr.shape[0]} trainig samples -- C={lm[1].C} eps={lm[1].epsilon})" )
    ax[0].plot( np.array( range( 0, lm[0].dual_coef_.shape[1] ) ), lm[0].dual_coef_[0, :], 'xb' )

    ax[1].grid( True )
    ax[1].set_title( f"Latitude ({lm[1].dual_coef_.shape[1]} samples out of {X_tr.shape[0]} trainig samples -- C={lm[0].C} eps={lm[0].epsilon})" )
    ax[1].plot( np.array( range( 0, lm[1].dual_coef_.shape[1] ) ), lm[1].dual_coef_[0, :], 'xb' )

    # save the figure
    plt.savefig(outpath) 


## SECONDA FUNZIONE -- VISUALIZZAZIONE TRAINING SET 

def plot_simple_map( long, lat, marker='x', set_grid=True, title="Simple Map", outpath="simple_map.png" ):
    ''' simple map plot
    
    Function to be called for plotting longitude and latitude
    data.

    Args:
        long () : values for longitude 
        lat () : values for latitude 
    '''

    # MATPLOTLIB figure and axes
    fig, ax = plt.subplots( )

    # build the figure
    ax.set_title( title )
    ax.grid( set_grid )
    ax.plot( long, lat, marker )

    # save the figure
    plt.savefig( outpath )

## TERZA FUNZIONE -- COMPARISON BTW ORIGINAL MAP AND MACHINE OUTPUT 

def original_vs_machine_out(y_tt, ym_long, ym_lat, outpath = "or_vs_machine_output.png"):
    '''Comparison between original map and machine output

    Function for inspecting twwo different outputs. Note that the
    original output belongs to  the test set  

    Args:
    '''

    # MATPLOTLIB figure and axes

    fig2, ax2 = plt.subplots( ncols = 2 )

    ## building the plot
    
    # 1st plot - original test set output
    ax2[0].set_title( "Real output" )
    ax2[0].grid( True )
    ax2[0].plot( y_tt[:, 0], y_tt[:, 1], 'xr' )
    ax2[0].set_xlabel( "longitude" )
    ax2[0].set_ylabel( "latitude" )

    # 2nd plot - machine's output
    ax2[1].set_title( "Machine output" )
    ax2[1].grid( True )
    ax2[1].plot( ym_long, ym_lat, 'ob' )
    ax2[1].set_xlabel( "longitude" )
    ax2[1].set_ylabel( "latitude" )


    # save figure
    plt.savefig( outpath )


## QUARTA FUNZIONE -- SCATTER PLOT PER SVR 

def scatter_plot_SVR(y_tt, ym_long, ym_lat, outpath = "scatter_plot_SVR.png"):
    '''Scatter Plot for SVR
    '''

    # MATPLOTLIB figure and axes 
    fig3, ax3 = plt.subplots( ncols = 2 )

    
    sc_minmax = list() 
    sc_minmax.append( [np.min(y_tt[:, 0]), np.max(y_tt[:, 0])] )
    sc_minmax.append( [np.min(y_tt[:, 1]), np.max(y_tt[:, 1])] )
    print( sc_minmax )

    fig3.suptitle( "Scatter Plots for Longitude and Latitude" )

    # scatter plot for longitude
    ax3[0].set_title( "Longitude" )
    ax3[0].grid( True )
    ax3[0].plot( sc_minmax[0], sc_minmax[0], '--b' )
    ax3[0].plot( y_tt[:, 0], ym_long, 'xr' )
    ax3[0].set_xlabel( "real longitude" )
    ax3[0].set_ylabel( "predicted longitude" )

    # scatter plot for latitude
    ax3[1].set_title( "Latitude" )
    ax3[1].grid( True )
    ax3[1].plot( sc_minmax[1], sc_minmax[1], '--b' )
    ax3[1].plot( y_tt[:, 1], ym_lat, 'xr' )
    ax3[1].set_xlabel( "real latitude" )
    ax3[1].set_ylabel( "predicted latitude" )

    # save the figure
    plt.savefig( outpath )



# QUINTA FUNZIONE -- PLOTTING DELLA COVERAGE SUL TRAINING SET 

def area_coverage_train(ds_tr, threshold, tolerance, min_db, outpath = "area_coverage_tr.png"):
    ''' Coverage plot over training set 
    '''
    thresh = threshold
    tol = tolerance 
    Ds_tr = ds_tr


    # esegui la classificazione in base alla coverage

    wapc = wap_coverage( thresh=6, tol=3, min_db=np.Inf )

    # wapc = wap_coverage( thresh = 13, tol = 10, min_db = np.Inf )

    # wapc = wap_coverage( thresh = 13, tol = 10, min_db = 90 )

    # wapc = wap_coverage( thresh = 13, tol = 10, min_db = 80 )

    # wapc = wap_coverage( thresh = 13, tol = 10, min_db = 70 )

    # wapc = wap_coverage( thresh = 13, tol = 10, min_db = 60 )

   
    wapc.analyze_data( Ds_tr[:, 0:520], Ds_tr[:, 520], Ds_tr[:, 521] )

    print( "good coverage size: ", len(wapc.good_coverage) )
    print( "mid coverage size: ", len(wapc.mid_coverage) )
    print( "low coverage size: ", len(wapc.low_coverage) )

    fig_cov_n, ax_cov_n = plt.subplots( )

    fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- mid N in [{wapc.threshold-wapc.tol}, {wapc.threshold+wapc.tol}] -- threshold N={wapc.threshold}  -- min dB={wapc.min_db}" )

    ax_cov_n.grid( True )
    if wapc.good_coverage.shape[0] > 0 : 
        ax_cov_n.plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
    if wapc.mid_coverage.shape[0] > 0 : 
        ax_cov_n.plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
    if wapc.low_coverage.shape[0] > 0 : 
        ax_cov_n.plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
    ax_cov_n.legend( )

    # save the figure
    plt.savefig( outpath )


# SESTA FUNZIONE -- PLOTTING DELLA COVERAGE SUL TEST  SET 

def area_coverage_test(ds_tt, threshold, tolerance, min_db, outpath = "area_coverage_tt.png"):
    ''' Coverage plot over test set
    
    '''
 #   thresh = threshold
    tol = tolerance 
    Ds_tt = ds_tt


    # esegui la classificazione in base alla coverage
    wapc = wap_coverage( thresh=6, tol=3, min_db=90 )
    wapc.analyze_data( Ds_tt[:, 0:520], Ds_tt[:, 520], Ds_tt[:, 521] )

    print( "good coverage size: ", len(wapc.good_coverage) )
    print( "mid coverage size: ", len(wapc.mid_coverage) )
    print( "low coverage size: ", len(wapc.low_coverage) )

    fig_cov_n, ax_cov_n = plt.subplots( )
    fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- {wapc.threshold - wapc.tol} to {wapc.threshold + wapc.tol} -- min dB={wapc.min_db}" )

    ax_cov_n.grid( True )
    if wapc.good_coverage.shape[0] > 0 : 
        ax_cov_n.plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
    if wapc.mid_coverage.shape[0] > 0 : 
        ax_cov_n.plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
    if wapc.low_coverage.shape[0] > 0 : 
        ax_cov_n.plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
    ax_cov_n.legend( )

    # save the figure
    plt.savefig( outpath )



# SETTIMA FUNZIONE -- PLOTTING DELLA COVERAGE USANDO PREDICTION

def area_coverage_predict(ds_tt, ym_long, ym_lat, thresh, tol, min_db, outpath = "area_coverage_predict.png"):
    """ Coverage plotting exploiting prediction 
    
    Note that the algorithm tends to give inaccurate results even with
    points with wide coverage, simply because we may have set the RSSI 
    value too low. In any case, the number of WAPs says nothing about the amount of signal. 
    """
    Ds_tt = ds_tt

    # esegui la classificazione in base alla coverage
    wapc = wap_coverage( thresh=6, tol=3, min_db=90 )
    wapc.analyze_data( Ds_tt[:, 0:520], Ds_tt[:, 520], Ds_tt[:, 521] )

    print( "good coverage size: ", len(wapc.good_coverage) )
    print( "mid coverage size: ", len(wapc.mid_coverage) )
    print( "low coverage size: ", len(wapc.low_coverage) )

    fig_cov_n, ax_cov_n = plt.subplots( ncols=2 )
    fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- threshold N={wapc.threshold} -- min dB={wapc.min_db}" )

    ax_cov_n[0].grid( True )
    if wapc.good_coverage.shape[0] > 0 : 
        ax_cov_n[0].plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
    if wapc.mid_coverage.shape[0] > 0 : 
        ax_cov_n[0].plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
    if wapc.low_coverage.shape[0] > 0 : 
        ax_cov_n[0].plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
    ax_cov_n[0].legend( )

    ax_cov_n[1].grid( True )
    
    wapc.analyze_data( Ds_tt[:, 0:520], ym_long, ym_lat )
    
    if wapc.good_coverage.shape[0] > 0 : 
        ax_cov_n[1].plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
    if wapc.mid_coverage.shape[0] > 0 : 
        ax_cov_n[1].plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
    if wapc.low_coverage.shape[0] > 0 : 
        ax_cov_n[1].plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
    ax_cov_n[1].legend( )

    # save the figure 
    plt.savefig(outpath)




# OTTAVA FUZNIONE -- QUALITA' DEL LEARNING -- ANALISI GRAFICA 

def learning_quality(ds_tt, ym_long, ym_lat, outpath = "learning_quality_SVR.png"):
    """ Graphical analysis of the Learning quality 

   Interestingly, the forecasts tend to be closer to the centre of the graph,
   because that is the 'central' area if we consider the distribution of WAP 
   in the area. 

    """
    Ds_tt = ds_tt

    prec = loc_precision( threshold=50, max_worst_cases=500 )
    prec.analyze_data( Ds_tt[:, 520], Ds_tt[:, 521], ym_long, ym_lat )


    figq, axq = plt.subplots( )
    figq.suptitle( f"Precision of the learner -- threshold={prec.threshold}" )


    axq.grid( True )
    axq.set_title( "LM output" )
    if prec.good_f.shape[0] > 0:
        axq.plot( prec.good_f[:, 0], prec.good_f[:, 1], '.g', label=f"in the threshold ({prec.good.shape[0]} samples, max {prec.threshold}m, min distance {np.round(prec.min_d, 1)}m)" )
    pass
    if prec.bad_f.shape[0] > 0:
        axq.plot( prec.bad_f[:, 0], prec.bad_f[:, 1], 'xr', label=f"out of threshold ({prec.bad.shape[0]} samples, max distance {np.round(prec.max_d, 1)}m)" )
    pass

    # print casi pessimi
    '''
    for i in range(0, prec.max_d_y.shape[0]):
    axq.plot( prec.max_d_x[i, :], prec.max_d_y[i, :], '--y' )

    axq.plot( prec.max_d_x[:, 0], prec.max_d_y[:, 0], 'ob' )
    axq.plot( prec.max_d_x[:, 1], prec.max_d_y[:, 1], 'or' )
    '''
    axq.legend( )

    # save figure
    plt.savefig( outpath ) 

# NONA FUNZIONE -- PRECISIONE VS N° WAP RILEVATI

def coverage_analysis_wap_detected(threshold, ds_tt, ym_long, ym_lat, outpath = "prec_vs_num_wap.png"):
    """# Precisione Vs. Numero di WAP rilevati"""
    
    Ds_tt = ds_tt
    # esegui la classificazione in base alla coverage
    prec = loc_precision( threshold=10, max_worst_cases=50 )
    prec.analyze_data( Ds_tt[:, 520], Ds_tt[:, 521], ym_long, ym_lat )
    wapc = wap_coverage( thresh=7, tol=0, min_db=75 )
    wapc.analyze_data( Ds_tt[:, 0:520], Ds_tt[:, 520], Ds_tt[:, 521] )

    # print( "good coverage size: ", len(wapc.good_coverage) )
    # print( "mid coverage size: ", len(wapc.mid_coverage) )
    # print( "low coverage size: ", len(wapc.low_coverage) )

    fig_cov_n, ax_cov_n = plt.subplots( ncols=2 )
    fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- threshold N={wapc.threshold} -- min dB={wapc.min_db}" )

    ax_cov_n[0].grid( True )
    ax_cov_n[0].set_title( "LM output" )
    if prec.good_f.shape[0] > 0:
        ax_cov_n[0].plot( prec.good_f[:, 0], prec.good_f[:, 1], '.g', label=f"in the threshold ({prec.good.shape[0]} samples, max {prec.threshold}m, min distance {np.round(prec.min_d, 1)}m)" )
    if prec.bad_f.shape[0] > 0:
        ax_cov_n[0].plot( prec.bad_f[:, 0], prec.bad_f[:, 1], 'xr', label=f"out of threshold ({prec.bad.shape[0]} samples, max distance {np.round(prec.max_d, 1)}m)" )

    for i in range(0, prec.max_d_y.shape[0]):
        ax_cov_n[0].plot( prec.max_d_x[i, :], prec.max_d_y[i, :], '--y' )

    ax_cov_n[0].plot( prec.max_d_x[:, 0], prec.max_d_y[:, 0], 'ob' )
    ax_cov_n[0].plot( prec.max_d_x[:, 1], prec.max_d_y[:, 1], 'or' )
    ax_cov_n[0].legend( )

    ax_cov_n[1].grid( True )
    wapc.analyze_data( Ds_tt[:, 0:520], ym_long, ym_lat )
    if wapc.good_coverage.shape[0] > 0 : 
        ax_cov_n[1].plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
    if wapc.mid_coverage.shape[0] > 0 : 
        ax_cov_n[1].plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
    if wapc.low_coverage.shape[0] > 0 : 
        ax_cov_n[1].plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
    ax_cov_n[1].legend( )

    # save figure
    plt.savefig( outpath ) 


def classify_thresh_coverage(ds_tt, ym_long, ym_lat, path = "../ujiProject" ):
    '''classification keeping into account coverage areas 
    '''
    Ds_tt = ds_tt
    # esegui la classificazione in base alla coverage
    prec = loc_precision( threshold=10, max_worst_cases=50 )
    prec.analyze_data( Ds_tt[:, 520], Ds_tt[:, 521], ym_long, ym_lat )
    wapc = wap_coverage( thresh=7, tol=0, min_db=75 )
    wapc.analyze_data( Ds_tt[:, 0:520], Ds_tt[:, 520], Ds_tt[:, 521] )

    # print( "good coverage size: ", len(wapc.good_coverage) )
    # print( "mid coverage size: ", len(wapc.mid_coverage) )
    # print( "low coverage size: ", len(wapc.low_coverage) )

    fig_cov_n, ax_cov_n = plt.subplots( )
    # fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- threshold N={wapc.threshold} -- min dB={wapc.min_db}" )

    ax_cov_n.grid( True )
    ax_cov_n.set_title( "LM output" )
    if prec.good_f.shape[0] > 0:
        ax_cov_n.plot( prec.good_f[:, 0], prec.good_f[:, 1], '.g', label=f"in the threshold ({prec.good.shape[0]} samples, max {prec.threshold}m, min distance {np.round(prec.min_d, 1)}m)" )
    if prec.bad_f.shape[0] > 0:
        ax_cov_n.plot( prec.bad_f[:, 0], prec.bad_f[:, 1], 'xr', label=f"out of threshold ({prec.bad.shape[0]} samples, max distance {np.round(prec.max_d, 1)}m)" )
    ax_cov_n.legend( )
    '''
    for i in range(0, prec.max_d_y.shape[0]):
    ax_cov_n[0].plot( prec.max_d_x[i, :], prec.max_d_y[i, :], '--y' )

    ax_cov_n[0].plot( prec.max_d_x[:, 0], prec.max_d_y[:, 0], 'ob' )
    ax_cov_n[0].plot( prec.max_d_x[:, 1], prec.max_d_y[:, 1], 'or' )
    ax_cov_n[0].legend( )
    '''
    fig_cov_n.savefig( path + "LM_prec.png" )

    fig_cov_n, ax_cov_n = plt.subplots( )

    ax_cov_n.grid( True )
    wapc.analyze_data( Ds_tt[:, 0:520], ym_long, ym_lat )
    if wapc.good_coverage.shape[0] > 0 : 
        ax_cov_n.plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
    if wapc.mid_coverage.shape[0] > 0 : 
        ax_cov_n.plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
    if wapc.low_coverage.shape[0] > 0 : 
        ax_cov_n.plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
    ax_cov_n.legend( )

    plt.savefig( path + "LM_cov.png" )
