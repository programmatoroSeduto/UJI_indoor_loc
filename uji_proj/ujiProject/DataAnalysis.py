
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 10)
import matplotlib.pyplot as plt

from ujiProject.locPrecision import *
from ujiProject.wapCoverage import *


def set_fig_scale( abs_scale ):
    ''' set the matplotlib low level scale.
    
    Note:
        calling this method is the same that giving measures
        ( abs_scale, abs_scale ). 
    '''

    mpl.rcParams['figure.figsize'] = ( abs_scale, abs_scale )


def set_fig_res( rx, ry ):
    ''' set up the low level resolution of MatPlotLib
    
    '''
    mpl.rcParams['figure.figsize'] = ( rx, ry )


def alpha_coefficients_SVR(rows, lm= [], outpath='alpha_coeff_svr.png'):
    '''alhpa coefficient plot 
    
    Function to be called for plotting longitude and latitude
    data.

    Args: 
        lm (list): 
            this list contains the two learning machines
    '''

    # MATPOTLIB figure and axes
    fig,ax = plt.subplots(nrows=2)

    fig.suptitle( "SVR coefficients for Longitude and Latitude" )
    
    ax[0].grid( True )
    ax[0].set_title( f"Longitude coefficients -- ({lm[0].dual_coef_.shape[1]} samples out of {rows} trainig samples ({ np.round((lm[0].dual_coef_.shape[1]/rows), 1)*100 }%) -- C={lm[0].C} eps={lm[0].epsilon})" )
    ax[0].plot( np.array( range( 0, lm[0].dual_coef_.shape[1] ) ), lm[0].dual_coef_[0, :], 'xb' )
    ax[0].set_xlabel( "sample no." )
    ax[0].set_ylabel( "SVR coefficient" )

    ax[1].grid( True )
    ax[1].set_title( f"Latitude coefficients -- ({lm[1].dual_coef_.shape[1]} samples out of {rows} trainig samples ({ np.round((lm[1].dual_coef_.shape[1]/rows), 1)*100 }%) -- C={lm[1].C} eps={lm[1].epsilon})" )
    ax[1].plot( np.array( range( 0, lm[1].dual_coef_.shape[1] ) ), lm[1].dual_coef_[0, :], 'xb' )
    ax[1].set_xlabel( "sample no." )
    ax[1].set_ylabel( "SVR coefficient" )

    # save the figure
    plt.savefig(outpath) 
    plt.close( fig )


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
    plt.close( fig )


def original_vs_machine_out(y_tt, ym_long, ym_lat, outpath="or_vs_machine_output.png"):
    '''Comparison between original map and machine output

    Function for inspecting twwo different outputs. Note that the
    original output belongs to  the test set  

    Args:
    '''

    # MATPLOTLIB figure and axes

    fig, ax = plt.subplots( ncols = 2 )

    fig.suptitle( "Real output Vs. Machine output" )
    
    # 1st plot - original test set output
    ax[0].set_title( "Real output" )
    ax[0].grid( True )
    ax[0].plot( y_tt[:, 0], y_tt[:, 1], 'xr' )
    ax[0].set_xlabel( "longitude" )
    ax[0].set_ylabel( "latitude" )

    # 2nd plot - machine's output
    ax[1].set_title( "Machine output" )
    ax[1].grid( True )
    ax[1].plot( ym_long, ym_lat, 'ob' )
    ax[1].set_xlabel( "longitude" )
    ax[1].set_ylabel( "latitude" )

    # save figure
    plt.savefig( outpath )
    plt.close( fig )


def scatter_simple( ax, y1, y2, smin, smax, marker_line="--b", marker_points="xr", xlabel="", ylabel="", title="", use_grid=True ):
    '''draw a scatter plot on a given axes object
    
    '''

    ax.set_title( title )
    ax.grid( use_grid )
    ax.plot( smin, smax, marker_line )
    ax.plot( y1, y2, marker_points )
    ax.set_xlabel( xlabel )
    ax.set_ylabel( ylabel )


def scatter_plot_SVR(y_tt, ym_long, ym_lat, marker_line="--b", marker_points="xr", outpath="scatter_plot_SVR.png"):
    '''Scatter Plot for SVR

    '''

    # MATPLOTLIB figure and axes 
    fig, ax = plt.subplots( ncols = 2 )

    sc_minmax = list() 
    sc_minmax.append( [np.min(y_tt[:, 0]), np.max(y_tt[:, 0])] )
    sc_minmax.append( [np.min(y_tt[:, 1]), np.max(y_tt[:, 1])] )
    # print( sc_minmax )

    fig.suptitle( "Scatter Plots for Longitude and Latitude" )

    # scatter plot for longitude
    scatter_simple( ax[0], 
        y_tt[:, 0], ym_long, 
        smin=sc_minmax[0], smax=sc_minmax[0], 
        xlabel="real longitude", ylabel="predicted longitude", 
        title="longitude",
        marker_line=marker_line )

    # scatter plot for latitude
    scatter_simple( ax[1], 
        y_tt[:, 1], ym_lat, 
        smin=sc_minmax[1], smax=sc_minmax[1], 
        xlabel="real latitude", ylabel="predicted latitude", 
        title="latitude",
        marker_points=marker_points )

    # save the figure
    plt.savefig( outpath )
    plt.close( fig )


def area_coverage_simple(X, y, threshold=6, tolerance=3, min_db=np.Inf, outpath="area_coverage.png"):
    ''' Coverage plot over training set 

    '''

    wapc = wap_coverage( threshold, tolerance, min_db )
    wapc.analyze_data( X, y[:, 0], y[:, 1] )

    fig, ax = plt.subplots( )

    fig.suptitle( f"Coverage analysis (num of WAP detected) -- mid N in [{wapc.threshold-wapc.tol}, {wapc.threshold+wapc.tol}] -- threshold N={wapc.threshold}  -- min dB={wapc.min_db}" )

    ax.grid( True )
    if wapc.good_coverage.shape[0] > 0 : 
        ax.plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
    if wapc.mid_coverage.shape[0] > 0 : 
        ax.plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
    if wapc.low_coverage.shape[0] > 0 : 
        ax.plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
    ax.legend( )

    # save the figure
    plt.savefig( outpath )
    plt.close( fig )


def plot_precision(y, ym_long, ym_lat, max_distance=50, print_good=True, print_bad=True, print_worst_cases=True, outpath="learning_quality_SVR.png", subtitle="", max_worst_cases=15):
    """ Graphical analysis of the Learning quality 

    Interestingly, the forecasts tend to be closer to the centre of the graph,
    because that is the 'central' area if we consider the distribution of WAP 
    in the area. 

    """

    prec = loc_precision( threshold=max_distance, max_worst_cases=max_worst_cases )
    prec.analyze_data( y[:, 0], y[:, 1], ym_long, ym_lat )

    fig, ax = plt.subplots( )
    fig.suptitle( f"Precision of the learner -- threshold={prec.threshold}m" + ( f" -- {subtitle}" if subtitle!="" else "" ) )


    ax.grid( True )
    ax.set_title( "LM output" )
    ax.set_xlabel( "longitude" )
    ax.set_ylabel( "latitude" )

    if prec.good_f.shape[0] > 0 and print_good:
        ax.plot( prec.good_f[:, 0], prec.good_f[:, 1], '.g', label=f"in the threshold ({prec.good.shape[0]} samples, max {prec.threshold}m, min distance {np.round(prec.min_d, 1)}m)" )
    
    if prec.bad_f.shape[0] > 0 and print_bad:
        ax.plot( prec.bad_f[:, 0], prec.bad_f[:, 1], 'xr', label=f"out of threshold ({prec.bad.shape[0]} samples, max distance {np.round(prec.max_d, 1)}m)" )
    

    # print casi pessimi
    if print_worst_cases:
        for i in range(0, prec.max_d_y.shape[0]):
            ax.plot( prec.max_d_x[i, :], prec.max_d_y[i, :], '--y', label="" )

        ax.plot( prec.max_d_x[:, 0], prec.max_d_y[:, 0], 'ob', label="" )
        ax.plot( prec.max_d_x[:, 1], prec.max_d_y[:, 1], 'or', label="" )

    ax.legend( )

    # save figure
    plt.savefig( outpath ) 
    plt.close( fig )
