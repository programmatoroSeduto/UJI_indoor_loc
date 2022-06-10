
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



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



