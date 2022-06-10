
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def plot_simple_map( long, lat, marker='x', set_grid=True, title="Simple Map", outpath="simple_map.png" ):
    ''' simple map plot
    
    '''

    # MATPLOTLIB figure and axes
    fig, ax = plt.subplots( )

    # build the figure
    ax.set_title( title )
    ax.grid( set_grid )
    ax.plot( long, lat, marker )

    # save the figure
    plt.savefig( outpath )