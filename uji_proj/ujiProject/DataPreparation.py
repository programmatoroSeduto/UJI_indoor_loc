
import datetime
import sys
import pickle as pk

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing

class data_loader:
    '''utilities for loading the dataset and interacting
        with it.

    load and split the data. Utilities to interact with the dataset.

    Attributes:
        ds (ndArray): the original dataset
        idx (row ndArray vector): vector of indices (see resample_data)
        idxc (row ndArray vector): complementary of idx
        X  (ndArray of samples): historical inputs (with a percentage of data)
        Y  (ndArray of samples): historical output (as matrix (percent rows)*2)
        Xc (ndArray of samples): complementary set of X
        Yc (ndArray of samples): complementary set of Y
        rng (np.random.default_rng) : random number generator source
        scaler
    '''


    def __init__( self, data_path="", empty_class=False ):
        ''' try to load the entire dataset
        
        the function tries to open the dataset at the given location,
        without worrying about its type (train or test).
        In case the data set is loaded successfully, calling a split
        method is required in order to obtain the data. 

        Args:
            data_path (string): 
                the path and the file name of the data source
            empty_class (bool, default=False):
                don't load from file.
        '''

        # random number generator
        self.rng = np.random.default_rng( )
        
        # try to read the dataset and take the columns for the problem
        if not empty_class:
            self.ds = (
                pd.read_csv( data_path ).to_numpy( )
                )[:, list(range(0, 520)) + [520, 521]]
        else:
            self.ds = None
        
        # data about the original set
        if not empty_class:
            self.ds_rows = self.ds.shape[0] 
            self.ds_cols = self.ds.shape[1]
        else:
            self.ds_rows = 0 
            self.ds_cols = 0

        # about the split set
        self.samples = 0
        
        # other structures
        self.X      = np.array( [] )
        self.Y      = np.array( [] )
        self.Xc     = np.array( [] )
        self.Yc     = np.array( [] )
        self.idx    = np.array( [] )
        self.idxc   = np.array( [] )
        self.scaler = None

        self._normalized_data = False


    def resample_data( self, percent, copy=False ):
        ''' make the internal data sets.
        
        the function shuffles the original set and takes a percentage
        of rows from it. Calling this is required in order to use
        the data in the training and test phase.

        Args:
            percent (positive float) : the percentage of samples 
                from the original dataset
            copy (bool) : if copy or not the samples. Default: True. 
                If False is set, calling the function copy() is required.

        '''

        # sequence of random indices
        idx = self.rng.permutation( np.arange( 0, self.ds_rows, 1 ) )
        # print( type( idx ) )
        # print( idx )

        # how many samples?
        self.samples = int( np.round( percent * self.ds_rows ) )
        
        # generate idx and idxc
        self.idx =  idx[ 0:self.samples ]
        self.idxc = idx[ (self.samples+1):self.ds_rows ]

        # copy or not
        if copy:
            self.make_sets( )
    

    def make_sets( self ):
        '''create the subsets.
        
        '''
        
        # print( "shape of self.ds" , self.ds.shape )
        # print( self.ds[ 1:15, [520, 521] ] )
        # print( "idx list: ", self.idx )
        # print( self.ds[  ] )

        self.X    = self.ds[ self.idx, 0:520 ]
        self.Y    = self.ds[ self.idx, : ][ :, [520, 521] ]
        self.Xc   = self.ds[ self.idxc, 0:520 ]
        self.Yc   = self.ds[ self.idxc, : ][ :, [520, 521] ]
    

    def save_data( self, save_path, prefix="dataset_", index=False,
        add_timestamp=True, save_complementary_sets=True ):
        '''save the datasets on files
        
        Args:
            save_path (string): the folder where to save the data
            prefix (string): each file will be labeled with a name
                starting with this prefix
            index (bool, optional): make or not index columns in 
                the csv file?
            add_timestamp (bool, optional, default: True):
                add a timestamp at the end of each file name
            save_complementary_sets (bool, optional, default: True):
                save also the complementary sets
        '''

        # correction on the save_path
        if not save_path.endswith( "/" ):
            save_path = save_path + "/"
        
        # timestamp
        tstamp = ""
        if add_timestamp:
            tstamp = "_" + str( datetime.datetime.now( ) ).replace( " ", "_" )
        
        # save the entire dataset
        save_name = save_path + prefix + "ds" + tstamp + ".csv"
        pd.DataFrame( self.ds ).to_csv( save_name, index=index )

        # save the selected samples
        save_name = save_path + prefix + "X" + tstamp + ".csv"
        pd.DataFrame( self.X ).to_csv( save_name, index=index )
        save_name = save_path + prefix + "Y" + tstamp + ".csv"
        pd.DataFrame( self.Y ).to_csv( save_name, index=index )

        # save the complementary samples
        if save_complementary_sets:
            save_name = save_path + prefix + "Xc" + tstamp + ".csv"
            pd.DataFrame( self.X ).to_csv( save_name, index=index )
            save_name = save_path + prefix + "Yc" + tstamp + ".csv"
            pd.DataFrame( self.Y ).to_csv( save_name, index=index )
        
        # save the indices
        save_name = save_path + prefix + "idx" + tstamp + ".csv"
        pd.DataFrame( self.idx ).to_csv( save_name, index=index )
        if save_complementary_sets:
            save_name = save_path + prefix + "idxc" + tstamp + ".csv"
            pd.DataFrame( self.idxc ).to_csv( save_name, index=index )

        # save the scaler if given
        save_name = save_path + prefix + "scaler" + tstamp + ".sav"
        if self.scaler is not None:
            with open( save_name, 'wb' ) as fil:
                pk.dump( self.scaler, fil )


    def save_data_on_file( self, filename ):
        ''' save the object in one .sav file
        
        '''
        with open( filename, 'wb' ) as fil:
            pk.dump( self, fil )


    def load_data( self, norm=False, 
        ds="", idx="", idxc="", X="", Xc="", Y="", Yc="", scaler="" ):
        '''load a previously created dataset from results. 
        
        '''

        self.ds = pd.read_csv( ds ).to_numpy( )
        self.idx = pd.read_csv( idx ).to_numpy( )
        self.idxc = pd.read_csv( idxc ).to_numpy( )
        self.X = pd.read_csv( X ).to_numpy( )
        self.Xc = pd.read_csv( Xc ).to_numpy( )
        self.Y = pd.read_csv( Y ).to_numpy( )
        self.Yc = pd.read_csv( Yc ).to_numpy( )

        with open( scaler, 'rb' ) as fil:
            self.scaler = pk.load( fil )
        self._normalized_data = norm

        self.ds_rows = self.ds.shape[0] 
        self.ds_cols = self.ds.shape[1]
        self.samples = self.X.shape[0]


    def normalize_data( self, inverse_norm=False ):
        ''' normalize the data using a given sciKitLearn scaler

        Args:
            inverse_norm (bool, optional, default: False):
                if True, it performs a inverse normalization;
                otherwise, a direct normalization is performed.
        
        Note:
            the class tracks when the data have been normalized or not
            by a bool private variable which has False as defalt value.
        '''

        # normalize the data
        if inverse_norm and self._normalized_data:
            # inverse normalization
            self.X = self.scaler.inverse_transform( self.X )
            self.X = self.scaler.inverse_transform( self.Xc )
            self._normalized_data = False

        elif not self._normalized_data:
            # direct noralization
            self.X = self.scaler.transform( self.X )
            self.Xc = self.scaler.transform( self.Xc )
            self._normalized_data = True