
# main frameworks 
from fileinput import _HasReadlineAndFileno
from tracemalloc import stop
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# sciKitLearn
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import sklearn.svm as svm
import sklearn.metrics as metrics

# other libs
import threading


class multithread_grid_search:
    '''Model Selecton for SVR using python threading.
    
    THe class implements a iterative way of using GridSearch in SciKitLearn
    which also uses multithreading in order to save a bit of time.
    '''

    def __init__( self ):
        ''' simple constructor
        
        '''

        # random number generator
        self.rng = np.random.default_rng( )

        # max num of threads
        self.__max_comb = 0
        self.__n_comb = 0
        
        # search set or not?
        self.__search_set = False

        # hyperparameters
        self.__C = []
        self.__gamma = []
        self.__eps = []

        # final hyperparameters
        self.params = []

        # return space for the threads
        # access: [thread_id][y_col]
        self.__return_space = []

        # others
        self.__X_train = []
        self.__y_train = []
    
    
    def set_search( self, C, gamma, epsilon, max_comb_per_thread=15 ):
        '''parameters for the SVR model selection.

        The method will expand the combinations of hyperparameters to be
        tried. This has to be called before starting the search
        of the hyperparameters. 

        Args:
            C (np.array):
                the list of C constants to try
            gamma (np.array):
                the list of gamma constants to try out
            epsilon (np.array):
                the list of epsilons constants to try out
            max_comb_per_thread (int, optional):
                the maximum number of combinations for thread
        '''

        self.__C = []
        self.__gamma = []
        self.__eps = []

        # set the numbe rof maximum combinations per thread
        self.__max_comb = max_comb_per_thread
        
        # generate the combinations (brute force)
        self.__n_comb = C.shape[0] * gamma.shape[0] * epsilon[0]
        for c in C:
            for g in gamma:
                for e in epsilon:
                    self.__C.append( c )
                    self.__gamma.append( g )
                    self.__eps.append( e )

        # read to search
        self.__search_set = True

        print( "[multithread_model_search]", "n combinations: ", self.__n_comb )
        print( "[multithread_model_search]", "max combinations: ", self.__max_comb )
        print( "[multithread_model_search]", "ready for the model search" )


    def __s_thread( self, idx, C, gamma, eps, cv, verbose=False ):
        '''(private) implementation of one GridSearch thread.
        
        Note:
            the arrays C, gamma and eps have the same length.
        '''
        # returns the sorted unique elements of the C,gamma and eps array
        C = np.unique( C )
        gamma = np.unique( gamma )
        eps = np.unique( eps )

        # grid search across the given hyperparameters
        svr_param = {
            'C'       : C,
            'gamma'   : gamma,
            'epsilon' : eps
        }

        for i in range( 0, self.__y_train.shape[1] ):
            print( "[multithread_model_Search] ", f"(thread {idx}) searching for coord. {i} out of {self.__y_train.shape[1]}" )
            H_params = model_selection.GridSearchCV( 
                estimator  = svm.SVR( kernel='rbf' ),
                param_grid = svr_param,
                scoring    = 'neg_mean_absolute_error',
                cv         = 2,
                verbose    = ( 2 if verbose else 0 )
            ).fit( self.__X_train, self.__y_train[:, i] )

            # save the combination of parameters
            #                       RACE CONDITIONS??????
            self.__return_space[idx][i] = H_params


    def search_model( self, X, y, perc_train=.25, perc_test=.10, n_cross_val=2, verbose=False ):
        ''' iterative search of the learning parameters. 
        
        let's suppose to have 100 combinations to test out with GridSearch
        with a maximum of 5 combinations per thread. So, the first step will
        require 100/5=20 threads, each of them trying 5 combinations. The step
        will return 20 different combinations of parameters to test, and because 
        20 >= 5 another step will be required.
        The second step will need 20/5 = 4 threads to go. At the end of this 
        iteration, 4 different combinations will be provided; 4 < 5 so only one
        thread is required to end the job. 
        '''
        
        # check if the search has been set
        if not self.__search_set:
            print( "ERROR: search not set! call .search_set() before calling search_model()" )
            return False
        
        n_tr = int( X.shape[0] * perc_train )
        # n_tt = int( X.shape[0] * perc_test  )
        
        # the algorithm
        # while self.__n_comb > self.__max_comb:
        iter = 1
        while True:

            # empty the return space
            self.__return_space.clear( )
            
            # select some rows of the set for training
            idx_tt = self.rng.permutation( X.shape[0] )
            self.__X_train = X[ idx_tt[0:n_tr], : ]
            self.__y_train = y[ idx_tt[0:n_tr], : ]

            # number of threads?
            n_thread = 1
            if ( self.__n_comb > self.__max_comb ):
                n_thread = int( np.ceil( self.__n_comb / self.__max_comb ) )

            # instanciate the threads and prepare the return space
            thread_list = []
            for i in range( 0, n_thread ):
                # indexes for the hyperparameters
                idx_start = i * self.__max_comb
                idx_stop = (i+1) * self.__max_comb
                if idx_stop > self.__n_comb:
                    idx_stop = self.__n_comb
                
                # create the allocaton space
                # https://stackoverflow.com/questions/10712002/create-an-empty-list-with-certain-size-in-python
                self.__return_space[i] = [None] * y.shape[1]

                # create the thread
                t = t = threading.Thread( 
                    target = self.__s_thread ,
                    args = (
                        i, 
                        self.__C[idx_start:idx_stop], 
                        self.__gamma[idx_start:idx_stop], 
                        self.__eps[idx_start:idx_stop],
                        n_cross_val,
                        verbose
                        ) ,
                    daemon = True
                )

                # append the thread 
                thread_list.append( t )
            
            print( "[multithread_model_Search] ", f"=== ITERATION {iter} launching {n_thread} threads for {self.__n_comb} combination (max is {self.__max_comb})")

            # start the threads
            for i in range( 0, n_thread ):
                t.start( )

            # wait for the thread to end
            for i in np.shuffle( range( 0, n_thread ) ):
                thread_list[i].join( )

            # replace the constants and reset the search
            temp_c = []
            temp_g = []
            temp_e = []
            for i in range( 0, n_thread ):
                temp_c.append( [ H.best_params_['C'] for H in self.__return_space[i] ] )
                temp_g.append( [ H.best_params_['gamma'] for H in self.__return_space[i] ] )
                temp_e.append( [ H.best_params_['epsilon'] for H in self.__return_space[i] ] )
            for i in range( 0, y.shape[1] ):
                temp_c[i, :] = np.unique( temp_c[i, :] )
                temp_g[i, :] = np.unique( temp_g[i, :] )
                temp_e[i, :] = np.unique( temp_e[i, :] )
            
            # reset the search settings
            self.set_search( temp_c, temp_g, temp_e, self.__max_comb )

            # check for stop condition
            if self.__n_comb == 1:
                break
            else:
                iter = iter + 1

        # found "the best" combination
        for i in range( 0, y.shape[1] ):
            self.params.append( self.__return_space[0, i] )
