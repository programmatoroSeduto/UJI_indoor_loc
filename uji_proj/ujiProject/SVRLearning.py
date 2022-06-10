
# main frameworks 
from concurrent.futures import thread
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

from ujiProject.time_utils import time_utils


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
        '''
        print( C.shape )
        print( gamma.shape )
        print( epsilon.shape )
        print( C.shape[0] * gamma.shape[0] * epsilon.shape[0] )
        '''
        # TODO update with a more clever method for generating the combinations
        self.__n_comb = C.shape[0] * gamma.shape[0] * epsilon.shape[0]
        for c in C:
            for g in gamma:
                for e in epsilon:
                    self.__C.append( c )
                    self.__gamma.append( g )
                    self.__eps.append( e )
        
        # print( self.__C )
        # print( self.__gamma )
        # print( self.__eps )

        # read to search
        self.__search_set = True

        print( " [multithread_grid_Search]", "n combinations: ", self.__n_comb )
        print( " [multithread_grid_Search]", "max combinations: ", self.__max_comb )
        print( " [multithread_grid_Search]", "ready for the model search" )


    def __s_thread( self, idx, C, gamma, eps, cv, verbose=False ):
        '''(private) implementation of one GridSearch thread.
        
        Note:
            the arrays C, gamma and eps have the same length.
        '''
        # returns the sorted unique elements of the C,gamma and eps array
        C = np.unique( C )
        gamma = np.unique( gamma )
        eps = np.unique( eps )

        svr_param = {
            'C'       : C,
            'gamma'   : gamma,
            'epsilon' : eps
        }

        if (len(C) * len(gamma) * len(eps) == 1):
            print( " [multithread_grid_Search] ", f"(thread {idx+1}) received only 1 combination! closing..." )
            H_params = model_selection.GridSearchCV( estimator=svm.SVR(), param_grid = svr_param )
            H_params.best_params_ = {}
            H_params.best_params_['C'] = C[0]
            H_params.best_params_['gamma'] = gamma[0]
            H_params.best_params_['epsilon'] = eps[0]

            # save the combination of parameters
            #                       RACE CONDITIONS??????
            self.__return_space[idx] = H_params

            # close the thread
            return

        #print( " [multithread_grid_Search] ", f"(thread {idx+1}) combinations: \nC={C}\ngamma={gamma}\neps={eps}" )

        print( " [multithread_grid_Search] ", f"(thread {idx+1}) JOB BEGINNING, {C.shape[0]*gamma.shape[0]*eps.shape[0]} combinations to test" )
        H_params = model_selection.GridSearchCV( 
            estimator  = svm.SVR( kernel='rbf' ),
            param_grid = svr_param,
            scoring    = 'neg_mean_absolute_error',
            cv         = 2,
            verbose    = ( 2 if verbose else 0 )
        ).fit( self.__X_train, self.__y_train )

        # save the combination of parameters
        #                       RACE CONDITIONS??????
        self.__return_space[idx] = H_params
        
        print( " [multithread_grid_Search] ", f"(thread {idx+1}) END OF THE JOB."  )


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

        Note:
            this implementation is not able to manage multiple outputs
            so far. the vector y must be one-dimensional.
        '''
        
        # check if the search has been set
        if not self.__search_set:
            print( "ERROR: search not set! call .search_set() before calling search_model()" )
            return
        
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
            self.__y_train = y[ idx_tt[0:n_tr] ]

            # number of threads?
            n_thread = 1
            if ( self.__n_comb > self.__max_comb ):
                n_thread = int( np.ceil( self.__n_comb / self.__max_comb ) )

            # instanciate the threads and prepare the return space
            thread_list = []
            for i in range( 0, n_thread ):
                # indexes for the hyperparameters
                idx_start = i * self.__max_comb
                idx_stop = ((i+1) * self.__max_comb)
                # print( idx_start )
                if idx_stop > self.__n_comb:
                    idx_stop = self.__n_comb
                # print( idx_stop )
                
                # create the allocaton space
                # https://stackoverflow.com/questions/10712002/create-an-empty-list-with-certain-size-in-python
                self.__return_space.append( None )

                '''
                print( 
                    self.__C[idx_start:idx_stop], 
                    self.__gamma[idx_start:idx_stop], 
                    self.__eps[idx_start:idx_stop]
                )
                '''

                # create the thread
                t = t = threading.Thread( 
                    target = self.__s_thread,
                    args = (
                        i, 
                        self.__C[idx_start:idx_stop], 
                        self.__gamma[idx_start:idx_stop], 
                        self.__eps[idx_start:idx_stop],
                        n_cross_val,
                        verbose
                        ),
                    daemon = True
                )

                # append the thread 
                thread_list.append( t )
            
            print( " [multithread_grid_Search] ", f"=== ITERATION {iter} launching {n_thread} threads for {self.__n_comb} combination (max is {self.__max_comb})")

            # start the threads
            for i in range( 0, n_thread ):
                thread_list[i].start( )

            # wait for the thread to end
            for i in range( 0, n_thread ):
                thread_list[i].join( )
            thread_list.clear( )

            # replace the constants and reset the search
            temp_c = np.zeros( (n_thread,) )
            temp_g = np.zeros( (n_thread,) )
            temp_e = np.zeros( (n_thread,) )
            for i in range( 0, n_thread ):
                temp_c[i] = self.__return_space[i].best_params_['C']
                temp_g[i] = self.__return_space[i].best_params_['gamma']
                temp_e[i] = self.__return_space[i].best_params_['epsilon']
            '''
            print( "temp_c : \n", temp_c )
            print( "temp_g : \n", temp_g )
            print( "temp_e : \n", temp_e )
            print( "temp_c : \n", temp_c.shape )
            print( "temp_g : \n", temp_g.shape )
            print( "temp_e : \n", temp_e.shape )
            print( "temp_c : \n", np.unique( temp_c, axis=0 ) )
            print( "temp_g : \n", np.unique( temp_g, axis=0 ) )
            print( "temp_e : \n", np.unique( temp_e, axis=0 ) )
            '''
            temp_c = np.unique( temp_c )
            temp_g = np.unique( temp_g )
            temp_e = np.unique( temp_e )

            '''
            print( "temp_c : \n", temp_c )
            print( "temp_g : \n", temp_g )
            print( "temp_e : \n", temp_e )
            print( "temp_c : \n", temp_c.shape )
            print( "temp_g : \n", temp_g.shape )
            print( "temp_e : \n", temp_e.shape )
            '''

            # reset the search settings
            self.set_search( temp_c, temp_g, temp_e, self.__max_comb )

            # check for stop condition
            if self.__n_comb == 1:
                # found "the best" combination
                self.params = { 'C' : temp_c[0] , 'gamma' : temp_g[0], 'epsilon' : temp_e[0] }
                
                # the cycle ends
                break

            else:
                iter = iter + 1



class multiout_grid_search:
    ''' execute one grid search for output column, in parallel
        using Python threads.
    
    '''

    def __init__( self ):
        ''' costructor: init the class
        
        '''
        self.mgs_list = []
        self.params = []

        self.__thread_list = []
        self.__X = []
        self.__y = []
    

    def __s_thread( self, idx, C, gamma, epsilon, n_comb ):
        ''' model search for one coordinate
        
        args:
            col: the column to use in the thread
        '''

        print( " [multiout_grid_search] ", f"(thread {idx+1}) START" )

        ms = multithread_grid_search( )
        ms.set_search( C, gamma, epsilon, max_comb_per_thread=n_comb )
        ms.search_model( self.__X, self.__y[:, idx], verbose=True )

        print( " [multiout_grid_search] ", f"(thread {idx+1}) SAVING MODEL" )

        self.params[idx]['C'] = ms.params['C']
        self.params[idx]['gamma'] = ms.params['gamma']
        self.params[idx]['epsilon'] = ms.params['epsilon']

        print( " [multiout_grid_search] ", f"(thread {idx+1}) STOP" )
    
    
    def search( self, X, y, C, gamma, epsilon, n_comb ):
        ''' for each y column to predict, make a parameter search
        
        '''

        # make the sets globally available inside the class
        self.__X = X
        self.__y = y

        # create the thread objects
        for i in range( 0, y.shape[1] ):
            t = threading.Thread(
                target = self.__s_thread,
                args = ( i, C, gamma, epsilon, n_comb ),
                daemon = True
            )

            self.__thread_list.append( t )
            self.mgs_list.append( None )
            self.params.append( {'C':0.0, 'gamma':0.0, 'epsilon':0.0} )
        
        # start the threads
        for t in self.__thread_list:
            t.start( )
        
        # wait until completion
        for t in self.__thread_list:
            t.join( )


class multithread_SVR_learning:
    ''' simple multithread implementation of the training phase
    
    '''

    def __init__(self):
        ''' constructor
        
        '''
        
        # list of threads for the training
        self.__thread_list = []

        # list of learning machines from the threads
        self.__lm_list = []

        # data sets
        self.__X = []
        self.__Y = []



    def __s_thread( self, idx, params ):
        '''perform the SVR training
        
        '''

        # inner timer of the thread
        tmu = time_utils( )
        print( " [multithread_SVR_learning] ", f"(thread {idx+1}) starting the SVR trainig phase ..." )

        # training
        tmu.start( )
        lm = svm.SVR( 
            kernel = 'rbf',
            C = params['C'],
            gamma = params['gamma'],
            epsilon = params['epsilon'] 
        ).fit( self.__X, self.__Y[:, idx] )
        tmu.stop( print_val=False )

        print( " [multithread_SVR_learning] ", f"(thread {idx+1}) training completed in {tmu.value}s" )

        # save the results
        #        RACE CONDITION?????????????
        self.__lm_list[idx] = lm


    def train( self, X, y, params ):
        ''' create a thread for each training
        
        Args:
            X (np.array)
            y (np.array)
            params (list of dicts{'C', 'gamma', 'epsilon'})
        
        Returns:
            a list of learners
        '''
        tmu = time_utils( )

        self.__X = X
        self.__Y = y
        
        # setup the threads
        for i in range( 0, y.shape[1] ):
            t = threading.Thread(
                target = self.__s_thread,
                args=( i, params[i] ),
                daemon=True
            )

            self.__thread_list.append( t )

            # prepare data allocation
            self.__lm_list.append( svm.SVR( ) )
        
        # start the threads
        print( " [multithread_SVR_learning] ", f"starting {y.shape[1]} threads ..." )
        tmu.start( )
        for th in self.__thread_list:
            th.start( )
        
        # wait the threads
        print( " [multithread_SVR_learning] waiting for threads ..." )
        for th in self.__thread_list:
            th.join( )
        tmu.stop( print_val=False )
        
        print( f" [multithread_SVR_learning] end of the training process in {tmu.value}s" )

        return self.__lm_list
