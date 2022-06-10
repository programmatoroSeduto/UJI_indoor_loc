
import time as tm
import numpy as np


class time_utils:
    '''a simple chronometer
    
    '''

    def __init__( self ):
        ''' init the chronometer with a empty value
        
        '''

        self.__t0 = 0.0
        self.__t1 = 0.0
        self.__started = False
        self.value = 0.0
    

    def start( self ):
        ''' start the chronometer
        
        Note:
            if the chronometer is actually running, the 
            value will be overwritten.
            the value of the chronometer will be zero until
            the stop() command isn't called.
        '''

        self.__t0 = tm.time( )
        self.__t1 = 0.0
        self.__started = True
    

    def stop( self, print_val=True ):
        '''stop the chronometer, and in case print 
            a string telling the time.
        
        Returns:
            the delta value of the chronometer
        '''

        self.__t1 = tm.time( )
        self.value = self.__t1 - self.__t0

        if print_val:
            print( f"done in {np.round( self.value, 2 )}s" )

        return self.value

        