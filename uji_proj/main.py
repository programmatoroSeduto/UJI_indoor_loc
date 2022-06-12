#! /usr/bin/python3

import os
import numpy as np
from ujiProject.make_visuals import make_visuals
from ujiProject.make_training import make_training
from ujiProject.time_utils import time_utils

if __name__ == "__main__":
	# set max priority
	os.nice( 19 )

	# timer
	tmu = time_utils( )

	# training
	print( "training ..." )
	tmu.start( )
	make_training( )
	tmu.stop( False )
	print( f"training ... OK in {np.round(tmu.value, 1)}s" )

	print( "generating plots ..." )
	tmu.start( )
	make_visuals( )
	tmu.stop( False )
	print( f"generating plots ... OK in {np.round(tmu.value, 1)}s" )
	
	print( "=== DONE ===" )
