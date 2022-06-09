# UJI_indoor_loc

## How to create Py threads

- guide: [realPython](https://realpython.com/intro-to-python-threading/)
- reference from the official docs : [reference](https://docs.python.org/3/library/threading.html)

Py threads are *daemons* (detached threads) unless the main programm doesn't wait for their completion.

### Libraries

The main class thread:

```py
import threading
```

### Creating and running a simple thread

Just create the callack and then define the object thread. 

```py
import threading

# the "thread function"
def quellochevuoi_threadfunction( ... your args ... ):
	# really, everything you want inside it
	pass

# define the thread
t = threading.Thread( 
	target = ...your thread function ... ,
	args = ... a tuple of args ... ,
	daemon = ... default True ... ,
)
```

To run a thread use `.start()`, and `.join()` to wait it until end.

```py

# start the thread
t.start( )

# wait the thread until it doesn't end
t.join( )

```

### Creating many threads -- simple way

see this example, and in particular take into account the function `threading.append( ...Thread obj ... )`. 
The example just runs up to 4 threads giving to each of them a different integer index:

```py

import threading
import time

def my_thread( index, name ):
	print( "I'm thread no.", index, "; my name is ", name )

names = [ "arturo", "ermenegildo", "alfredo", "bernadette" ]
wait_time = 2

for i in range(1, 5):
	# create the thread number 'i'
	t = threading.Thread(
		target = my_thread,
		args = ( index, names[i] ),
		daemon = True # detached thread, exit when the program ends
	)
	
	# append the thread 
	threading.append( t )
	
	# start the thread
	t.start( )

# wait until the end of the (universe) time
time.sleep( wait_time )

```

### Return a value from a thread

just notice that `.join( )` doesn't have a return value. If you want a return value the best way is to pass a reference to some memory location by argument. 
For instance, an object passed by reference make this work. 


