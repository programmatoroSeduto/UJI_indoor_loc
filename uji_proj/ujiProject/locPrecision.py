#! /usr/bin/python3

# importing libraries

import numpy as np 

## IMPLEMENTAZIONE SECONDA CLASSE -- LOC PRECISION -- VALUTAZIONE GRAFICA DELLA PRECISIONE DELL'ALGORITMO

class loc_precision:
    ''' Class devoted to the precision measurement 

    it is intended as the euclidean distance between the real point and the predicted one 
    '''
  
    # threshold in metri
    def __init__( self, threshold=5, max_worst_cases=50 ):
      # settings
      self.threshold = float(threshold)

      # i due gruppi
      self.good = []
      self.bad = []
      self.good_f = []
      self.bad_f = []

      # ricerca anche il massimo errore e i punti che lo hanno generato
      self.max_d = 0.0
      self.min_d = np.Inf
      self.max_d_x = [ ]
      self.max_d_y = [ ]
      self.worst_sample_idx = [ ]
      self.max_worst_samples = max_worst_cases
    
    # analisi dei dati
    def analyze_data( self, real_x, real_y, fore_x, fore_y ):
      '''
      la funzione classifica i punti in base alle distanze
      '''
      self.good = []
      self.bad = []
      self.max_d = 0.0
      self.min_d = np.Inf
      self.max_d_x = [ ]
      self.max_d_y = [ ]
      self.worst_sample_idx = [ ]

      n_list = 0

      # print( "max worst: ", self.max_worst_samples )

      for i, rx, fx, ry, fy in zip( range(0, real_x.shape[0]), real_x, fore_x,  real_y, fore_y ) :

        # distanza tra i due sample
        d = np.sqrt( ( rx - fx ) ** 2 + ( ry - fy ) ** 2 )

        if d < self.min_d:
          self.min_d = d

        # classificazione in base alla distanza
        if d <= self.threshold:
          self.good.append( [rx, ry] )
          self.good_f.append( [fx, fy] )
        else:
          self.bad.append( [rx, ry] )
          self.bad_f.append( [fx, fy] )
          if self.max_d < d:
            self.max_d = d
            self.max_d_x.append( [ rx, fx ] )
            self.max_d_y.append( [ ry, fy ] )
            self.worst_sample_idx.append( i )
            n_list = n_list + 1
          
          if n_list > self.max_worst_samples:
            self.max_d_x.pop( n_list-1 )
            self.max_d_y.pop( n_list-1 )
            self.worst_sample_idx.pop( n_list-1 )
            n_list = n_list - 1
        
      self.good = np.array( self.good )
      self.bad = np.array( self.bad )
      self.good_f = np.array( self.good_f )
      self.bad_f = np.array( self.bad_f )
      self.max_d_x = np.array( self.max_d_x )
      self.max_d_y = np.array( self.max_d_y )
      self.worst_sample_idx = np.array( self.worst_sample_idx )
