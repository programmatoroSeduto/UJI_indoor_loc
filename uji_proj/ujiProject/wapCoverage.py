
import numpy as np 

class wap_coverage:
  
  # costruttore: threshold e numero minimo di decibel
  def __init__( self, thresh, tol=0, min_db=90 ):
    # settings
    self.threshold = thresh
    self.min_db = -float(min_db)
    self.tol = tol

    # i tre gruppi
    self.low_coverage = []
    self.mid_coverage = []
    self.good_coverage = []
  
  # analisi dei dati
  def analyze_data( self, Xs, long, lat ):
    '''
    input: Xs i 520 input, long e lat predetti (allineati a Xs)
    la funzione crea i tre vettori della coverage
    '''
    self.low_coverage = []
    self.mid_coverage = []
    self.good_coverage = []

    # print( "threshold: ", self.threshold, " num of WAPs detected" )
    # print( "min_db : ", self.min_db, "dB" )
    # print( "tol : ", self.tol, " num of WAPs" )
    for i, x in zip( range(0, Xs.shape[0]), Xs ) :
      # conta la copertura
      n_cover = 0
      for rssi in x:
        if ( rssi <= 0 ) and ( rssi >= self.min_db ):
          # print( rssi, ">=", self.min_db )
          n_cover = n_cover + 1
      # print( f"{i} -- {n_cover}" )
      
      # classifica l'output
      if n_cover < ( self.threshold - self.tol ):
        # print( "LOW -- ", n_cover, " < ", ( self.threshold - self.tol ) )
        self.low_coverage.append( [long[i], lat[i]] )

      elif (n_cover >= ( self.threshold - self.tol )) and (n_cover < ( self.threshold + self.tol )):
        #print( "MID -- ", n_cover, " in [", ( self.threshold - self.tol ), ", ", ( self.threshold + self.tol ), ")" )
        self.mid_coverage.append( [long[i], lat[i]] )

      else:
        # print( "HIGH -- ", n_cover, " > ", ( self.threshold + self.tol ) )
        self.good_coverage.append( [long[i], lat[i]] )
      
    self.low_coverage = np.array( self.low_coverage )
    self.mid_coverage = np.array( self.mid_coverage )
    self.good_coverage = np.array( self.good_coverage )
