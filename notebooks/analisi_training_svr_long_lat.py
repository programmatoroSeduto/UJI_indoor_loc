# -*- coding: utf-8 -*-
"""Analisi Training SVR long lat.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FB_QU-VDX7leok9ASW8TGoYHAuXgb9o2

# Analisi ultimo training SVR

## Librerie
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# sciKitLearn
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import sklearn.metrics as metrics

# per il salvataggio del modello su file
import pickle as pk

# percorsi
path = '/content/drive/MyDrive/UJI_indoor_loc/prev/'

# Commented out IPython magic to ensure Python compatibility.
# %pylab inline
mpl.rcParams['figure.figsize'] = (15, 7)

"""## Caricamento dati precedenti del training"""

# learning machines
LM_long = None
LM_lat = None

with open( path + 'LM_long_data.sav', 'rb' ) as fil:
  LM_long = pk.load( fil )

with open( path + 'LM_lat_data.sav', 'rb' ) as fil:
  LM_lat = pk.load( fil )

## scaler
scaler = None
with open( path + 'input_scaler.sav', 'rb' ) as fil:
  scaler = pk.load( fil )

# training set e test set
Ds_tr = pd.read_csv( path + 'myTrainingSet.csv' ).to_numpy( )
Ds_tt = pd.read_csv( path + 'myTestSet.csv' ).to_numpy( )

X_tr = scaler.transform( Ds_tr[:, 0:520] )
y_tr = Ds_tr[:, [520, 521]]
X_tt = scaler.transform( Ds_tt[:, 0:520] )
y_tt = Ds_tt[:, [520, 521]]

print( "Training Set Samples: ", Ds_tr.shape[0] )
print( "Validation Set Samples: ", Ds_tt.shape[0] )

"""## Un test sui coefficienti

Tutti gli alpha sono minori di C?
[questo](https://www.notion.so/ML2-formulazione-di-SVR-9436a407f9ea48578fa6ff3fff98c07b#37780c88c18e44eb9951805ecf2ee4dd) aiuta a capire perchè questi coefficienti possono anche venire negativi oltre che positivi. 
"""

# print( LM_long.C )
print( " lat: ", LM_lat )
print( " long: ", LM_long )

figt, axt = plt.subplots( nrows=2 )


figt.suptitle( "Alpha Coefficients of SVR" )

axt[0].grid( True )
axt[0].plot( [-250, X_tr.shape[0]*1.05], [LM_long.C, LM_long.C], '--y' )
axt[0].plot( [-250, X_tr.shape[0]*1.05], [-LM_long.C, -LM_long.C], '--y' )
axt[0].set_title( f"Longitude ({LM_long.dual_coef_.shape[1]} samples out of {X_tr.shape[0]} trainig samples -- C={LM_lat.C} eps={LM_lat.epsilon})" )
axt[0].plot( np.array( range( 0, LM_long.dual_coef_.shape[1] ) ), LM_long.dual_coef_[0, :], 'xb' )

axt[1].grid( True )
axt[1].plot( [-250, X_tr.shape[0]*1.05], [LM_lat.C, LM_lat.C], '--y' )
axt[1].plot( [-250, X_tr.shape[0]*1.05], [-LM_lat.C, -LM_lat.C], '--y' )
axt[1].set_title( f"Latitude ({LM_lat.dual_coef_.shape[1]} samples out of {X_tr.shape[0]} trainig samples -- C={LM_long.C} eps={LM_long.epsilon})" )
axt[1].plot( np.array( range( 0, LM_lat.dual_coef_.shape[1] ) ), LM_lat.dual_coef_[0, :], 'xb' )

plt.show( )

"""## Visualizzazione training set"""

fig1, ax1 = plt.subplots( )

# Ds_tr = pd.read_csv( save_path + 'myTrainingSet.csv' ).to_numpy( )
# print( "cols of Ds_tr : ", Ds_tr.shape[1] )

ax1.grid( True )
ax1.plot( Ds_tr[:, 520], Ds_tr[:, 521], 'x' ) # longitudine - latitudine

plt.show( )

"""## Predizione"""

ym_long = LM_long.predict( X_tt )
ym_lat = LM_lat.predict( X_tt )

"""## Confronto diretto tra i due output"""

fig2, ax2 = plt.subplots( ncols = 2 )

fig2.suptitle( "(test set) Comparison original map Vs.machine output" )

# il primo plot mostra il test set originale
ax2[0].set_title( "Real output" )
ax2[0].grid( True )
ax2[0].plot( y_tt[:, 0], y_tt[:, 1], 'xr' )
ax2[0].set_xlabel( "longitude" )
ax2[0].set_ylabel( "latitude" )

# il secondo plot mostra invece l'output della macchina
ax2[1].set_title( "Machine output" )
ax2[1].grid( True )
ax2[1].plot( ym_long, ym_lat, 'ob' )
ax2[1].set_xlabel( "longitude" )
ax2[1].set_ylabel( "latitude" )

plt.show( )

"""## Scatter plot per SVR"""

fig3, ax3 = plt.subplots( ncols = 2 )

sc_minmax = list() 
sc_minmax.append( [np.min(y_tt[:, 0]), np.max(y_tt[:, 0])] )
sc_minmax.append( [np.min(y_tt[:, 1]), np.max(y_tt[:, 1])] )
print( sc_minmax )

fig2.suptitle( "Scatter Plots for Longitude and Latitude" )

# scatter plot per la longitudine
ax3[0].set_title( "Longitude" )
ax3[0].grid( True )
ax3[0].plot( sc_minmax[0], sc_minmax[0], '--b' )
ax3[0].plot( y_tt[:, 0], ym_long, 'xr' )
ax3[0].set_xlabel( "real longitude" )
ax3[0].set_ylabel( "predicted longitude" )

# scatter plot per la latitudine
ax3[1].set_title( "Latitude" )
ax3[1].grid( True )
ax3[1].plot( sc_minmax[1], sc_minmax[1], '--b' )
ax3[1].plot( y_tt[:, 1], ym_lat, 'xr' )
ax3[1].set_xlabel( "real latitude" )
ax3[1].set_ylabel( "predicted latitude" )

plt.show( )

"""## Altri tipi di visualizzazione

- un istogramma indicativo della copertura di segnale. Per ogni ennupla, contare quanti WAP vengono rilevati, e fare un istogramma che riporti questi valori. Il grafico mostra più o meno che copertura c'è per ogni ennupla, e mette in risalto più o meno la percentuale di casi con una copertura ridotta (già riportato nell'articolo)

- un indicatore di qualità del learning. Per ogni sample, date la posizione reale e quella prevista, calcolare la distanza euclidea tra le due posizioni; fissata una threshold, calcolare poi la percentuale di samples predetti che rientrano effettivamente nella tolleranza, e quella dei samples fuori tolleranza. Che errore massimo possamo accettare in un environment indoor?

# Visualizzazione dataset con indicazione di copertura

è possibile definire la copertura di segnale in diversi modi:

- il numero di RSSI il cui valore non è 100dB
- la media dei valori RSSI individuati con successo (prendi solo quelli con valore diverso da 100dB)

Per l'analisi della copertura si fissa una threshold e si dividono i samples in tre gruppi:

- (x rossa) copertura pessima: sotto la threshold
- (cerchio arancione) copertura media: nell'intorno superiore della threshold (fissa una tolleranza)
- (punto pieno verde) copertura eccellente: sopra la threshold più tolleranza

## Specifiche per le visualizzazioni

- si applicano al training set o al test set o a entrambi?
- si applicano sui dati o sulle prediction? 
- che edificio visualizzare?

## Coverage con numero di WAP

Definiamo copertura il numero di WAP rilevati che hanno un valore oltre una certa soglia. 

Motivazioni di questa analisi:

- nelle varie zone del complesso il numero di WAP rilevati supera una certa soglia? ci vogliono almeno 3 WAP per poter eseguire una localizzazione
- supponendo che fenomeni quali il riflesso delle onde elettromagnetiche generi un valore di RSSI non più alto di una certa soglia, quante posizioni hanno dei WAP almeno ad una certa potenza?
- c'è una connessione tra la pessima copertura in un certo punto e la correttezza della classificazione? Mi aspetto che il classificatore possa essere tratto in inganno specialmente in punti in cui ci sono pochi WAP rilevati oltre una certa soglia
- quando il segnale è sotto una certa soglia, non si può essere certi della posizione precisa perchè il fingerprint nel punto con poca copertura (in termini di soglia minima) potrebbe assomigliare ad un altro fingerprint in un'altra zona, quindi c'è ambiguità. E in questi casi, la learning machine è portata a sbagliare più facilmente. Questo si somma al problema che RSSI non è esattamente proporzionale alla distanza, ma quasi, e quel "quasi" genera un'ambiguità sul segnale.

### La classe wap_coverage

Prima versione: implementa l'analisi descritta sopra.
"""

# la classe contiene le ennuple classificate in 3 vettori diversi
# .threshold : (int) minimo numero di WAP
# .tol : (float >= 0)  tolleranza attorno alla threshold
# .min_db : (float >= 0) valore minimo per WAP
# le tre classi: (liste di tuple) .low_coverage, .md_coverage, .good_coverage
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
        # print( "MID -- ", n_cover, " in [", ( self.threshold - self.tol ), ", ", ( self.threshold + self.tol ), ")" )
        self.mid_coverage.append( [long[i], lat[i]] )
      else:
        # print( "HIGH -- ", n_cover, " > ", ( self.threshold + self.tol ) )
        self.good_coverage.append( [long[i], lat[i]] )
      
    self.low_coverage = np.array( self.low_coverage )
    self.mid_coverage = np.array( self.mid_coverage )
    self.good_coverage = np.array( self.good_coverage )

"""### Plot della coverage sul training set"""

# esegui la classificazione in base alla coverage
'''
3 è il numero minimo di WAP per la localizzazione negli altri metodi 
la zona low coverage comprende tutti i punti in cui vengono rilevati meno di 3 WAP
nessun minimo di potenza considerato (possono esserci medie molto basse di potenza, rischio di rumore)
'''
wapc = wap_coverage( thresh=6, tol=3, min_db=np.Inf )

'''
questo dimostra che la stragrande maggioranza dei punti rileva un numero di WAP tra 3 e 23
niente soglia minima
'''
# wapc = wap_coverage( thresh = 13, tol = 10, min_db = np.Inf )

'''
fissiamo ora un minimo per la potenza RSSI (può essere lecito nell'applicazione di interesse, ad esempio per escludere il rumore)
questa sequenza di tre grafici mostra quanto sia difficile avere dei WAP con soglia minima di potenza
confermando l'osservazione del grafico dal paper sui valori tipici di RSSI 
'''
# wapc = wap_coverage( thresh = 13, tol = 10, min_db = 90 )
# wapc = wap_coverage( thresh = 13, tol = 10, min_db = 80 )
# wapc = wap_coverage( thresh = 13, tol = 10, min_db = 70 )
# wapc = wap_coverage( thresh = 13, tol = 10, min_db = 60 )

# print( Ds_tt[50:60, 0:520] )
wapc.analyze_data( Ds_tr[:, 0:520], Ds_tr[:, 520], Ds_tr[:, 521] )

print( "good coverage size: ", len(wapc.good_coverage) )
print( "mid coverage size: ", len(wapc.mid_coverage) )
print( "low coverage size: ", len(wapc.low_coverage) )

fig_cov_n, ax_cov_n = plt.subplots( )
fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- mid N in [{wapc.threshold-wapc.tol}, {wapc.threshold+wapc.tol}] -- threshold N={wapc.threshold}  -- min dB={wapc.min_db}" )

ax_cov_n.grid( True )
if wapc.good_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
if wapc.mid_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
if wapc.low_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
ax_cov_n.legend( )

plt.show( )

"""### Plot della coverage sul test set

usando i samples reali
"""

# esegui la classificazione in base alla coverage
wapc = wap_coverage( thresh=6, tol=3, min_db=90 )
wapc.analyze_data( Ds_tt[:, 0:520], Ds_tt[:, 520], Ds_tt[:, 521] )

print( "good coverage size: ", len(wapc.good_coverage) )
print( "mid coverage size: ", len(wapc.mid_coverage) )
print( "low coverage size: ", len(wapc.low_coverage) )

fig_cov_n, ax_cov_n = plt.subplots( )
fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- {wapc.threshold - wapc.tol} to {wapc.threshold + wapc.tol} -- min dB={wapc.min_db}" )

ax_cov_n.grid( True )
if wapc.good_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
if wapc.mid_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
if wapc.low_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
ax_cov_n.legend( )

plt.show( )

"""### Plot della coverage usando però la prediction

Nota che l'algoritmo tende a dare rsultati imprecisi anche con punti con un'ampia copertura, per il semplice fatto che magari abbiamo fissato una soglia minima di valore RSSI troppo bassa. In ogni caso, il numero di WAP non dice nulla sulla quantità di segnale. 
"""

# esegui la classificazione in base alla coverage
wapc = wap_coverage( thresh=6, tol=3, min_db=90 )
wapc.analyze_data( Ds_tt[:, 0:520], Ds_tt[:, 520], Ds_tt[:, 521] )

print( "good coverage size: ", len(wapc.good_coverage) )
print( "mid coverage size: ", len(wapc.mid_coverage) )
print( "low coverage size: ", len(wapc.low_coverage) )

fig_cov_n, ax_cov_n = plt.subplots( ncols=2 )
fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- threshold N={wapc.threshold} -- min dB={wapc.min_db}" )

ax_cov_n[0].grid( True )
if wapc.good_coverage.shape[0] > 0 : 
  ax_cov_n[0].plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
if wapc.mid_coverage.shape[0] > 0 : 
  ax_cov_n[0].plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
if wapc.low_coverage.shape[0] > 0 : 
  ax_cov_n[0].plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
ax_cov_n[0].legend( )

ax_cov_n[1].grid( True )
wapc.analyze_data( Ds_tt[:, 0:520], ym_long, ym_lat )
if wapc.good_coverage.shape[0] > 0 : 
  ax_cov_n[1].plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
if wapc.mid_coverage.shape[0] > 0 : 
  ax_cov_n[1].plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
if wapc.low_coverage.shape[0] > 0 : 
  ax_cov_n[1].plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
ax_cov_n[1].legend( )

plt.show( )

"""## Coverage con media valori RSSI tra quelli rilevati

Il metodo è simile a prima, solo che stavolta l'analizzatore misura la media degli RSSI individuati.

### Implementazione classe wap_coverage_mean
"""

# la classe contiene le ennuple classificate in 3 vettori diversi
# .threshold : (float) valore accettabile di media algebrica di segnale RSSI
# .tol : (float >= 0)  tolleranza attorno alla threshold
# .min_db : (float >= 0) valore minimo per WAP
# le tre classi: (matrice) .low_coverage, .md_coverage, .good_coverage
class wap_coverage:
  
  # costruttore: threshold e numero minimo di decibel
  def __init__( self, thresh=80, tol=0, min_db=90 ):
    # settings
    self.threshold = -float(thresh)
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
      avg = 0.0
      for rssi in x:
        if ( rssi <= 0 ) and ( rssi >= self.min_db ):
          # print( rssi, ">=", self.min_db )
          n_cover = n_cover + 1
          avg = avg + rssi
      if n_cover > 0:
        avg = avg / float( n_cover )
      else:
        avg = -np.Inf
      # avg = avg / float( Xs.shape[1] )
      # print( f"{i} -- {avg}" )
      
      # classifica l'output
      if avg < ( self.threshold - self.tol ):
        # print( "LOW -- ", n_cover, " < ", ( self.threshold - self.tol ) )
        self.low_coverage.append( [long[i], lat[i]] )
      elif (avg >= ( self.threshold - self.tol )) and (avg < ( self.threshold + self.tol )):
        # print( "MID -- ", n_cover, " in [", ( self.threshold - self.tol ), ", ", ( self.threshold + self.tol ), ")" )
        self.mid_coverage.append( [long[i], lat[i]] )
      else:
        # print( "HIGH -- ", n_cover, " > ", ( self.threshold + self.tol ) )
        self.good_coverage.append( [long[i], lat[i]] )
      
    self.low_coverage = np.array( self.low_coverage )
    self.mid_coverage = np.array( self.mid_coverage )
    self.good_coverage = np.array( self.good_coverage )

"""### Plot coverage sul training set"""

# esegui la classificazione in base alla coverage
wapc = wap_coverage( thresh=70, tol=10, min_db=90 )
# print( Ds_tt[50:60, 0:520] )
wapc.analyze_data( Ds_tr[:, 0:520], Ds_tr[:, 520], Ds_tr[:, 521] )

print( "good coverage size: ", len(wapc.good_coverage) )
print( "mid coverage size: ", len(wapc.mid_coverage) )
print( "low coverage size: ", len(wapc.low_coverage) )

fig_cov_n, ax_cov_n = plt.subplots( )
fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- threshold rssi={wapc.threshold}dB -- min={wapc.min_db}dB" )

ax_cov_n.grid( True )
if wapc.good_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
if wapc.mid_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
if wapc.low_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
ax_cov_n.legend( )

plt.show( )

"""### Avg coverage e prediction"""

# esegui la classificazione in base alla coverage
wapc = wap_coverage( thresh=13, tol=10, min_db=90 )
wapc.analyze_data( Ds_tt[:, 0:520], Ds_tt[:, 520], Ds_tt[:, 521] )

print( "good coverage size: ", len(wapc.good_coverage) )
print( "mid coverage size: ", len(wapc.mid_coverage) )
print( "low coverage size: ", len(wapc.low_coverage) )

fig_cov_n, ax_cov_n = plt.subplots( ncols=2 )
fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- threshold N={wapc.threshold} -- min dB={wapc.min_db}" )

ax_cov_n[0].grid( True )
if wapc.good_coverage.shape[0] > 0 : 
  ax_cov_n[0].plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
if wapc.mid_coverage.shape[0] > 0 : 
  ax_cov_n[0].plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
if wapc.low_coverage.shape[0] > 0 : 
  ax_cov_n[0].plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
ax_cov_n[0].legend( )

ax_cov_n[1].grid( True )
wapc.analyze_data( Ds_tt[:, 0:520], ym_long, ym_lat )
if wapc.good_coverage.shape[0] > 0 : 
  ax_cov_n[1].plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
if wapc.mid_coverage.shape[0] > 0 : 
  ax_cov_n[1].plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
if wapc.low_coverage.shape[0] > 0 : 
  ax_cov_n[1].plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
ax_cov_n[1].legend( )

plt.show( )

"""# Valutazione grafica della precisione dell'algoritmo

## Classe per la misura della precisione

intesa come distanza euclidea tra il punto reale e il punto predetto.
"""

class loc_precision:
  
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

"""## Qualità del learning -- analisi grafica

Interessante notare che le previsioni tendono ad essere più vicine al centro del grafico, perchè quella è la zona "centrale" se consideriamo la distribuzione di WAP nella zona. 
"""

prec = loc_precision( threshold=50, max_worst_cases=500 )
prec.analyze_data( Ds_tt[:, 520], Ds_tt[:, 521], ym_long, ym_lat )

# print( prec.bad )

# figq, axq = plt.subplots( ncols=2 )
figq, axq = plt.subplots( )
figq.suptitle( f"Precision of the learner -- threshold={prec.threshold}" )

'''
axq[0].grid( True )
axq[0].set_title( "real data" )
if prec.good.shape[0] > 0:
  axq[0].plot( prec.good[:, 0], prec.good[:, 1], '.g', label=f"within the threshold ({prec.good.shape[0]} samples)" )
if prec.bad.shape[0] > 0:
  axq[0].plot( prec.bad[:, 0], prec.bad[:, 1], 'xr', label=f"out of threshold ({prec.bad.shape[0]} samples)" )
axq[0].legend( )

axq[1].grid( True )
axq[1].set_title( "LM output" )
if prec.good_f.shape[0] > 0:
  axq[1].plot( prec.good_f[:, 0], prec.good_f[:, 1], '.g', label=f"within the threshold ({prec.good.shape[0]} samples)" )
if prec.bad_f.shape[0] > 0:
  axq[1].plot( prec.bad_f[:, 0], prec.bad_f[:, 1], 'xr', label=f"out of threshold ({prec.bad.shape[0]} samples)" )
axq[1].legend( )
'''

axq.grid( True )
axq.set_title( "LM output" )
if prec.good_f.shape[0] > 0:
  axq.plot( prec.good_f[:, 0], prec.good_f[:, 1], '.g', label=f"in the threshold ({prec.good.shape[0]} samples, max {prec.threshold}m, min distance {np.round(prec.min_d, 1)}m)" )
  pass
if prec.bad_f.shape[0] > 0:
  axq.plot( prec.bad_f[:, 0], prec.bad_f[:, 1], 'xr', label=f"out of threshold ({prec.bad.shape[0]} samples, max distance {np.round(prec.max_d, 1)}m)" )
  pass

# print casi pessimi
'''
for i in range(0, prec.max_d_y.shape[0]):
  axq.plot( prec.max_d_x[i, :], prec.max_d_y[i, :], '--y' )

axq.plot( prec.max_d_x[:, 0], prec.max_d_y[:, 0], 'ob' )
axq.plot( prec.max_d_x[:, 1], prec.max_d_y[:, 1], 'or' )
'''
axq.legend( )

plt.show( )

"""# Precisione Vs. Numero di WAP rilevati"""

# esegui la classificazione in base alla coverage
prec = loc_precision( threshold=10, max_worst_cases=50 )
prec.analyze_data( Ds_tt[:, 520], Ds_tt[:, 521], ym_long, ym_lat )
wapc = wap_coverage( thresh=7, tol=0, min_db=75 )
wapc.analyze_data( Ds_tt[:, 0:520], Ds_tt[:, 520], Ds_tt[:, 521] )

# print( "good coverage size: ", len(wapc.good_coverage) )
# print( "mid coverage size: ", len(wapc.mid_coverage) )
# print( "low coverage size: ", len(wapc.low_coverage) )

fig_cov_n, ax_cov_n = plt.subplots( ncols=2 )
fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- threshold N={wapc.threshold} -- min dB={wapc.min_db}" )

ax_cov_n[0].grid( True )
ax_cov_n[0].set_title( "LM output" )
if prec.good_f.shape[0] > 0:
  ax_cov_n[0].plot( prec.good_f[:, 0], prec.good_f[:, 1], '.g', label=f"in the threshold ({prec.good.shape[0]} samples, max {prec.threshold}m, min distance {np.round(prec.min_d, 1)}m)" )
if prec.bad_f.shape[0] > 0:
  ax_cov_n[0].plot( prec.bad_f[:, 0], prec.bad_f[:, 1], 'xr', label=f"out of threshold ({prec.bad.shape[0]} samples, max distance {np.round(prec.max_d, 1)}m)" )

for i in range(0, prec.max_d_y.shape[0]):
  ax_cov_n[0].plot( prec.max_d_x[i, :], prec.max_d_y[i, :], '--y' )

ax_cov_n[0].plot( prec.max_d_x[:, 0], prec.max_d_y[:, 0], 'ob' )
ax_cov_n[0].plot( prec.max_d_x[:, 1], prec.max_d_y[:, 1], 'or' )
ax_cov_n[0].legend( )

ax_cov_n[1].grid( True )
wapc.analyze_data( Ds_tt[:, 0:520], ym_long, ym_lat )
if wapc.good_coverage.shape[0] > 0 : 
  ax_cov_n[1].plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
if wapc.mid_coverage.shape[0] > 0 : 
  ax_cov_n[1].plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
if wapc.low_coverage.shape[0] > 0 : 
  ax_cov_n[1].plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
ax_cov_n[1].legend( )

plt.show( )

# esegui la classificazione in base alla coverage
prec = loc_precision( threshold=10, max_worst_cases=50 )
prec.analyze_data( Ds_tt[:, 520], Ds_tt[:, 521], ym_long, ym_lat )
wapc = wap_coverage( thresh=7, tol=0, min_db=75 )
wapc.analyze_data( Ds_tt[:, 0:520], Ds_tt[:, 520], Ds_tt[:, 521] )

# print( "good coverage size: ", len(wapc.good_coverage) )
# print( "mid coverage size: ", len(wapc.mid_coverage) )
# print( "low coverage size: ", len(wapc.low_coverage) )

fig_cov_n, ax_cov_n = plt.subplots( )
# fig_cov_n.suptitle( f"Coverage analysis (num of WAP detected) -- threshold N={wapc.threshold} -- min dB={wapc.min_db}" )

ax_cov_n.grid( True )
ax_cov_n.set_title( "LM output" )
if prec.good_f.shape[0] > 0:
  ax_cov_n.plot( prec.good_f[:, 0], prec.good_f[:, 1], '.g', label=f"in the threshold ({prec.good.shape[0]} samples, max {prec.threshold}m, min distance {np.round(prec.min_d, 1)}m)" )
if prec.bad_f.shape[0] > 0:
  ax_cov_n.plot( prec.bad_f[:, 0], prec.bad_f[:, 1], 'xr', label=f"out of threshold ({prec.bad.shape[0]} samples, max distance {np.round(prec.max_d, 1)}m)" )
ax_cov_n.legend( )
'''
for i in range(0, prec.max_d_y.shape[0]):
  ax_cov_n[0].plot( prec.max_d_x[i, :], prec.max_d_y[i, :], '--y' )

ax_cov_n[0].plot( prec.max_d_x[:, 0], prec.max_d_y[:, 0], 'ob' )
ax_cov_n[0].plot( prec.max_d_x[:, 1], prec.max_d_y[:, 1], 'or' )
ax_cov_n[0].legend( )
'''
fig_cov_n.savefig( path + "LM_prec.png" )

fig_cov_n, ax_cov_n = plt.subplots( )

ax_cov_n.grid( True )
wapc.analyze_data( Ds_tt[:, 0:520], ym_long, ym_lat )
if wapc.good_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.good_coverage[:, 0], wapc.good_coverage[:, 1], '.g', label=f"good coverage ({wapc.good_coverage.shape[0]} samples)" )
if wapc.mid_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.mid_coverage[:, 0], wapc.mid_coverage[:, 1], 'oy', label=f"mid coverage ({wapc.mid_coverage.shape[0]} samples)", mfc='none' )
if wapc.low_coverage.shape[0] > 0 : 
  ax_cov_n.plot( wapc.low_coverage[:, 0], wapc.low_coverage[:, 1], 'xr', label=f"poor coverage ({wapc.low_coverage.shape[0]} samples)" )
ax_cov_n.legend( )

fig_cov_n.savefig( path + "LM_cov.png" )

"""## Per visualizzazioni più avanzate

Per queste visualizzazioni sarebbe meglio usare una color map piuttosto che un semplice plot 2D. La colormap renderebbe meglio l'idea.

# Visualizzazione dataset con precisione media

Il concetto è lo stesso della coverage; stavolta però siamo interessati a stimare quanto è preciso il classificatore nell'individuare la posizione dell'utente. Inoltre, questo grafico si applica solo al test set

La metrica è la semplice distanza euclidea tra il punto reale e quello individuato col machine learning. Si fissa una certa tolleranza (che indica quando la precisione è accettabile) e si divide ogni sample in 3 classi:

- precisione pessima : ampiamente al di sopra della tolleranza
- precisione accettabile : il risultato è nell'intorno della tolleranza, al limite
- precisione eccellente : la distanza è sotto la threshold, preferibilmente nulla

Per questo genere di applicazione sarebbe preferibile applicare una threshold di massimo 4m con tolleranza 0.5m, che comunque è tanto nelle applicazioni. 

Il grafico dovrebbe riportare anche qualche caso notevole di distanza, ad esempio i casi peggiori come una linea tratteggiata che va dal punto reale al punto approssimato.
"""

# TODO implementazione con grafico