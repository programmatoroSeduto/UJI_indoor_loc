#! /usr/bin/python3

# main frameworks
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

# UJI files
from ujiProject import paths as uji_paths
from ujiProject.DataAnalysis import *
from ujiProject.DataPreparation import *
from ujiProject.SVRLearning import *

# per il salvataggio del modello su file
import pickle as pk

if __name__ == "__main__":
	print( "MAIN" )