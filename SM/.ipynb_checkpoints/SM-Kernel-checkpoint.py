#import standard ML libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#import graph libraries library
import networkx as nx
from grakel.datasets import fetch_dataset
from grakel import graph_from_networkx
from grakel import GraphKernel, Graph

#import quantum optics librarys
from thewalrus.samples import torontonian_sample_state
from thewalrus.quantum import gen_Qmat_from_graph, Covmat
from strawberryfields.apps import sample

#import standard py libraries
from tabulate import tabulate
import sys

import warnings  #to ignore complex cast warning
warnings.filterwarnings('ignore')

def preprocessDatasets(datasetName, removeIsolates=True, keepLargestCC=False):

  graphData = fetch_dataset(datasetName, verbose=False, as_graphs=True) # get dataset 
  graphs, y = graphData.data, graphData.target # split into X and y set
  
  filteredGraphs = []
  filteredLabels = []
  for i in range(0, len(graphs)): 
        
    G = nx.from_numpy_array( graphs[i].get_adjacency_matrix() ) 
        
    if (removeIsolates==True): # remove isolated nodes if opted for
      G.remove_nodes_from(list(nx.isolates(G)))  
        
    if (keepLargestCC == True): # extract largest connected component if opted for
      largestCC = max(nx.connected_components(G), key=len)
      G = G.subgraph(largestCC).copy()
        
    #graphs[i] = G #store the GraKel version of the graph for classical kernel application
        
    # filter datasets to have 6 <= x <= 25 nodes
    if ( graphs[i].get_adjacency_matrix().shape[0] >= 6 and  graphs[i].get_adjacency_matrix().shape[0] <= 25):
      filteredGraphs.append(graphs[i]) 
      filteredLabels.append(y[i])

  # return a tuple: index 0 is the preprocessed graphs, index 1 is their labels
  return(filteredGraphs, filteredLabels)


#read in command line args
datasetIndex = int(sys.argv[1])

#Fetch and preprocess graph datasets
datasetNames = ["AIDS", "BZR_MD", "COX2_MD", "ENZYMES", "ER_MD", "IMDB-BINARY", "MUTAG", "NCI1", "PROTEINS", "PTC_FM", "DD", "COLLAB", "FINGERPRINT"]


X, y = preprocessDatasets(datasetNames[datasetIndex], False, False) 

table = [ [ "Data set", datasetNames[datasetIndex] ] ]
table.append( [ "Number of\n Graphs" ] ) #row 1
table.append( [ "Avg. Number\n of Vertices" ] )
table.append( [ "Avg. Number\n of Edges" ] )
table.append( [ "Number of\n Classes" ] )

avgNodes = sum([ g.get_adjacency_matrix().shape[0] for g in X ]) / len(X) 
avgEdges = sum([ len(g.get_edges())/2 for g in X ]) / len(X) 
numClasses = len(set(y)) 
table[1].append( len(X) )  
table[2].append(avgNodes)
table[3].append(avgEdges)
table[4].append(numClasses)

table = np.array(table).T.tolist()

print(tabulate(table, headers='firstrow', tablefmt='simple'), flush=True)
    

print("\nKernel: Subgraph Matching", "\nParameters: Default GraKel parameters", "\n", flush=True)

table = [ [ "Data set", datasetNames[datasetIndex] ] ] #reset table
table.append( ["RF (SE)"] ) #RF standard error column
table.append( [ "SVM (SE)" ] ) #SVM standard error column


# subgraph matching kernel
sm = {"name": 'subgraph_matching', 'kv': None, 'ke': None}


paramGridSVM = [{'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0 ,1e2, 1e3]}]
    
paramGridRF = [{'n_estimators': [10, 50, 100, 250], 
                'min_samples_leaf': [3, 4], 
                'max_depth': [5, 10, 50, 250], 
                'bootstrap': [False] }]

accuraciesSVM = []
accuraciesRF = []

for i in range(10): # 10 repeats of double cross validation
  seed = i
  kfold1 = KFold(n_splits=10, shuffle=True, random_state=None)
  classifierSVM = make_pipeline(
       GraphKernel(kernel=sm, normalize=True, verbose=True, random_state=seed),
       GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid=paramGridSVM,
                      scoring="accuracy", cv=kfold1))
  classifierRF = make_pipeline(
       GraphKernel(kernel=sm, normalize=True, verbose=True, random_state=seed),
       GridSearchCV(RandomForestClassifier(random_state=1, class_weight='balanced') , param_grid=paramGridRF,
                      scoring="accuracy", cv=kfold1))

  kfold2 = KFold(n_splits=10, shuffle=True, random_state=None)

  scoresSVM = cross_validate(classifierSVM, X=X, y=y, cv=kfold2, n_jobs=-1, return_train_score=False)
  scoresRF = cross_validate(classifierRF, X=X, y=y, cv=kfold2, n_jobs=-1, return_train_score=False)

  # Get best output of this fold
  accuraciesSVM.append(np.mean(scoresSVM['test_score']))
  accuraciesRF.append(np.mean(scoresRF['test_score']))

  # Get stats of this fold and print
  testMeanSVM = np.mean(scoresSVM['test_score'])
  testStdSVM = np.std(scoresSVM['test_score'])
    
  testMeanRF = np.mean(scoresRF['test_score'])
  testStdRF = np.std(scoresRF['test_score'])
      

resultsSVM = np.array(accuraciesSVM)
resultsRF = np.array(accuraciesRF)

table = [ [ "Data set", datasetNames[datasetIndex] ] ] #reset table
table.append( ["SVM"] ) #RF standard deviation column
table.append( [ "RF" ] ) #SVM standard deviation column
table[1].append( str( '%2.2f'% (100*np.mean(resultsSVM) ) ) + " \u00B1 " + str( '%2.2f'% (100*np.std(resultsSVM) ) ) )
table[2].append( str( '%2.2f'% (100*np.mean(resultsRF) ) ) + " \u00B1 " + str( '%2.2f'% (100*np.std(resultsRF) ) ) )
print(tabulate(table, headers='firstrow', tablefmt='simple'))