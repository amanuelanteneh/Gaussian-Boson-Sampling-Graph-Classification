# Classifcation of graph-structred data using Gaussian boson sampling with thershold detectors
Repo for the code used to generate the data in the paper I co-authored *A Quantum Graph Kernel using Gaussian Boson Sampling with Threshold Detectors* forthcoming in the 
jounral Physical Review A.

The basic idea of the quantum machine learning algorithm is to encode the adjacancey matrix of a graph into a Gaussian boson sampler that measures photon detection events 
using thereshold detectors which click when they detect any photons as opposed to photon number resolving detectors which count exactly how many photons are detected.

After the matrix is encoded into the device we sample the device *S* times. Using those samples we construct a feature vector that contains useful information about the graph encoded.
In our case the features are related to the number of perfect mathings of all possible subgraphs of the encoded graph. 
The probabilities $p(\textbf{n})$ of detecting the detection events is

Folders are named after the model used to benchmark the new feature vectors. NN being the multi-layer perceptron model, GBS being the Gaussian boson sampling model, RW being the random walk kernel, SP being the shortest path kernel, SM being the subgraph matchin kernel and GS being the graphlet sampling kernel.