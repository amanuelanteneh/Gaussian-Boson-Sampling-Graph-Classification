# Classifcation of graph-structred data using Gaussian boson sampling with threshold detectors
Repo for the code used to generate the data in the paper I co-authored *A Quantum Machine Learning Algorithm for Graph Classification using Gaussian Boson Sampling* forthcoming in the 
journal Physical Review A.

The basic idea of the quantum machine learning algorithm is to encode the adjacancey matrix of a graph into a Gaussian boson sampler that measures photon detection events 
using thereshold detectors which click when they detect any photons as opposed to photon number resolving detectors which count exactly how many photons are detected.

After the matrix is encoded into the device we sample the device *S* times. Using those samples we construct a feature vector that contains useful information about the graph encoded.
In our case the features are related to the number of perfect mathings of all possible subgraphs of the encoded graph. 
The probability of a detection event, dentoed $p(\textbf{n})$, is proportinal the Hafnian of a submatrix of the 
adjacency matrix of the graph encoded into the quantum sampling device. The exact equation for the $p(\textbf{n})$ is

$$ p(\textbf{n}) = \frac{1}{\sqrt{\textrm{det}(Q)}}\frac{|\textrm{Haf}(A_{\textbf{n}})|^2}{\textbf{n}!} $$

where

$$ Q = (\mathbb{I}_{2M} - X\tilde{A})^{-1}, \quad X =  \begin{bmatrix}0 & \mathbb{I}_M\\\ \mathbb{I}_M & 0\end{bmatrix}, $$
   
$\textbf{n}! = n_1!\times...\times n_M!$ and $\textrm{Haf()}$ denoting the Hafnian of the matrix.

Folders are named after the model used to benchmark the new feature vectors. NN being the multi-layer perceptron model, GBS being the Gaussian boson sampling model, RW being the random walk kernel, SP being the shortest path kernel, SM being the subgraph matchin kernel and GS being the graphlet sampling kernel.