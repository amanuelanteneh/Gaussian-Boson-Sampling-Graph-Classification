# Classifcation of graph-structred data using Gaussian boson sampling with thershold detectors
Repo for the code used to generate the data in the paper I authored *A Quantum Graph Kernel using Gaussian Boson Sampling with Threshold Detectors* forthcoming in the 
jounral Physical Review A.

The basic idea of the quantum machine learning algorithm is to encode the adjacancey matrix of a graph into a Gaussian boson sampler that measures photon detection events 
using thereshold detectors which click when they detect any photons as opposed to photon number resolving detectors which count exactly how many photons are detected.

The probability of detecting a specific photon pattern <img src="https://render.githubusercontent.com/render/math?math=\textbf{n}"> is 
<img src="https://bit.ly/3tSDGNo" align="center" border="0" alt="p(\textbf{n})=\frac{1}{\sqrt{\textrm{det}(Q)}}\frac{|\textrm{Haf}(A_{\textbf{n}})|^2}{\textbf{n}!" width="221" height="51" />.

Folders are named after the model used to benchmark the new