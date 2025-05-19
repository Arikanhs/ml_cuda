This repository is created to find optimal CUDA counts of graph algorithnms 

code folder stores the source code to find timings of triangle counting using different CUDA block sizes, makefile and and CMakeLists.txt. make files are for kokkos compiling.

Kokkos should be downloaded and the path should be provided so that the make files can compile properly.

gnn folder stores the graph neural network implementation using pyton and networkx and a script to run it on clusters.

inputs folder stores scripts to handle edge cases of different dataset formats, and stores a script to download all the datasets that will be used in the experiments (100 for now). 
