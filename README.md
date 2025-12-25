# Numerical method for feasible and approximately optimal solutions of multi-marginal optimal transport beyond discrete measures

+ By Ariel Neufeld and Qikun Xiang
+ Article link (arXiv): https://arxiv.org/abs/2203.01633

# Description of files

+ **cutplane/** contains class files for the instances of the cutting-plane algorithm
	 - **LSIPMinCuttingPlaneAlgo.m** defines an abstract class for cutting-plane algorithms used for solving linear semi-infinite programming problems or large-scale linear programming problems
	 - **MMOT1DFluid.m** defines a class for the multi-marginal optimal transport problem that stems from fluid dynamics; it is used in Experiment 1
	 - **MMOT2DWassersteinBarycenter.m** defines a class for the multi-marginal optimal transport formulation of the Wasserstein barycenter problem; it is used in Experiment 2
	 - **MMOT1DCPWA.m** defines a class for the multi-marginal optimal transport problem involving a continuous piece-wise affine cost function and one-dimensional marginals; it is used in Experiment 3
	 - **OTDiscrete.m** defines a class for large-scale discrete optimal transport problem; a cutting-plane/constraint generation scheme is needed since the linear programming formulation is too large and will cause memory throttling issues

+ **probmeas/** contains class files for probability measures used to represent the input measures of multi-marginal optimal transport problems
    - **HasTractableQuadraticIntegrals.m** defines an abstract class for probability measures which admits tractably computable first and second moments (i.e., mean vector and covariance matrix)
    - **ProbMeas1DInterval.m** defines an abstract class for one-dimensional probability meausres supported on compact intervals
    - **ProbMeas1DCPWADens.m** defines a class for one-dimensional probability measures supported on compact intervals with continuous piece-wise affine probability density functions
    - **ProbMeas1DMixNorm.m** defines a class for one-dimensional probability measures supported on compact intervals whose distributions are mixtures of normal distributions (truncated to the support intervals)
    - **ProbMeas2DConvexPolytope.m** defines an abstract class for two-dimensional probability measures whose supports are contained in convex polytopes
    - **ProbMeas2DConvexPolytopeWithW2OT.m** defines an abstract class for two-dimensional probability measures whose supports are contained in convex polytopes which support semi-discrete 2-Wasserstein optimal transport (in additional to semi-discrete 1-Wasserstein optimal transport that is supported by all instances of ProbMeas2DConvexPolytope)
    - **ProbMeas2DAffDens.m** defines a class for two-dimensional probability measures whose supports are convex polytopes with affine probability density functions; supports semi-discrete 2-Wasserstein optimal transport
    - **ProbMeas2DCPWADens.m** defines a class for two-dimensional probability measures whose supports are unions of interior-disjoint triangles with continuous piece-wise affine probability density functions; supports semi-discrete 2-Wasserstein optimal transport
    - **ProbMeas2DMixNorm.m** defines a class for two-dimensional probability measures whose supports are unions of interior-disjoint triangles that correspond to mixtures of multivariate normal distributions (truncated to the supports)

+ **mex/** contains the C++ code used when solving the 2D Wasserstein barycenter problem
    - **Makefile** is used for compiling the C++ code into mex files for MATLAB
    - **power\_diagram\_intersection.m** is an empty MATLAB file for creating the MATLAB interface for the mex function power\_diagram\_intersection; this function computes power diagrams corresponding to multiple collections of circles as well as the polygonal complex formed by their intersections; it is used in the global minimization oracle for the class MMOT2DWassersteinBarycenter
    - **mesh\_intersect\_power\_diagram.m** is an empty MATLAB file for creating the MATLAB interface for the mex function mesh\_intersect\_power\_diagram; this function computes a power diagram corresponding to a collection of circles as well as the polygonal complex formed by the intersection of this power diagram and a given triangular mesh; it is used when computing semi-discrete 2-Wasserstein optimal transport for the class ProbMeas2DCPWADens
    - **src/** contains the C++ source code
        - **power\_diagram\_intersection.hpp** is the C++ header file for all of the C++ code
        - **power\_diagram\_intersection.cpp** is the C++ source file containing the two functions power\_diagram\_intersection and mesh\_intersect\_power\_diagram
        - **mex\_wrapper\_power\_diagram\_intersection.cpp** contains the mex wrapper class for the C++ function power\_diagram\_intersection implemented in **power\_diagram\_intersection.cpp**
        - **mex\_wrapper\_mesh\_intersect\_power\_diagram.cpp** contains the mex wrapper class for the C++ function mesh\_intersect\_power\_diagram implemented in **power\_diagram\_intersection.cpp**
        - **test\_power\_diagram\_intersection.cpp** contains the C++ code for testing the function power\_diagram\_intersection
        - **test\_mesh\_intersect\_power\_diagram.cpp** contains the C++ code for testing the function mesh\_intersect\_power\_diagram
        - **kdtree-cpp/** contains the C++ Kd-tree library implemented by Christoph Dalitz and Jens Wilberg (2018), which is used by **power\_diagram\_intersection.cpp**

+ **utils/** contains additional utility functions and external libraries
    - **check\_triangulation\_simplicial\_cover.m** defines a function which checks whether a given triangular mesh satisfies the conditions of a simplicial cover
    - **cleanup\_trangles.m** defines a function which iterates through a triangular mesh and removes the degenerate triangles (i.e., those triangles with close to 0 area)
    - **comonotone\_coupling.m** defines a function which computes a coupling of several discrete probability measures (i.e., it implements Algorithm 0 in the paper)
    - **discrete\_OT.m** defines a function which computes discrete optimal transport with any cost function via linear programming; for large-scale discrete optimal transport problems this function becomes memory-inefficient and one should use the OTDiscrete class instead
    - **discrete\_reassembly\_1D.m** defines a function which computes a *reassembly* involving a discrete coupling and discrete one-dimensional marginals; gluing is done with respect to the comonotone couplings between the marginals
    - **discrete\_reassembly.m** defines a function which computes a *reassmebly* involving a discrete coupling and discrete marginals that are not necessarily one-dimensional; gluing is done with respect to the discrete couplings between the marginals that are provided as input
    - **tight\_subplot/** contains a MATLAB library that is used for creating figures with narrow margins

+ **exp/** contains the scripts to run the numerical experiments (see below for detailed instructions)


# Instructions to run the numerical experiments

## Configurations

+ MATLAB (version 2024a or above) must be installed.
+ All folders and subfolders must be added to the MATLAB search path. 
+ Gurobi optimization (version 11.0.3 or above) must be installed and the relevant files must be added to the MATLAB search path. 
+ The following lines in the file **exp/global_config.m** can be edited to change the directories for the save files and log files if necessary.

		% root folder for all save files
		CONFIG.SAVEPATH_ROOT = '../';

		% root folder for all log files
		CONFIG.LOGPATH_ROOT = '../Logs/';

## Experiment 1: fluid dynamics

+ All the relevant files used in this experiment are located in **exp/IncompressibleFluid1D_Exp/**.

### Step 1: generate the input file
+ Run **IFMM\_prepare.m** to generate an input file containing the set-up of the experiment.

### Step 2: run the experiment and generate the output file
+ Run **IFMM\_run.m** to compute the lower bounds, the upper bounds, and the volume preserving maps at intermediate time points for each setting and each final volume preserving map (i.e., each arrangement function). The outputs will be saved in an output file. 

### Step 3: plot the results
+ Run **IFMM\_plot\_maps.m** to plot the computed volume preserving maps at intermediate time points.
+ Run **IFMM\_plot\_trajectories.m** to plot the computed particle trajectories at intermediate time points.
+ Run **IFMM\_plot\_bounds.m** to plot the computed lower and upper bounds as well as the comparison between the computed sub-optimality estimates and their a priori upper bounds.

		
## Experiment 2: Wasserstein barycenter

+ All the relevant files used in this experiment are located in **exp/WassersteinBarycenter_Exp/**.
+ Part of the code uses mex functions which need to be compiled from C++. The Makefile for the compilation process only supports macOS (both Intel or Apple Silicon are supported). The Makefile needs to be modified for other platforms.
+ The Computational Geometry Algorithms Library (CGAL) must be installed. The installation can be done either directly from the [GitHub repository](https://github.com/CGAL/cgal/), or via a package manager such as [Homebrew](https://brew.sh) or [Conda](https://anaconda.org/anaconda/conda).
+ Uncomment the line with `MEX_SUFFIX` in the following part of **mex/Makefile** depending on whether the machine has an Intel or Apple Silicon chip.

		# Suffix of mex files on this operating system
		# macOS with Apple Silicon
		#MEX_SUFFIX = mexmaca64
		
		# macOS with Intel
		#MEX_SUFFIX = mexmaci64
+ Modify the value of the variable `MATLAB_PATH` in **mex/Makefile** to the location of the MATLAB app on the machine.

		# Path to MATLAB
		MATLAB_PATH = /Applications/MATLAB_R2024a.app
		
+ Modify the following lines in **mex/Makefile** to the locations of the header files and the library files of CGAL (e.g., under `/usr/local`, `/opt/homebrew`, or `/opt/miniconda3`)

		# Include path (for header files)
		INCLUDE_PATH = /usr/local/include

		# Library path (for linking to libraries)
		LIBRARY_PATH = /usr/local/lib

### Step 0: compile the mex functions
+ In Terminal, use `cd` to change the current working directory to the **mex/** directory.
+ Execute the following commands in Terminal to build the mex files.

		mkdir build
		make


### Step 1: generate the input file
+ Run **WBMM\_prepare.m** to generate an input file containing the set-up of the experiment.

### Step 2: run the experiment and generate the output files
+ Run **WBMM\_run.m** to compute the lower bounds for the optimal value of the Wasserstein barycenter problem. The outputs will be saved in an output file. 
+ Run **WBMM\_OT.m** to compute the pairwise semi-discrete 1-Wasserstein optimal transport for the marginals. The outputs will be saved in an output file. 
+ Run **WBMM\_UB.m** to compute the upper bounds for the optimal value of the Wasserstein barycenter problem using the semi-discrete 1-Wasserstein optimal couplings. The outputs will be saved in an output file. 
+ Run **WBMM\_W2OT.m** to compute the pairwise semi-discrete 2-Wasserstein optimal transport for the marginals. The outputs will be saved in an output file. 
+ Run **WBMM\_W2OTUB.m** to compute the upper bounds for the optimal value of the Wasserstein barycenter problem using the semi-discrete 2-Wasserstein optimal couplings. The outputs will be saved in an output file. 

### Step 3: plot the results
+ Run **WBMM\_plot\_density.m** to plot the probability density functions of the input measures.
+ Run **WBMM\_plot\_histogram.m** to plot the histograms of the computed approximate Wasserstein barycenters. 
+ Run **WBMM\_plot\_bounds.m** to plot the computed lower and upper bounds as well as the comparison between the computed sub-optimality estimates and their a priori upper bounds.



## Experiment 3: continuous piece-wise affine cost function
+ All the relevant files used in this experiment are located in **exp/CPWACost_Exp/**.

### Step 1: generate the input file
+ Run **CPWAMM\_prepare.m** to generate an input file containing the set-up of the experiment.

### Step 2: run the experiment and generate the output file
+ Run **CPWAMM\_run.m** to compute the lower bounds and the upper bounds for the optimal value of the multi-marginal optimal transport problem. The outputs will be saved in an output file. 

### Step 3: plot the results
+ Run **CPWAMM\_plot\_bounds.m** to plot the computed lower and upper bounds as well as the comparison between the computed sub-optimality estimates and their a priori upper bounds.