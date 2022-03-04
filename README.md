# Numerical method for feasible and approximately optimal solutions of multi-marginal optimal transport beyond discrete measures

+ By Ariel Neufeld and Qikun Xiang
+ Article link (arXiv): https://arxiv.org/abs/2203.01633

# Description of files

+ func/truncmixnorm/      contains functions related to truncated mixture of normal distributions
    - norm\_partialexp.m: function used for computing a specific form of expectation related to a normal distribution
    - truncmixnorm\_partialexp.m: function used for computing a specific form of expectation related to a truncated mixture of normal distributions
    - truncmixnorm\_invcdf.m: function used for computing the inverse cumulative distribution function of a truncated mixture of normal distributions via the bisection method
    - truncmixnorm\_momentset.m: function used for computing the integrals of the functions in an interpolation function basis (which characterizes a moment set) with respect to a truncated mixture of normal distributions
    - truncmixnorm\_momentset\_construct.m: function used for iteratively constructing a moment set surrounding a truncated mixture of normal distributions with a given number of knots

+ func/momentset1d/      contains functions related to moment set surrounding one-dimensional distribution with bounded support
    - momentset1d\_basisfunc\_bounded.m: function used for computing the values of the functions in an interpolation function basis for given inputs
    - momentset1d\_measinit\_bounded.m: function that returns a discrete measure from a moment set characterized by a given interpolation function basis

+ func/CPWA/       contains functions related to multivariate continuous piece-wise affine (CPWA) functions
    - CPWA\_diff\_eval.m: function used for evaluating the difference between a separable CPWA function and a non-separable CPWA function at given inputs
    - CPWA\_nonsep\_eval.m: function used for evaluating a non-separable CPWA function at given inputs
    - CPWA\_diff\_min\_MILP\_gurobi.m: function that returns a gurobi model which formulates a global minimization problem (in which the difference between a separable CPWA function and a non-separable CPWA function is minimized) into a mixed-integer linear programming (MILP) problem

+ func/reassembly/       contains functions related to coupling and reassembly
    - comonotone\_coupling.m: function used for computing the comonotone coupling of discrete measures (i.e., a joint distribution formed with the given set of discrete marginals and the comonotone copula)
    - discretecopula\_samp.m: function used for generating independent samples from the copula of a given discrete measure
    - reassembly\_increment.m: function used for computing the reassembly of a given discrete measure with a given set of discrete marginals

+ func/MMOT/       contains functions used in the algorithm for approximately solving multi-marginal optimal transport problems
    - MMOT\_CPWA\_cutplane.m: implementation of the cutting plane algorithm for solving linear semi-infinite programming problems
    - MMOT\_CPWA\_feascons.m: function that returns constraints (also known as feasibility cuts) in the cutting plane algorithm
    - MMOT\_CPWA\_primal\_approx.m: function used for approximately computing an upper bound on the optimal value of the multi-marginal optimal transport problem via Monte Carlo integration

+ exp/            contains the scripts to run the experiment (see below for detailed instructions)

+ utils/          contains external libraries
    - utils/tight\_subplot/:             used for creating figures with narrow margins
    - utils/parfor\_progress/:           used for printing a progress bar in the console

# Instructions to run the numerical experiment

## Configurations

+ All folders and subfolders must be added to the MATLAB search path. 
+ Gurobi optimization (version 9.5.0 or above) must be installed on the machine and relevant files must be added to the search path. 

## Running

### Step 1: generate the input files
+ Run exp/exp\_prepare1.m to generate a file exp/inputs.mat containing the settings and the approximation scheme.
+ Run exp/exp\_prepare2.m to generate a file exp/invcdf.mat containing the necessary inputs for approximating the inverse cumulative distribution functions of the marginals.

### Step 2: compute the lower and upper bounds on the optimal value of the multi-marginal optimal transport problem
+ Run exp/exp\_run1\_LB.m to compute lower bounds by solving linear semi-infinite programming problems via the cutting plane algorithm. This will create an output file exp/rst\_LB.mat.
+ Run exp/exp\_run2\_UB.m to compute upper bounds by first computing reassemblies and then approximating integrals with respect to the reassemblies via Monte Carlo integration. This will create an output file exp/rst\_UB.mat.

### Step 3: plot the results
+ Run exp/exp\_plot\_results.m to plot the upper and lower bounds as well as the comparison between the differences between the bounds and their theoretical upper bounds.
