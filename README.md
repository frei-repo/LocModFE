# LocModFE - Locally modified finite elements for approximating interface problems 
(by Stefan Frei, Thomas Richter & Thomas Wick)

This source code contains an implementation of the locally modified finite element method first proposed  by S. Frei, T. Richter, SIAM Journal on Numerical Analysis 52 (5), 2315-2334. This method is a simple approach to solve interface problems, as arising in fluid-structure-interactions or multi-phase flows, or problems on complex boundaries.

The implementation is contained in the files locmodfe.cc/h, together with a user program (step-modfe.cc) that is similar to many deal.II steps (www.dealii.org) and a problem-specific file (problem.h) with specific data for some numerical tests provided in the src folder. The source code is based on the deal.II.8.5.0 version.
 

In the test folder, we provide the parameter files parameters_test_case_1/2.prm, where algorithmic, numerical and physical parameters can be changed. In the current version, parameters are pre-set for the two numerical tests. For the meaning of the parameters, see the detailed comments in the parameter files.
 
Moreover, we provide the output that the program writes to a terminal in the files test_case_1/2.dlog in the same folder, for a comparison of the terminal output results.

In the build folder, the file CMakeLists.txt can be used to create a Makefile using cmake, see the explanations given in README.txt. Moreover, the README.txt file provides detailed installation instructions and a compact overview of all source files.

Algorithmic descriptions and details of the delivered source code are provided in S. Frei, T. Richter, T. Wick; 2018, in the arXiv documentation arXiv:1806.00999


# Installation and compilation

The implementation is based on the deal.II 8.5.0 version, which 
needs to be installed first according to the instruction given 
on www.dealii.org.

For compilation create a build folder (for example in the code/ folder) and type the 
following command in a terminal to create a Makefile

cmake . 

The code can then be compiled by writing
 
make release

in the same folder. To execute the examples, go to the test folder and run 
the following command (for example 1)
 
../build/step-modfe parameters_test_case_1.prm

or, for Example 2 
 
../build/step-modfe parameters_test_case_2.prm

In the parameter files, one can among other options set and unset the boolean 
parameter {Use Hierarchical} to choose the basis to be used and select 
the {Preconditioner type} among the options {none}, {jacobi} (i.e., diagonal preconditioning)
or {ssor} in order to reproduce the results of the test cases in the paper. 


For the provided parameter files, we provide also dlog files
with the code. These allow to check via `diff' or `meld' 
if the obtained functional values match the original ones. 
The dlog files are named {test_case_1.dlog} and {test_case_2.dlog}.

Further details can be found in the file code/README.txt
