# LocModFE

LocModFE - Locally modified finite elements for approximating interface problems
(by Stefan Frei, Thomas Richter & Thomas Wick)

This source code contains an implementation of the locally modified finite element method first proposed  by S. Frei, T. Richter, SIAM Journal on Numerical Analysis 52 (5), 2315-2334. This method is a simple approach to solve interface problems, as arising in fluid-structure-interactions or multi-phase flows, or problems on complex boundaries.

The implementation is contained in the files locmodfe.cc/h, together with a user program (step-modfe.cc) that is similar to many deal.II steps (www.dealii.org) and a problem-specific file (problem.h) with specific data for some numerical tests provided in the src folder. The source code is based on the deal.II.8.5.0 version.
 

In the test folder, we provide the parameter files parameters_test_case_1/2.prm, where algorithmic, numerical and physical parameters can be changed. In the current version, parameters are pre-set for the two numerical tests. For the meaning of the parameters, see the detailed comments in the parameter files.
 
Moreover, we provide the output that the program writes to a terminal in the files test_case_1/2.dlog in the same folder, for a comparison of the terminal output results.

In the build folder, the file CMakeLists.txt can be used to create a Makefile using cmake, see the explanations given in README.txt. Moreover, the README.txt file provides detailed installation instructions and a compact overview of all source files.

Algorithmic descriptions and details of the delivered source code are provided in S. Frei, T. Richter, T. Wick; 2018, in the arXiv documentation arXiv:1806.00999
