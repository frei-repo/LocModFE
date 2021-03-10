/**
   Copyright (C) 2018 by Stefan Frei (1), Thomas Richter (2),
       and Thomas Wick (3)
   
   (1) Department of Mathematics and Statistics, University of Konstanz, 
       Universitätsstr. 10, 78464 Konstanz, Germany, 
       stefan.frei@uni-konstanz.de and
       https://www.math.uni-konstanz.de/~frei

   (2) Institut für Analysis und Numerik, Universit\"at Magdeburg,
       Universit\"atsplatz 2, 39106 Magdeburg, Germany, 
       thomas.richter@ovgu.de and 
       https://www.math.uni-magdeburg.de/~richter/

   (3) Institut f\"ur Angewandte Mathematik, Leibniz Universit\"at Hannover,
       Welfengarten 1, 30167 Hannover, Germany, 
       thomas.wick@ifam.uni-hannover.de and 
       https://www.ifam.uni-hannover.de/wick.html
   
   This file is subject to the GNU Lesser General Public License
   (LGPL) as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later
   version. Please refer to the webpage http://www.dealii.org/
   under the link License for the text and further information
   on this license. You can use it, redistribute it, and/or
   modify it under the terms of this licence.
*/


/*********************** Introduction *******************************************
 *
 * This source code contains an implementation of the locally 
 * modified finite element method (files locmodfe.cc/h), 
 * together with a user program (step-modfe.cc) that is similar to
 * many deal.ii steps and a problem file-specific file (problem.h) 
 * with specific data for some numerical tests in the src folder.
 * The source code is based on the deal.II.8.5.0 version
 *
 * In the test folder, we provide the parameter files parameters_test_case_1/2.prm,
 * where algorithmic, numerical and physical parameters can be changed. In the 
 * current version, parameters are pre-set for the two numerical tests presented 
 * in the accompanying paper. For the meaning of the parameters, see the detailed 
 * comments in the parameter files).
 *
 * Moreover, we provide the output that the program writes to a terminal in the 
 * files test_case_1/2.dlog in the same folder, for a comparison of the results.
 *
 * In the build folder, the file CMakeLists.txt can be used to create a Makefile
 * using cmake, see the following explanation.
 *
 ********************************************************************************/


/************ Installation and compilation ******************************************
*
* The implementation is based on the deal.II 8.5.0 version, which 
* needs to be installed first according to the instruction given 
* on www.dealii.org.
*
*
* For compilation create a build folder (for example in the code folder), enter this folder and type the 
* following command in a terminal to create a Makefile
*
* cmake .. 
*
* The code can then be compiled by writing
* 
* make release
*
* in the same folder. To execute the examples, go to the data folder and run 
* the following command (for example 1)
* 
* ../code/build/step-modfe parameters_test_case_1.prm
*
* or, for Example 2 
* 
* ../code/build/step-modfe parameters_test_case_2.prm
*
* In the parameter files, one can among other options set and unset the boolean 
* parameter {Use Hierarchical} to choose the basis to be used and select 
* the {Preconditioner type} among the options {none}, {jacobi} (i.e., diagonal preconditioning)
* or {ssor} in order to reproduce the results of the test cases in the paper. 
*
*
* For the provided parameter files, we provide also dlog files
* with the code. These allow to check via `diff' or `meld' 
* if the obtained functional values match the original ones. 
* The dlog files are named {test_case_1.dlog} and {test_case_2.dlog}.
*
****************************************************************************************/


/************************* Content of the source code ****************************
 *
 * The source code includes the following files and classes:
 *
 * 1) locmodfe.cc/h: Contain all functions that are specific to the locally 
 *              modified FE method
 *    a) class LocModFEValues : Extends the FEValues class in deal.ii, where the 
 *              local basis functions on the reference patches are evaluated 
 *    b) class LocModFE : Key class of the locally modified finite element method
 *
 * 2) step-modfe.cc: 
 *    a) class ParameterReader: Read in parameters from a seperate parameter file
 *    b) class InterfaceProblem : local user file similar to many deal.II tutorial 
 *              steps, which controls the general workflow of the code, for example the 
 *              solution algorithm, assembly of system matrix and right-hand side 
 *              and output
 *    c) int main()
 *
 * 3) problem.h: Problem-specific definition of geometry, boundary conditions and 
 *              analytical solution
 *    a) class LevelSet : Implicit definition of the interface and the sub-domains 
 *    b) classDirichletBoundaryConditions : Definition of the Dirichlet data
 *    c) class ManufacturedSolution : Analytical solution for error estimation 
 *
 *
 * In addition, we provide some files to execute the numerical tests provided in the 
 * accompanying paper:
 *
 * 1) unit_square.inp: A simple geometry file considiting of one large unit cell, 
 *                     that will be refined in the example programs
 * 2) parameters_test_case_1/2.prm: One parameter file for each test_case to change 
 *                                  some algorithmical, numerical and physical parameters 
 *                                  without subsequent compilations
 * 3) CMakeLists.txt: A cmake file as a possibility to create a Makefile to compile the source code
 *
 ***********************************************************************************/


