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


/** Abstract, keywords, library information and code structure
 * -----------------------------------------------------------
 * 
 * Abstract: In this work, we describe a simple finite element approach 
 * that is able to resolve weak discontinuities in interface problems accurately.
 * The approach is based on a fixed patch mesh consisting of quadrilaterals, 
 * that will stay unchanged independent of the position of the interface. Inside the
 * patches we refine once more, either in eight triangles or in four quadrilaterals, 
 * in such a way that the interface is locally resolved.
 * The resulting finite element approach can be considered a fitted 
 * finite element approach. In our practical implementation, we do not construct 
 * this fitted mesh, however. Instead, the local degrees of freedom are included 
 * in a parametric way in the finite element space, or to be more precise in the 
 * local mappings between a reference patch and the physical patches.
 * We describe the implementation in deal.II in detail and show two 
 * numerical examples to illustrate the performance of the approach.
 *
 *
 * Keywords: locally modified finite element method, interface problems, 
 *           cut cell, fitted finite elements, hierarchical basis 
 *          
 *
 * Source code is based on the deal.II.8.5.0 version
 *
 *
 * This source code includes the following files and classes:
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
 *              solution algorithm, assembly of system matrix and right-hand side and output
 *    c) int main()
 *
 * 3) problem.h: Problem-specific definition of geometry, boundary conditions and 
 *              analytical solution
 *    a) class LevelSet : Implicit definition of the interface and the sub-domains 
 *    b) classDirichletBoundaryConditions : Definition of the Dirichlet data
 *    c) class ManufacturedSolution : Analytical solution for error estimation 
 */


// Include files
//--------------

//First, we include some functionalities of 
//the deal.II library and some C++ header files.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h> 
#include <deal.II/base/parameter_handler.h> 

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>


// C++ includes
#include <fstream>
#include <sstream>

// Include user files
#include "locmodfe.h"
#include "problem.h"


// Important global namespaces from 
// deal.II and C++
using namespace dealii;
using namespace std;

/**
   Class for reading parameters from external input files
*/
class ParameterReader : public Subscriptor
{
  public:
    ParameterReader (
        ParameterHandler &);
    void
    read_parameters (
        const std::string);

  private:
    void
    declare_parameters ();
    ParameterHandler &prm;
};

ParameterReader::ParameterReader (
    ParameterHandler &paramhandler)
    :
        prm(paramhandler)
{
}

void
ParameterReader::read_parameters (
    const std::string parameter_file)
{
  declare_parameters();

  if (!prm.read_input(parameter_file, true))
    AssertThrow(false, ExcMessage("could not read .prm!"));
}

void
ParameterReader::declare_parameters ()
{
  prm.enter_subsection("Global parameters");
  {
    prm.declare_entry("Grid filename", "grid_",  Patterns::Anything());
    prm.declare_entry("Output filename", "solution_",  Patterns::Anything());

    prm.declare_entry("Test case", "1", Patterns::Integer(0));
    prm.declare_entry("Use hierarchical", "false", Patterns::Bool());

    prm.declare_entry("Visc 1", "0.0", Patterns::Double(0));
    prm.declare_entry("Visc 2", "0.0", Patterns::Double(0));

    prm.declare_entry("Global ref", "1", Patterns::Integer(0));
    prm.declare_entry("Max cycles", "1", Patterns::Integer(0));
  }
  prm.leave_subsection();  


  prm.enter_subsection("Solver parameters");
  {
    prm.declare_entry("Direct linear solver", "true", Patterns::Bool());
    prm.declare_entry("Preconditioner type", "none",  Patterns::Anything());

    prm.declare_entry("Max linear iter", "0", Patterns::Integer(0));
    prm.declare_entry("Tol linear solver", "0.0", Patterns::Double(0));
    prm.declare_entry("Omega prec", "0.0", Patterns::Double(0));
  }
  prm.leave_subsection();  

}



/**
   The class InterfaceProblem controls the general workflow of the program,
   starting from the initialisation of objects and variables and the definition
   of the initial solution, over the Newton loop, the assembly of matrices and 
   right-hand sides, the solution of the linear systems to the computation
   of functionals and the generation of output files in vtk format
*/
template <int dim>
class InterfaceProblem 
{
public:
  
  //Constructor and Destructor
  InterfaceProblem (const unsigned int degree, ParameterHandler &param);
  ~InterfaceProblem (); 

  //The function run is the function, that is called from the main routine
  //and contains initialisations, a call of the newton iteration, the
  //computation of functional values and output  
  void run ();
  
private:
  
  // Setting the runtime parameters, material 
  // parameters, model parameters and the mesh.
  void set_runtime_parameters ();
  
  //Construction of the matrix structure and initialization of variables
  void setup_system ();

  //Assembly of the linear system of equations
  void assemble_system_matrix ();   
  void assemble_system_rhs ();
  
  //Dirichlet boundary conditions
  void set_initial_bc ();
  void set_newton_bc ();

  //Solve the linear system of equations 
  void solve ();

  //Implementation of Newton's method 
  void newton_iteration();

  //Compute functional values (error norms, etc)
  void compute_functional_values (bool convergence_rate); 


  //This object is used to access the functions specific to the 
  //locally modified finite element method
  LocModFE<dim> lmfe;

  //Standard (deal.II) variables
  const unsigned int   degree;

  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;  
  ParameterHandler     &prm;

  ConstraintMatrix     constraints;
  SparsityPattern      sparsity_pattern; 
  SparseMatrix<double> system_matrix; 
  
  Vector<double>       solution, newton_update;
  Vector<double>       system_rhs;
  
  //Runtime output
  TimerOutput          timer;

  //Basic part of the filenames for the vtk output files 
  std::string          output_basis_filename;

  // Problem-specific parameters and variables
  // visc_i stands for the diffusion coefficient in Omega_i 
  double viscosity, visc_1, visc_2; 
  double _yoffset; // y-position of the circle

  // Element information
  double cell_diameter, min_cell_diameter;
  double cell_vertex_distance, min_cell_vertex_distance;
  double old_min_cell_diameter;

  // Value of the right hand side f 
  double force;

  // Further model and global parameters
  unsigned max_no_refinement_cycles;
  unsigned int N_testcase2;
  unsigned int max_obtained_no_newton_steps;
  double lower_bound_newton_residuum;
  unsigned int linear_iterations;
  bool bool_use_direct_solver;
  bool _hierarchical;
  int test_case;

  // Values for error estimation
  double old_local_error_L2, old_local_error_H1, old_local_error_H1_semi;


};


//Constructor
template <int dim>
InterfaceProblem<dim>::InterfaceProblem (const unsigned int degree, 
					 ParameterHandler &param)
  :
  degree (degree),
  triangulation (Triangulation<dim>::maximum_smoothing),
  fe (degree),                   
  dof_handler (triangulation),
  prm(param),
  timer (std::cout, TimerOutput::summary, TimerOutput::cpu_times)		
{}


// This is the standard destructor.
template <int dim>
InterfaceProblem<dim>::~InterfaceProblem () 
{}


/**
   Setting the runtime parameters, material 
   parameters, model parameters and the mesh.
   Most of these values are obtained from a 
   parameter file.
*/
template <int dim>
void InterfaceProblem<dim>::set_runtime_parameters ()
{
  // Getting runtime parameters from 
  // parameter files
  prm.enter_subsection("Global parameters");

  // Two test cases
  // Test case 1: circle and grid refinement
  // Test case 2: moving circle
  test_case = prm.get_integer("Test case");


  _hierarchical = prm.get_bool("Use hierarchical");
  lmfe.set_bool_hierarchical (_hierarchical);

  _yoffset = 0.;
  lmfe.LevelSetFunction()->set_y_offset (_yoffset);

  // Parameters
  viscosity = 1.0;
  visc_1 = prm.get_double("Visc 1"); //0.1; // inner (\Omega_1)
  visc_2 = prm.get_double("Visc 2"); //1.0; // outer (\Omega_2)

  // Right hand side force (redefined below)
  force = 1.0;

  std::string grid_name;
  grid_name  = prm.get ("Grid filename");
  
  GridIn<dim> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file(grid_name.c_str());      
  Assert (dim==2, ExcInternalError());
  grid_in.read_ucd (input_file); 

  unsigned int global_refinement_steps = prm.get_integer("Global ref");
  triangulation.refine_global (global_refinement_steps); 
  
  // Cycles in the final loop: here
  // refinement cycles
  if (test_case == 1)
    max_no_refinement_cycles = prm.get_integer("Max cycles");
  
  // Cycles in the final loop: here 
  // the number of yoffset numbers n
  if (test_case == 2)
    N_testcase2 = prm.get_integer("Max cycles");

  // Getting output filename
  output_basis_filename        = prm.get ("Output filename");
  
  prm.leave_subsection();


  // Remaining hard-coded values
  max_obtained_no_newton_steps = 0;
  lower_bound_newton_residuum = 1.0e-12; 


  // Initialize values needed for 
  // computing the convergence order 
  // in the error norms
  old_min_cell_diameter = 1.;
  old_local_error_L2 = 1.;
  old_local_error_H1 = 1.;
  old_local_error_H1_semi = 1.;


  std::cout << "\n==========================================" 
	    << "============================================" 
	    << std::endl; 
  std::cout << "Parameters\n" 
	    << "==========\n"
	    << "Test case:       "   <<  test_case << "\n"
	    << "Inner viscosity: "   <<  visc_1 << "\n"
	    << "Outer viscosity: "   <<  visc_2 << "\n"
	    << "Force:           "   <<  force << "\n"
	    << std::endl;

}


/**
   This function is similar to many deal.II tuturial steps.
   We set-up the degrees of freedom 
   and intialize the matrix, right hand side vector and solution. 
   We then intialize the quadrature and compute cell diameters.
*/
template <int dim>
void InterfaceProblem<dim>::setup_system ()
{
  timer.enter_section("Setup system.");

  system_matrix.clear ();
  
  dof_handler.distribute_dofs (fe);  
  DoFRenumbering::Cuthill_McKee (dof_handler);

 

  {				 
    constraints.clear ();
    set_newton_bc ();
    DoFTools::make_hanging_node_constraints (dof_handler,
					     constraints);
  }
  constraints.close ();
  

  std::cout << "Cells:\t"
            << triangulation.n_active_cells()
            << std::endl  	  
            << "DoFs:\t"
            << dof_handler.n_dofs()
            << std::endl;


 
      
 {
   DynamicSparsityPattern csp (dof_handler.n_dofs());
   
   DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);
   
   sparsity_pattern.copy_from (csp);
 }
 
 system_matrix.reinit (sparsity_pattern);
 
 // Current solution 
 solution.reinit (dof_handler.n_dofs());
 
 // Updates for Newton's method (even that the problem 
 // is linear, we use a Newton method to allow 
 // for an easier future extension for nonlinear problems)
 newton_update.reinit (dof_handler.n_dofs());
 
 // Residual for  Newton's method
 system_rhs.reinit (dof_handler.n_dofs());
 
 

 // Compute element diameters 
 // and longest side
  min_cell_diameter = 1.0e+10;
  min_cell_vertex_distance = 1.0e+10;
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  for (; cell!=endc; ++cell)
    { 
      cell_diameter = cell->diameter();
      cell_vertex_distance = cell->minimum_vertex_distance ();
      if (min_cell_diameter > cell_diameter)
	    min_cell_diameter = cell_diameter;

      if (min_cell_vertex_distance > cell_vertex_distance)
	    min_cell_vertex_distance = cell_vertex_distance;
      
    }

  timer.exit_section(); 
}


/**
   Assembly of the system matrix
   Left hand side of Newton's method.
*/
template <int dim>
void InterfaceProblem<dim>::assemble_system_matrix ()
{
  timer.enter_section("Assemble Matrix.");
  system_matrix=0;

 				     				   
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
      
  unsigned int femtype=0;
  const unsigned int dofs_per_cell   = fe.dofs_per_cell; 
  std::vector<double> LocalDiscChi;
  std::vector<int> NodesAtInterface;
  double ChiValue=0.;
  FullMatrix<double> M(dim, dofs_per_cell);
  FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
      
  std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
  std::vector<double> phi_i_u (dofs_per_cell); 
  std::vector<Tensor<1,dim> > phi_i_grads_u(dofs_per_cell);

  LocModFEValues<dim>* fe_values;       
      
  //We initialize one LocModFEValue object for patch type 0 and one for patch 
  //types 1 to 3, due to the different number of integration points
  Quadrature<dim> quadrature_formula0 = lmfe.compute_quadrature(0);
  LocModFEValues<dim> fe_values0 (fe, quadrature_formula0,
				  _hierarchical, update_values | update_quadrature_points  |
				  update_JxW_values | update_gradients);
  
  Quadrature<dim> quadrature_formula1 = lmfe.compute_quadrature(1);
  LocModFEValues<dim> fe_values1 (fe, quadrature_formula1,
				  _hierarchical, update_values | update_quadrature_points  |
				  update_JxW_values | update_gradients);
  
  //Loop over all patches
  for (unsigned int cell_counter = 0; cell!=endc; ++cell,++cell_counter)
    { 
      local_matrix=0;
      
      //Set patch type (femtype), map T_P (M), local level set function (LocalDiscChi)
      //and list of nodes at the interface
      lmfe.init_FEM (cell,cell_counter,M,dofs_per_cell,femtype, LocalDiscChi, NodesAtInterface);
      
      Quadrature<dim> quadrature_formula = lmfe.compute_quadrature(femtype);
      const unsigned int   n_q_points      = quadrature_formula.size();
      
      //Choose one of the initialized objects for LocModFEValues      
      if (femtype==0) fe_values = &fe_values0;
      else fe_values = &fe_values1;
      
      fe_values->SetFemtypeAndQuadrature(quadrature_formula, femtype, M);
      
      std::vector<double> J(n_q_points); 
      
      //Now the shape functions on the reference patch are initialized
      fe_values->reinit(J);
      
      for (unsigned int q=0; q<n_q_points; ++q)
	{
	  for (unsigned int k=0; k<dofs_per_cell; ++k)
	    {
	      phi_i_u[k]       = fe_values->shape_value (k, q);
	      phi_i_grads_u[k] = fe_values->shape_grad (k, q);
	    }
	  
	  //Get the domain affiliation to set the diffusion coefficient (viscosity). 
          //This is based on the discrete level set function, such that
          //all quadrature points in a sub-cell lie in the same sub-domain 
          lmfe.ComputeLocalDiscChi(ChiValue, q, *fe_values, dofs_per_cell, LocalDiscChi);				

	  if (ChiValue < 0.) 	
	    viscosity = visc_1; //Subdomain Omega_1 (inside the circle) 
	  else 
	    viscosity = visc_2; //Subdomain Omega_2 (outside the circle)
	  
	  //Compute matrix entries as in other deal.ii program.
	  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    {
	      for (unsigned int j=0; j<dofs_per_cell; ++j)
		{	
		  local_matrix(j,i) += viscosity * 
		    phi_i_grads_u[i] * phi_i_grads_u[j] * J[q] * quadrature_formula.weight(q); 
		}	   
	    }	   
	}    
      
      //Write into global matrix
      //This is the same as discussed in other deal.II tutorial steps.
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_matrix, local_dof_indices,
					      system_matrix);
    }   
  
  timer.exit_section();
}



/**
   In this function we assemble the PDE, 
   which is realized as right hand side of Newton's method (its residual).
   The framework is in principal the same as for the 
   system matrix.
*/
template <int dim>
void
InterfaceProblem<dim>::assemble_system_rhs ()
{
  timer.enter_section("Assemble Rhs.");
  system_rhs=0;

  //Initializations  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  const unsigned int   dofs_per_cell   = fe.dofs_per_cell; 
  std::vector<double> LocalDiscChi;
  std::vector<int> NodesAtInterface;
  unsigned int cell_counter = 0;
  double ChiValue=0.;
  Vector<double> local_rhs (dofs_per_cell);
  
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  FullMatrix<double> M(dim, dofs_per_cell);
  Point<dim> IntPoint;
  
  Tensor<1,dim> phi_i_grads_u;
  double        phi_i_u;
  Tensor<1,dim> grad_u;
  
  //Get maximum number of integration points
  Quadrature<dim> quadrature_form = lmfe.compute_quadrature(1);
  const unsigned int max_n_q_points = quadrature_form.size();   
  
  std::vector<double> J (max_n_q_points); 
  
  LocModFEValues<dim>* fe_values;       
  
  //We initialize one LocModFEValue object for patch type 0 and one for patch 
  //types 1 to 3, due to the different number of integration points
  Quadrature<dim> quadrature_formula0 = lmfe.compute_quadrature(0);
  LocModFEValues<dim> fe_values0 (fe, quadrature_formula0,
				  _hierarchical, update_values | update_quadrature_points  |
				  update_JxW_values | update_gradients);
  
  Quadrature<dim> quadrature_formula1 = lmfe.compute_quadrature(1);
  LocModFEValues<dim> fe_values1 (fe, quadrature_formula1,
				  _hierarchical, update_values | update_quadrature_points  |
				  update_JxW_values | update_gradients);
  
   //Loop over all patches
   for (; cell!=endc; ++cell, cell_counter++)
     { 
       local_rhs=0.;   	
 
       // InitFem: This function needs to be called 
       // in each patch before assembling the matrix
       // and the right hand side. Here, based on the cut cell
       // information, the type of the element (_femtype) is determined
       // and the correct finite element is build.
       unsigned int femtype = 0;
       lmfe.init_FEM (cell,cell_counter,M,dofs_per_cell,femtype, LocalDiscChi, NodesAtInterface);
       
       //Initialize the quadrate formula based on the reference patch type
       Quadrature<dim> quadrature_formula = lmfe.compute_quadrature(femtype);
       const unsigned int   n_q_points    = quadrature_formula.size();
 
			 std::vector<std::vector<Tensor<1,dim> > > 
    			old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (1));
        
       //Choose one of the two objects for LocModFEValues initialized above
       if (femtype == 0) 
	    	fe_values = &fe_values0;
       else 
	    	fe_values = &fe_values1;    
       
       fe_values->SetFemtypeAndQuadrature(quadrature_formula, femtype, M);
       
       //By calling the reinit function of the LocModFEValues object, we compute
       //the value of the shape functions and gradients and the derivatives of
       //the map T_P in the quadrature point on the reference patch
       fe_values->reinit (J);	 
       
       fe_values->get_function_gradients (cell, solution, old_solution_grads);
       
       //Loop over all integration points q
       for (unsigned int q=0; q<n_q_points; ++q)
	 {
	   // Domain affiliation (ChiValue) based on the discrete level-set function chi_h
           //parametrised in the vector LocalDiscChi
	   lmfe.ComputeLocalDiscChi(ChiValue, q, *fe_values, dofs_per_cell, LocalDiscChi);
	   
	   // Evaluation of the right-hand side f and setting the diffusion coefficient
	   if (ChiValue < 0.)
	     {
	       // inner sub-domain (Omega_1)
	       lmfe.ComputePhysicalIntegrationPoint(IntPoint, *fe_values, M, dofs_per_cell,q);
	       double r_squared = 
		 IntPoint[0]*IntPoint[0] + (IntPoint[1]-_yoffset) * (IntPoint[1]-_yoffset);
	       
	       force = 32.0 * visc_1 * visc_2 * r_squared;
	       viscosity = visc_1;
	     }
	   else 
	     {
	       // outer sub-domain (Omega_2)
	       force = 4.0 * visc_1 * visc_2;
	       viscosity = visc_2;
	     }
	   
	   
	   grad_u = old_solution_grads[q][0];
	   
	   // Assembly of the right-hand as usual in a finite element program
	   for (unsigned int i=0; i<dofs_per_cell; ++i)
	     {
	       phi_i_grads_u = fe_values->shape_grad (i, q);
	       phi_i_u       = fe_values->shape_value (i, q);
	       
	       local_rhs[i] -= (viscosity * grad_u * phi_i_grads_u 
				- force * phi_i_u) * J[q] * quadrature_formula.weight(q);
	       
	     } 
	   
	 }    
       
       
       cell->get_dof_indices (local_dof_indices);
       constraints.distribute_local_to_global (local_rhs, local_dof_indices,
					       system_rhs);
       
     }  
   
   timer.exit_section();
}

/**
   The next function sets (possibly nonhomogeneous) Dirichlet conditions
   for the initial Newton guess.
   Attention: `initial' in this context has nothing to do 
   with a time-dependent problem
*/
template <int dim>
void
InterfaceProblem<dim>::set_initial_bc ()
{ 
  std::map<unsigned int,double> boundary_values;  
  
  //Modified function to set boundary values due to the 
  //(possibly) moved boundary vertices and the hierarchical basis
  //which is (in terms of deal.ii "non-primitive")
  lmfe.interpolate_boundary_values (dof_handler,
				    0,
				    DirichletBoundaryConditions<dim>(visc_1,visc_2,_yoffset),
				    boundary_values);  
  
  
  for (typename std::map<unsigned int, double>::const_iterator
	 i = boundary_values.begin();
	 i != boundary_values.end();
       ++i) {
    solution(i->first) = i->second;
  }
  
}


/**
   This function applies boundary conditions 
   to the Newton iteration steps. For all variables that
   have (non-homogeneous) Dirichlet conditions on some (or all) parts
   of the outer boundary, we apply zero-Dirichlet
   conditions, now. 
*/
template <int dim>
void
InterfaceProblem<dim>::set_newton_bc ()
{
  std::vector<bool> component_mask (1, true);
  
  //As we set zero Dirichlet data, we can use the standard function
  //from the deal.ii library  
  VectorTools::interpolate_boundary_values (dof_handler,
					    0,
					    ZeroFunction<dim>(1),                       
					    constraints,
					    component_mask); 
}  


/**
   Solution of the linear system of equations. 
   The choice is made in the parameter file. 
   We have taken from deal.II three methods: direct (UMFPACK), Jacobi, SSOR.
   Moreover, we also make possible use of preconditioners.
   Even that these methods are standard in deal.II, one novelty 
   in this work (paper and code) in comparison to FreiRichter2014 
   is to use iterative solvers for the locally modified FEM method.
   The behavior of the solvers is carefully studied in the accompanying paper.
*/
template <int dim>
void 
InterfaceProblem<dim>::solve () 
{
  timer.enter_section("Solve linear system.");

  prm.enter_subsection("Solver parameters");  
  bool_use_direct_solver = prm.get_bool("Direct linear solver");
  prm.leave_subsection();


  if (bool_use_direct_solver)
    {
      Vector<double> sol, rhs;    
      sol = newton_update;    
      rhs = system_rhs;
      
      SparseDirectUMFPACK A_direct;
      A_direct.factorize(system_matrix);     
      A_direct.vmult(sol,rhs); 
      linear_iterations = 0;

      newton_update = sol;
    }
  else 
    {
      // Use a CG solver with Jacobi/SSOR preconditioning
      prm.enter_subsection("Solver parameters");
      unsigned int max_linear_iter = prm.get_integer("Max linear iter");
      double tol_linear_solver = prm.get_double("Tol linear solver");
      double omega_prec = prm.get_double("Omega prec");

      // Types: None, Jacobi, SSOR
      std::string preconditioner_type = prm.get("Preconditioner type");
      
      prm.leave_subsection();
      
      
      SolverControl solver_control (max_linear_iter, 
				    tol_linear_solver);
      
      SolverCG<> solver (solver_control);
  

      if (preconditioner_type == "none")
	{
	  solver.solve (system_matrix, newton_update, system_rhs,
			PreconditionIdentity ());

	}
      else if (preconditioner_type == "jacobi")
	{
	  PreconditionJacobi<SparseMatrix<double> >  preconditioner;
	  preconditioner.initialize(system_matrix);
	  solver.solve (system_matrix, newton_update, system_rhs,
			preconditioner);

	}
      else if (preconditioner_type == "ssor")
	{
	  PreconditionSSOR<> preconditioner;
	  preconditioner.initialize(system_matrix, omega_prec);
	  solver.solve (system_matrix, newton_update, system_rhs,
			preconditioner);

	}
      else 
	{
	  std::cerr << "No such preconditioner" << std::endl;
	  abort();
	}
    
      
      linear_iterations = solver_control.last_step();
    }
  
  constraints.distribute (newton_update);
  timer.exit_section();
}


/**
   The Newton solver, which is more or less copy and paste, from 
   T. Wick; ANS, Vol. 1 (2013), pp. 1-19,
   doi https://doi.org/10.11588/ans.2013.1.10305

   The Newton solver is not necessary in this program,
   but kept in order to allow easily for nonlinear future extensions.
*/ 
template <int dim>
void InterfaceProblem<dim>::newton_iteration () 
					       
{ 
  // NewIter = No of Newton iterations
  // LS = line search
  // Newt. = Newton
  // Res = Residual
  // Reduct = Reduction
  // BuiMat. = Build Jacobian matrix
  // Time = CPU per Newton step
  std::cout << "NewtIt.\t" << "Newt.Res.\t" << "Newt.Reduct\t"
	    << "BuiMat\t" << "LSIter\t" 
	    << "LinIter\t" << "Time" << std::endl;


  Timer timer_newton;
  // Very low number because problem is linear 
  // and Newton should converge in one step.
  const unsigned int max_no_newton_steps  = 3;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1; 
 
  // Line search parameters
  unsigned int line_search_step;
  const unsigned int  max_no_line_search_steps = 1;
  const double line_search_damping = 0.6;
  double new_newton_residuum;
  
  // Application of the boundary conditions of the 
  // initial Newton guess:
  set_initial_bc ();
  assemble_system_rhs();

  double newton_residuum = system_rhs.linfty_norm(); 
  double old_newton_residuum= newton_residuum;
  unsigned int newton_step = 1;
  max_obtained_no_newton_steps = newton_step;

  if (newton_residuum < lower_bound_newton_residuum)
    {
      std::cout << '\t' 
		<< std::scientific 
		<< newton_residuum 
		<< std::endl;     
    }
  
  while (newton_residuum > lower_bound_newton_residuum &&
	 newton_step < max_no_newton_steps)
    {
      timer_newton.start();
      old_newton_residuum = newton_residuum;
      
      assemble_system_rhs();
      newton_residuum = system_rhs.linfty_norm();

      if (newton_residuum < lower_bound_newton_residuum)
	{
	  max_obtained_no_newton_steps = newton_step - 1;
	  std::cout << '\t' 
		    << std::scientific 
		    << newton_residuum << std::endl;
	  break;
	}
  
      // Simplied Newton steps when 
      // previous matrix is still good enough.
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
	assemble_system_matrix ();	

      // Solve Ax = b
      solve ();	  
        
      line_search_step = 0;	  
      for ( ; 
	    line_search_step < max_no_line_search_steps; 
	    ++line_search_step)
	{	     					 
	  solution += newton_update;
	  
	  assemble_system_rhs ();			
	  new_newton_residuum = system_rhs.linfty_norm();
	  
	  if (new_newton_residuum < newton_residuum)
	      break;
	  else 	  
	    solution -= newton_update;
	  
	  newton_update *= line_search_damping;
	}	   
     
      timer_newton.stop();
      
      // Terminal output with current user information 
      // of the solver.
      std::cout << std::setprecision(5) <<newton_step << '\t' 
		<< std::scientific << newton_residuum << '\t'
		<< std::scientific << newton_residuum/old_newton_residuum  <<'\t' ;
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
	std::cout << "r" << '\t' ;
      else 
	std::cout << " " << '\t' ;
      std::cout << line_search_step  << '\t' 
		<< linear_iterations << '\t'
		<< std::scientific << timer_newton ()
		<< std::endl;


      // Updates
      timer_newton.reset();
      newton_step++;      
    }
}


/**
  Having solved the problem, we compute now the L2 norm and H1 seminorm errors
  in order to validate our code.
*/
template<int dim>
void InterfaceProblem<dim>::compute_functional_values(bool convergence_rates)
{
 
  ManufacturedSolution<dim> exact_u(visc_1,visc_2,_yoffset);
  Vector<float> error_L2 (triangulation.n_active_cells());
  Vector<float> error_H1 (triangulation.n_active_cells());
  Vector<float> error_H1_semi (triangulation.n_active_cells());

  ComponentSelectFunction<dim> value_select (0, 1);

  // L2 error
  std::string norm_string = "L2";
  lmfe.integrate_difference_norms (dof_handler,
				   fe,
				   solution,
				   exact_u,
				   error_L2,
				   &value_select,
				   norm_string); 
  
  // Then, we want to get the global L2 norm. 
  // This can be obtained by summing the squares of the norms on each cell, 
  // and taking the square root of that value. 
  // This is equivalent to taking the l2 (lower case l) 
  // norm of the vector of norms on each cell:
  
  double local_error_L2 = error_L2.l2_norm();
 
  // H1 semi norm error
  norm_string = "H1_semi";
  lmfe.integrate_difference_norms (dof_handler,
				   fe,
				   solution,
				   exact_u,
				   error_H1_semi,	      
				   &value_select,
				   norm_string); 
  
  double local_error_H1_semi = error_H1_semi.l2_norm();
  
  // Compute convergence rate
  // r = log(e_1/e_2) / log(h_1/h_2)
  
  double conv_rate_L2, conv_rate_H1;
  
  if (convergence_rates) {
    conv_rate_L2 = std::log(old_local_error_L2/local_error_L2) / std::log(old_min_cell_diameter/min_cell_diameter);
    conv_rate_H1 = std::log(old_local_error_H1_semi/local_error_H1_semi) / std::log(old_min_cell_diameter/min_cell_diameter);
  }
 	
  std::cout << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::setprecision(2)<< "Error: L2 : " << local_error_L2 
	    << "  H1_semi : " << local_error_H1_semi;
  if (convergence_rates) {
    std::cout  << "  CRL2: " << conv_rate_L2
	       << "  CRH1: " << conv_rate_H1;
  }
  std::cout  << "  h: "  << min_cell_diameter << std::endl;
  
  
  old_min_cell_diameter = min_cell_diameter;
  old_local_error_L2 = local_error_L2;
  old_local_error_H1_semi = local_error_H1_semi;
  
}


/**
   The function run is standard in many deal.II tutorials and is called 
   from the main routine and specified the workflow
   of the program. Two numerical examples are implemented: In test case 1, a 
   solution to the model problem is computed on different meshes, obtained by global
   refinement. In test case 2, the interface is moved gradually in vertical direction
   in order to study the dependence on the position of the interface and the 
   resulting anisotropies of the mesh
*/
template <int dim>
void InterfaceProblem<dim>::run () 
{ 
  
  // Initializations (self-explaining!)
  set_runtime_parameters();
  setup_system();
  lmfe.initialize_quadrature();
  
  //Memorize initial solution  
  Vector<double> initial_solution = solution;
  std::cout << std::endl;
  
  if (test_case == 1)
    {
      for (unsigned int cycle=0; cycle<max_no_refinement_cycles; ++cycle)
	{
	  std::cout << "\n==========================================" 
		    << "============================================" 
		    << std::endl; 
	  std::cout << "Refinement cycle " << cycle << ':' << std::endl;
	  
	  // Take as starting (Newton) solution the same initial 
	  // solution on all meshes in order to study the asymptotical
	  // iteration numbers for the CG method
	  if (cycle != 0)
	    {
	      Vector<double> tmp_solution;
	      tmp_solution = initial_solution;
	      
	      SolutionTransfer<dim, Vector<double> > solution_transfer (dof_handler);
	      solution_transfer.prepare_for_coarsening_and_refinement(tmp_solution);
	      
	      triangulation.refine_global (1);
	      setup_system();
	      solution_transfer.interpolate(tmp_solution, solution); 
	      initial_solution=solution;
	    }
	  
	  
	  lmfe.set_material_ids (dof_handler, triangulation);
	  // Solve system with Newton solver
	  // (should converge in one step as the problem is linear)	
	  newton_iteration (); 
	  
	  // Compute functional values (error norms)
	  compute_functional_values(true);
	  
	  // Write solutions as *.vtk file
	  lmfe.plot_vtk (dof_handler,fe,solution,cycle,output_basis_filename);
	  
	  
	} // end refinement loop
      
    } 
  else if (test_case == 2)
    {
      for (unsigned int i=0; i < N_testcase2; ++i)
	{
	  solution = initial_solution;
	  // Move y-position of circle at each step
	  _yoffset = (double)i / (double)N_testcase2 * min_cell_vertex_distance;
	  lmfe.LevelSetFunction()->set_y_offset (_yoffset);
	  
	  //Reset material_ids based on the new interface location
	  lmfe.set_material_ids (dof_handler, triangulation);
	  
	  std::cout << "\n==============================" 
		    << "=============================================" 
		    << std::endl; 
	  std::cout << "Step n: " << i << std::endl;
	  std::cout << "Position of the mid-point: (0," << _yoffset << ")" << std::endl;
	  std::cout << std::endl;
	  
	  // Solve system with Newton solver
	  newton_iteration (); 
	  
	  // Compute functional values (error norms)
	  std::cout << std::endl;
	  compute_functional_values(false);
	  
	  // Additional output for better display of linear solver iterations
	  std::cout << "------------------" << std::endl;
	  std::cout << "Step (n,y,LinIter): " << i << "  " 
		    << _yoffset << "  " 
		    << linear_iterations << std::endl;
	  
	  
	  // Write solutions as *.vtk file
	  lmfe.plot_vtk (dof_handler,fe,solution,i,output_basis_filename);
	  
	}			
    } // end test_case 2  
}


/**
   The main function looks almost the same
   as in many deal.II tuturial steps. 
*/
int main (int argc, char *argv[]) 
{
  try
    {
      deallog.depth_console (0);
      
      ParameterHandler prm;
      ParameterReader param (prm);
      
      if (argc>1)
        param.read_parameters(argv[1]);
      else
        param.read_parameters("parameters_test_case_1.prm");
      
      // We base our implementation on the structure of a Q2 
      // finite element. Therefore, we set the degree to 2 here
      unsigned int patch_degree = 2;
      InterfaceProblem<2> ip_problem(patch_degree, prm);
      ip_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      
      return 1;
    }
  catch (...) 
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}




