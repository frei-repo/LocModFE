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

/**
    In this file (and the corresponding .cc file), we provide the classes LocModFEValues and LocModFE, that contain
    all functions that are specific for the locally modified finite element discretisation. 

    The class LocModFEValues extends 
    the FEValues class in deal.ii. In this class the values of the basis functions and their gradients
    (in deal.ii ``shape functions'') as well as the derivatives of the map $\hat{T}_P$ are evaluated in quadrature 
    points on the reference patch, depending on the reference patch type (P_0,...,P_3) and the boolean parameter 
    _hierarchical, which specifies if a hierarchical basis is to be used (which is necessary to get condition 
    numbers that are bounded independent of the position of the interface.

    In the class {LocModFE}, we check if patches are cut
    and in which sub-domains they are (function {set_material_ids}), define the type of the cut (configurations A,...,D), 
    the reference patch type (P_0,...,P_3) and the local mappings T_P (function {init_FEM}).
    Moreover, we initialise the respective quadrature formulas depending on the reference patches (function {compute_quadrature}), 
    provide functions to compute norm errors (function {integrate_difference_norms}), to set Dirichlet boundary values in cut patches 
    (function {interpolate_boundary_values}) and to visualise the solution (function {plot_vtk}).
*/


// Include files
//--------------
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h> 
#include <deal.II/base/parameter_handler.h> 

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
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

// Include application-specific problem file
#include "problem.h"


// Important global namespaces from deal.II and 
// C++				
using namespace dealii;
using namespace std;



/**
   The class LocModFEValues replaces the FEValues class in 
   deal.II. In this class the values of the shape 
   functions and their gradients as well as the derivatives of 
   the map T_P are evaluated in quadrature points on the reference patch
*/
template <int dim, int spacedim=dim>
class LocModFEValues : public FEValues<dim,spacedim>
{

 protected:

 //Standard variables of the FEValues class
 Table<2,double>           _shape_values;
 Table<2,Tensor<1,dim> >   _shape_gradients, _shape_gradients_tilde;
 Table<2,double>           _shape_values_standard;
 Table<2,Tensor<1,dim> >   _shape_gradients_tilde_standard;
 
 const unsigned int        _dofs_per_cell;
 const unsigned int        _n_q_points;

 UpdateFlags               _update_flags;
 
 //Patch type P_0,...,P_3
 unsigned int _femtype;
 
 //One of the four quadrature formulas corresponding to the patch type
 Quadrature<dim> _quadrature;
 
 //Parametrisation of the map T_P
 FullMatrix<double> _M;

 //Defines if a hierarchical basis is to be used
 bool _hierarchical;
 
 public:
 
 //Constructor
 LocModFEValues(const FiniteElement<dim, spacedim> &fe,
		Quadrature<dim> quadrature,
		bool hierarchical,
		UpdateFlags update_flags);

 //Set LocModFE-specific local variables 
 void SetFemtypeAndQuadrature(Quadrature<dim> quadrature,
			      unsigned int femtype, FullMatrix<double>& M); 
 
 //Computes ShapeValues and Gradients for the non-hierarchical basis
 //These are needed in some places also in the hierarchical case
 void ComputeStandardReferenceShapeValuesAndGradients(std::vector<double>& shape_values, 
	    				    std::vector<Tensor<1,dim> >& shape_gradients_tilde, 
    	       				std::vector<double>& tmp_shape_values, 
						    std::vector<Tensor<1,dim> >& tmp_shape_gradients_tilde, double s_x, double s_y);
 
 //This is the main function, that will be called from outside this class. All the shape values and gradients
 //as well as the derivatives of the domain map (incl the determinant J) are computed in all quadrature points
 void reinit(std::vector<double> &J);
 
 //Get specific values in quadrature point
 double shape_value(const unsigned int function_no,
		    const unsigned int point_no);
 double shape_value_standard(const unsigned int function_no,
			     const unsigned int point_no);
 Tensor<1,dim> shape_grad(const unsigned int function_no,
                           const unsigned int point_no);
 
 //Get the whole vector of values in quadrature points
 void get_function_values(const typename DoFHandler<dim,spacedim>::cell_iterator &cell,
			  const Vector<double>& fe_function, std::vector<Vector<double> > &values);
 
 void get_function_gradients(const typename DoFHandler<dim,spacedim>::cell_iterator &cell,
			     const Vector<double>& fe_function, std::vector<std::vector<Tensor<1,dim> > > &values);
 
 
 //The following auxiliary functions define the values of the shape funcions and their
 //derivatives on the reference patch: L is needed in quadrilateral patches
 //E_i in triangular patches
  double L  (int i,double x) const
 {
   if (i==0) { return (x<0.5)?(1.0-2.0*x):0.0; }
   if (i==1) { return (x<0.5)?(2.0*x):(2.0-2.0*x); }
   if (i==2) { return (x<0.5)?0.0:(2.0*x-1.0); }
   abort();
 }

 double DL  (int i,double x) const
 {
    if (i==0) { return (x<0.5)?-2.0:0.0; }
    if (i==1) { return (x<0.5)?2.0:-2.0; }
    if (i==2) { return (x<0.5)?0.0:2.0; }
    abort();
 }
 
 double L2h  (int i,double x) const
 {
    if (i==0) { return 1.0-x; }
    if (i==1) { cout << " L2h should not be called for i=1" << endl;}
    if (i==2) { return x; }
    abort();
  }
 
 double DL2h  (int i,double /*x*/) const
 {
   if (i==0) { return -1.; }
   if (i==1) { cout << " DL2h should not be called for i=1" << endl;}
   if (i==2) { return 1.; }
   abort();
 }
 
 /* E0 calculates the basis functions on the reference element for
    two triangles that form a quadrilateral in the following way
    ------
    |\   |
    | \  |
    |  \ |
    |   \|
    ------*/
 
 
 double E0(int ix, int iy, double x, double y)const
 {
   assert((ix>=0)&&(ix<2)&&(iy>=0)&&(iy<2));
   if (x+y<1.0)
     {
       if      ((ix==0)&&(iy==0)) return 1.0-x-y;
       else if ((ix==1)&&(iy==0)) return x;
       else if ((ix==0)&&(iy==1)) return y;
       else if ((ix==1)&&(iy==1)) return 0.0;
	  }
   else
     {
       if      ((ix==0)&&(iy==0)) return 0.0;
       else if ((ix==1)&&(iy==0)) return 1-y;
       else if ((ix==0)&&(iy==1)) return 1-x;
       else if ((ix==1)&&(iy==1)) return x+y-1;
     }
   abort();
 }
 
 
 double E0x(int ix, int iy, double x, double y)const
 {
   assert((ix>=0)&&(ix<2)&&(iy>=0)&&(iy<2));
   if (x+y<1.0)
     {
       if      ((ix==0)&&(iy==0)) return -2.0;
       else if ((ix==1)&&(iy==0)) return 2.0;
       else if ((ix==0)&&(iy==1)) return 0.0;
       else if ((ix==1)&&(iy==1)) return 0.0;
     }
   else
     {
       if      ((ix==0)&&(iy==0)) return 0.0;
       else if ((ix==1)&&(iy==0)) return 0.0;
       else if ((ix==0)&&(iy==1)) return -2.0;
       else if ((ix==1)&&(iy==1)) return 2.0;
     }
   abort();
 }
 double E0y(int ix, int iy, double x, double y) const
 {
   assert((ix>=0)&&(ix<2)&&(iy>=0)&&(iy<2));
   if (x+y<1.0)
     {
       if      ((ix==0)&&(iy==0)) return -2.0;
       else if ((ix==1)&&(iy==0)) return 0.0;
       else if ((ix==0)&&(iy==1)) return 2.0;
       else if ((ix==1)&&(iy==1)) return 0.0;
     }
   else
     {
       if      ((ix==0)&&(iy==0)) return 0.0;
       else if ((ix==1)&&(iy==0)) return -2.0;
       else if ((ix==0)&&(iy==1)) return 0.0;
       else if ((ix==1)&&(iy==1)) return 2.0;
     }
   abort();
 }
 
 /* E1 calculates the basis functions on the reference element for
    two triangles that form a quadrilateral in the following way
    ------
    |   /|
    |  / |
    | /  |
    |/   |
    ------*/
 
 double E1(int ix, int iy, double x, double y) const
 {
   assert((ix>=0)&&(ix<2)&&(iy>=0)&&(iy<2));
   if (x<y)
     {
       if      ((ix==0)&&(iy==0)) return 1.0-y;
       else if ((ix==1)&&(iy==0)) return 0;
       else if ((ix==0)&&(iy==1)) return y-x;
       else if ((ix==1)&&(iy==1)) return x;
     }
   else
     {
       if      ((ix==0)&&(iy==0)) return 1.0-x;
       else if ((ix==1)&&(iy==0)) return x-y;
       else if ((ix==0)&&(iy==1)) return 0;
       else if ((ix==1)&&(iy==1)) return y;
     }
   abort();
 }
 double E1x(int ix, int iy, double x, double y)const
 {
   assert((ix>=0)&&(ix<2)&&(iy>=0)&&(iy<2));
   if (x<y)
     {
       if      ((ix==0)&&(iy==0)) return 0.0;
       else if ((ix==1)&&(iy==0)) return 0;
       else if ((ix==0)&&(iy==1)) return -2.0;
       else if ((ix==1)&&(iy==1)) return 2.0;
     }
   else
     {
       if      ((ix==0)&&(iy==0)) return -2.0;
       else if ((ix==1)&&(iy==0)) return 2.0;
       else if ((ix==0)&&(iy==1)) return 0;
       else if ((ix==1)&&(iy==1)) return 0.0;
     }
   abort();
 }
 
 double E1y(int ix, int iy, double x, double y) const
 {
   assert((ix>=0)&&(ix<2)&&(iy>=0)&&(iy<2));
   if (x<y)
     {
       if      ((ix==0)&&(iy==0)) return -2.0;
       else if ((ix==1)&&(iy==0)) return 0;
       else if ((ix==0)&&(iy==1)) return 2.0;
       else if ((ix==1)&&(iy==1)) return 0.0;
     }
   else
     {
       if      ((ix==0)&&(iy==0)) return 0.0;
       else if ((ix==1)&&(iy==0)) return -2.0;
       else if ((ix==0)&&(iy==1)) return 0;
       else if ((ix==1)&&(iy==1)) return 2.0;
     }
   abort();
 }
  
};





/**
 * The class {LocModFE} provides all the remaining functionalities that are needed for the LocModFE discretisation.
 * In particular, we check if patches are cut and in which sub-domains they are (function {set_material_ids}), 
 * define the type of the cut (configurations A,...,D),  the reference patch type (P_0,...,P_3)
 * and the local mappings $T_P$ (function {init_FEM}). Moreover, we initialise the respective quadrature formulas 
 * depending on the reference patches (function {compute_quadrature}), provide functions to compute norm errors 
 * (function {integrate_difference_norms}), to set Dirichlet boundary values in cut patches 
 * (function {interpolate_boundary_values}) and to visualise the solution (function {plot_vtk}).
*/
template <int dim>
class LocModFE
{

private:
  //Four quadrature formulas needed, depending on reference patch type
  Quadrature<dim>*  Quadrature0;
  Quadrature<dim>*  Quadrature1;
  Quadrature<dim>*  Quadrature2;
  Quadrature<dim>*  Quadrature3;

  //Pointer to LevelSet Function  
  LevelSet<dim>* chi;
  
  //List of patch (cell_colors) and vertex colors (node_colors)
  //based on their domain affiliation: -1 or 1 fo vertices, 
  //-1,0,1 for patches (0 specifiying an interface patch)
  Vector<float> cell_colors, node_colors;

  //Specifies of a hierarchical basis is used
  bool _hierarchical;


public:
  //Constructor and destructor
  LocModFE();  
  ~LocModFE() {};
  
  //Set _hierarchical variable, that specifies if a hierarchical basis is to be used
  void set_bool_hierarchical (bool hierarchical) 
  {
    _hierarchical = hierarchical;
  }
 
  //Fina the point on the line connecting x1 and x2, where the interface intersects
  double find_cut (Point<dim> x1, Point<dim> x2);

  //Computes cut position in terms of s and t of the lines (x0,y0) + s(x1-x0, y1-y0) 
  //and (x2, y2) + t (x3-x2, y3-y2)
  void FindCutOfTwoLines(double &s, double& t, double x0, double y0, double x1, double y1, 
				double x2, double y2, double x3, double y3);

  //Set colors for nodes and cells: -1 and 1 for the sub-domains
  //0 for interface 
  void set_material_ids (const DoFHandler<dim> &dof_handler,
			 const Triangulation<dim> &triangulation);

 
  // Initialize the lcoal variables specific to LocModFE
  // Specifically the different fem_types are computed depending
  // on the cut of the interface, the local map T_P(parametrised in M)
  // and the discrete level set function (parametrised by its values in the 
  // vertices. A list NodesAtInterface is created that contains all local
  // indices at the interface 
  void init_FEM(const typename DoFHandler<dim>::active_cell_iterator &cell,
		unsigned int cell_counter, 
		FullMatrix<double> &M,
		const unsigned int   dofs_per_cell,
		unsigned int &femtype_int,
	    std::vector<double> &LocalDiscChi,
        std::vector<int>& NodesAtInterface);

  //This function initialises the four quadrature formulas, that are needed depending on
  //the reference patch type (fem_type)
  void initialize_quadrature();

  //Then, one of the four formulas can be returned depending on the femtype 
  Quadrature<dim> compute_quadrature (int femtype);

  //Get physical position of integration points, needed to evaluate data 
  //(for example a right-hand side f) 
  void ComputePhysicalIntegrationPoint(Point<dim>& IntPoint, 
				       LocModFEValues<dim>& fe_values, 
				       FullMatrix<double>& M, 
				       int dofs_per_cell, 
				       int q);

  //Local values of the discrete level set function chi_h, parametrised
  //by the 9 values in the vertices
  void ComputeLocalDiscChi(double &ChiValue, 
			   int q, 
			   LocModFEValues<dim>& fe_values, 
			   int dofs_per_cell, 
			   std::vector<double> LocalDiscChi);

  // Modified output function in order to plot
  // the physical cut-cells including the cut-interfaces.
  void plot_vtk (const DoFHandler<dim> &dof_handler,
		 const FiniteElement<dim> &fe,
		 const Vector<double>& solution,
		 const unsigned int refinement_cycle,
		 const std::string output_basis_filename);
  
  //Evaluate error norms
  void integrate_difference_norms 
    (const DoFHandler<dim> &dof,
     const FiniteElement<dim> &fe,
     const Vector<double> &fe_function,
     ManufacturedSolution<dim> exact_solution,
     Vector<float> &difference,
     const Function<dim> *weight=0,
     const std::string norm_string="L2"
     );
  
  //The function interpolate_boundary_values has to be modified compared to the standard implementation 
  //in deal.ii due to (possible) moved vertices at the boundary and due to the hierarchical basis
  //(whose basis fucntions are non-primitive)
  void interpolate_boundary_values(const DoFHandler<dim> &dof,
				   const types::boundary_id boundary_component,
				   const Function<dim,double> &boundary_function,
				   std::map<types::global_dof_index,double> &boundary_values);

  //Get LevelSet-Function
  LevelSet<dim>* LevelSetFunction() {
		return chi;	
  }

};

