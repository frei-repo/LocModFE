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
   In this file, we provide the implementation of the classes LocModFEValues and LocModFE, that contain
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




// Include header file
//--------------
#include "locmodfe.h"


/****************Class LocModFEValues*****************************************/

//Constructor
template<int dim,int spacedim>
LocModFEValues<dim,spacedim>::LocModFEValues(const FiniteElement<dim, spacedim> &fe,
					     Quadrature<dim> quadrature, 
					     bool hierarchical, 
					     UpdateFlags update_flags)
  : FEValues<dim,spacedim>(fe,quadrature,update_flags | update_quadrature_points),
    _dofs_per_cell(fe.dofs_per_cell),
    _n_q_points(quadrature.size()),
    _update_flags(update_flags)
{
  _quadrature = quadrature;
  _hierarchical = hierarchical;
}


//Set LocModFE-specific variables
template<int dim,int spacedim>
void LocModFEValues<dim,spacedim>::SetFemtypeAndQuadrature(Quadrature<dim> quadrature,
							   unsigned int femtype, FullMatrix<double>& M) 
{
  
  _femtype = femtype;
  _quadrature = quadrature;
  _M = M;
 
}

//Computes the standard shape functions and gradients locally on the reference patch
//consisting of 4 sub-quadrilaterals (femtype 0) or 8 sub-triangles (femtype 1, 11, 2, 3)
template <int dim,int spacedim>
void LocModFEValues<dim,spacedim>
::ComputeStandardReferenceShapeValuesAndGradients(std::vector<double>& shape_values, 
						  std::vector<Tensor<1,dim> >& shape_gradients_tilde, 
						  std::vector<double>& tmp_shape_values, 
						  std::vector<Tensor<1,dim> >& tmp_shape_gradients_tilde, 
						  double s_x, 
						  double s_y)
{
  
  for (unsigned int i=0; i<_dofs_per_cell; i++) {
    tmp_shape_values[i]=0.;
    tmp_shape_gradients_tilde[i]=0.;
  }
  
  
  //Numbering: ComputeStandardReferenceShapeValuesAndGradients uses a different numbering
  //of the vertices and corresponding shape functions, which makes the assembly of the nine
  //shape functions easier. At the end, we renumber the vectors to the local deal.II numbering:
  //ComputeSt... deal.II
  //3*k_y + k_x  
  //
  // 6--7--8     2--7--3
  // |  |  |     |  |  |
  // 3--4--5     4--8--5
  // |  |  |     |  |  |
  // 0--1--2     0--6--1
  
  //Femtype 0 means piecewise bilinear elements
  if (_femtype==0) 
    {
      for (int x=0;x<3;++x)
        for (int y=0;y<3;++y)
        {
           //Tensor product structure
           //L is a linear function on a line, depending on the first index
           //it is one in one of the three vertices on the line, zero in the others
           tmp_shape_values[3*y+x] = L(x,s_x) * L(y,s_y);
  
           tmp_shape_gradients_tilde[3*y+x][0] =  DL(x,s_x)*L(y,s_y);
           tmp_shape_gradients_tilde[3*y+x][1] =  L(x,s_x)*DL(y,s_y);
    
        }
    }
  else if ((_femtype==1)||(_femtype==11))
    {
      if ((s_x<=0.5)&&(s_y<=0.5))
      {
         for (int x=0;x<2;++x)
            for (int y=0;y<2;++y)
            {
                /*E0 and E1 set shape functions on two triangles that form a 
                  quadrilateral in the following way
                  E0 					E1
                  ------      ------
                  |\   |			|   /|
                  | \  |			|  / |
                  |  \ |			| /  |
                  |   \|			|/   |
                  ------ 		  -----*/
                
                tmp_shape_values[3*y+x] = E1(x,y, 2.0*s_x, 2.0* s_y);
                tmp_shape_gradients_tilde[3*y+x][0] = E1x(x,y,2.0*s_x, 2.0*s_y);
                tmp_shape_gradients_tilde[3*y+x][1] = E1y(x,y,2.0* s_x,2.0* s_y);
            }
      }
      if ((s_x>0.5)&&(s_y>0.5))
      {
         for (int x=0;x<2;++x)
            for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*(y+1)+x+1] = E1(x,y, 2.0* s_x-1.0, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x+1][0] = E1x(x,y, 2.0* s_x-1.0, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x+1][1] = E1y(x,y, 2.0* s_x-1.0, 2.0*s_y-1.0);
            }
        }
      
      if ((s_x>0.5)&&(s_y<=0.5))
        {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*y+x+1] = E0(x,y, 2.0* s_x-1.0, 2.0*s_y);
               tmp_shape_gradients_tilde[3*y+x+1][0] = E0x(x,y, 2.0* s_x-1.0, 2.0*s_y);
               tmp_shape_gradients_tilde[3*y+x+1][1] = E0y(x,y, 2.0* s_x-1.0, 2.0*s_y);
            }
        }
      if ((s_x<=0.5)&&(s_y>0.5))
      {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*(y+1)+x] = E0(x,y, 2.0* s_x, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x][0] = E0x(x,y, 2.0* s_x, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x][1] = E0y(x,y, 2.0* s_x, 2.0*s_y-1.0);
            }
      }
      
      
    }
  else if (_femtype==2)
    {
      if ((s_x<=0.5)&&(s_y<=0.5))
      {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*y+x] = E0(x,y, 2.0*s_x, 2.0* s_y);
               tmp_shape_gradients_tilde[3*y+x][0] = E0x(x,y,2.0*s_x, 2.0*s_y);
               tmp_shape_gradients_tilde[3*y+x][1] = E0y(x,y,2.0* s_x,2.0* s_y);
            }
      }
      if ((s_x>0.5)&&(s_y>0.5))
      {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*(y+1)+x+1] = E0(x,y, 2.0* s_x-1.0, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x+1][0] = E0x(x,y, 2.0* s_x-1.0, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x+1][1] = E0y(x,y, 2.0* s_x-1.0, 2.0*s_y-1.0);
            }
      }
      
      if ((s_x>0.5)&&(s_y<=0.5))
      {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*y+x+1] = E0(x,y, 2.0* s_x-1.0, 2.0*s_y);
               tmp_shape_gradients_tilde[3*y+x+1][0] = E0x(x,y, 2.0* s_x-1.0, 2.0*s_y);
               tmp_shape_gradients_tilde[3*y+x+1][1] = E0y(x,y, 2.0* s_x-1.0, 2.0*s_y);
            }
      }
      if ((s_x<=0.5)&&(s_y>0.5))
      {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*(y+1)+x] = E0(x,y, 2.0* s_x, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x][0] = E0x(x,y, 2.0* s_x, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x][1] = E0y(x,y, 2.0* s_x, 2.0*s_y-1.0);
            }
      }	  
      
      
    }
  else if (_femtype==3)
    {
      if ((s_x<=0.5)&&(s_y<=0.5))
      {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*y+x] = E1(x,y, 2.0*s_x, 2.0* s_y);
               tmp_shape_gradients_tilde[3*y+x][0] = E1x(x,y,2.0*s_x, 2.0*s_y);
               tmp_shape_gradients_tilde[3*y+x][1] = E1y(x,y,2.0* s_x,2.0* s_y);
            }
      }
      if ((s_x>0.5)&&(s_y>0.5))
      {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*(y+1)+x+1] = E1(x,y, 2.0* s_x-1.0, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x+1][0] = E1x(x,y, 2.0* s_x-1.0, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x+1][1] = E1y(x,y, 2.0* s_x-1.0, 2.0*s_y-1.0);
            }
      }
      
      if ((s_x>0.5)&&(s_y<=0.5))
      {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*y+x+1] = E1(x,y, 2.0* s_x-1.0, 2.0*s_y);
               tmp_shape_gradients_tilde[3*y+x+1][0] = E1x(x,y, 2.0* s_x-1.0, 2.0*s_y);
               tmp_shape_gradients_tilde[3*y+x+1][1] = E1y(x,y, 2.0* s_x-1.0, 2.0*s_y);
            }
      }
      if ((s_x<=0.5)&&(s_y>0.5))
      {
        for (int x=0;x<2;++x)
          for (int y=0;y<2;++y)
            {
               tmp_shape_values[3*(y+1)+x] = E1(x,y, 2.0* s_x, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x][0] = E1x(x,y, 2.0* s_x, 2.0*s_y-1.0);
               tmp_shape_gradients_tilde[3*(y+1)+x][1] = E1y(x,y, 2.0* s_x, 2.0*s_y-1.0);
            }
      }
      
      
    }
  
  //Now, we renumber the vectors to the local deal.II numbering
  //3*k_y + k_x  deal.II
  //
  // 6--7--8     2--7--3
  // |  |  |     |  |  |
  // 3--4--5     4--8--5
  // |  |  |     |  |  |
  // 0--1--2     0--6--1
  
  shape_values[0] = tmp_shape_values[0];
  shape_values[1] = tmp_shape_values[2];
  shape_values[2] = tmp_shape_values[6];
  shape_values[3] = tmp_shape_values[8];
  shape_values[4] = tmp_shape_values[3];
  shape_values[5] = tmp_shape_values[5];
  shape_values[6] = tmp_shape_values[1];
  shape_values[7] = tmp_shape_values[7];
  shape_values[8] = tmp_shape_values[4];
  
  for (int k=0; k<dim; k++) {
    shape_gradients_tilde[0][k] = tmp_shape_gradients_tilde[0][k];
    shape_gradients_tilde[1][k] = tmp_shape_gradients_tilde[2][k];
    shape_gradients_tilde[2][k] = tmp_shape_gradients_tilde[6][k];
    shape_gradients_tilde[3][k] = tmp_shape_gradients_tilde[8][k];
    shape_gradients_tilde[4][k] = tmp_shape_gradients_tilde[3][k];
    shape_gradients_tilde[5][k] = tmp_shape_gradients_tilde[5][k];
    shape_gradients_tilde[6][k] = tmp_shape_gradients_tilde[1][k];
    shape_gradients_tilde[7][k] = tmp_shape_gradients_tilde[7][k];
    shape_gradients_tilde[8][k] = tmp_shape_gradients_tilde[4][k];
  }
  
}

//This function reinits shape functions, gradients and cell weights
template <int dim,int spacedim>
void LocModFEValues<dim,spacedim>::reinit(std::vector<double> &J)
{
  LocModFE<dim> lmfe;
  
  _shape_values.reinit(_dofs_per_cell,_n_q_points);
  _shape_values_standard.reinit(_dofs_per_cell,_n_q_points);
  _shape_gradients.reinit(_dofs_per_cell,_n_q_points);
  _shape_gradients_tilde.reinit(_dofs_per_cell,_n_q_points);
  _shape_gradients_tilde_standard.reinit(_dofs_per_cell,_n_q_points);
  
  std::vector<double> tmp_shape_values(_dofs_per_cell);
  std::vector<Tensor<1,dim> > tmp_shape_gradients_tilde(_dofs_per_cell);
  std::vector<double> tmp_shape_values_standard(9);
  std::vector<Tensor<1,dim> > tmp_shape_gradients_tilde_standard(9);
  std::vector<double> shape_values(9);
  std::vector<Tensor<1,dim> > shape_gradients_tilde(9);
	
  J.resize(_n_q_points);
  
  Tensor<2,dim> F, F2h;
  Tensor<2,dim> F_inv_transpose, F2h_inv_transpose;
  FullMatrix<double> M2h(dim,_dofs_per_cell);
  
  Vector<double> res(dim);
  Vector<double> delta(dim);
  FullMatrix<double> GradT2h(dim, dim);
  std::vector<double> shape_values_Newton(9);
  std::vector<Tensor<1,dim> > shape_gradients_tilde_Newton(9);
  std::vector<double> tmp_shape_values_Newton(9);
  std::vector<Tensor<1,dim> > tmp_shape_gradients_tilde_Newton(9);
  
  
  for (unsigned int q_point=0; q_point<_n_q_points; ++q_point)
    {
      
      for (unsigned int i=0; i<_dofs_per_cell; i++) {
        tmp_shape_values_standard[i]=0.;
        tmp_shape_gradients_tilde_standard[i]=0.;
        tmp_shape_values[i]=0.;
        tmp_shape_gradients_tilde[i]=0.;
      }
      
      
      double s_x = _quadrature.point(q_point)[0];
      double s_y = _quadrature.point(q_point)[1];
      
      ComputeStandardReferenceShapeValuesAndGradients(shape_values, shape_gradients_tilde, 
						      tmp_shape_values, tmp_shape_gradients_tilde, s_x, s_y);
      
      
      for (unsigned int i=0; i<_dofs_per_cell; i++)
        {
           _shape_values(i,q_point)=shape_values[i];
           _shape_gradients_tilde(i,q_point)=shape_gradients_tilde[i];
        }
      
      F.clear();
      F2h.clear();
      
      //Mapping T(xhat) = sum_k (x_k phihat_k(x)), where hat(phi) is shape function 
      //on reference patch
      //To calculate the cell weights and the transformation of derivatives, we need the gradient F
      //of the mapping: F_ij(x) = d_i T_j(x) = sum_k (x_k^j d_i phihat_k(x))
      // The 2x9 matrix M contains entries M_jk= x_k^j, j=0,1, k=0,...,8
      
      for (int i=0;i<9;i++)
      {
        
        F[0][0] += _shape_gradients_tilde(i,q_point)[0] * _M[0][i];
        F[0][1] += _shape_gradients_tilde(i,q_point)[1] * _M[0][i];
        F[1][0] += _shape_gradients_tilde(i,q_point)[0] * _M[1][i];
        F[1][1] += _shape_gradients_tilde(i,q_point)[1] * _M[1][i];
      }
      
      F_inv_transpose.clear();
      F_inv_transpose = transpose(invert(F));
      
      
      //Grad phi = F^{-T} gradhat phihat
      for (int i=0; i<9;i++)
      {
        _shape_gradients(i,q_point) = F_inv_transpose *  _shape_gradients_tilde(i,q_point);
        
      }
      
      //Cell weights: Determinant of F (will be used for both the standard and the hierarchical ansatz
      J[q_point] = determinant(F);
      
      //The hierarchical basis is used to get a bound for the condition number of the system matrix:
      //O(h^-2) for elliptic interface problems, independent of the position of the interface
      if (_hierarchical) {
      //For the base functions in V_2h we need not only different shape functions on the 
      //reference patch, but also a different mapping T_2h, as a
      //function phi_2h(T_h) would have kinks within the large triangles 
      
      //Although the mapping T_2h is in principle defined only by the four
      //outer patch nodes, we define the mapping in the interior points also
      //for ease of implementation. The matrix M2h contains the position of 
      //the 9 nodes on the reference patch  
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          M2h[0][v] = _M[0][v];
          M2h[1][v] = _M[1][v];
        }
      
      M2h[0][4] = 0.5 * (_M[0][0] + _M[0][2]);
      M2h[1][4] = 0.5 * (_M[1][0] + _M[1][2]);
      
      M2h[0][5] = 0.5 * (_M[0][1] + _M[0][3]);
      M2h[1][5] = 0.5 * (_M[1][1] + _M[1][3]);
      
      M2h[0][6] = 0.5 * (_M[0][1] + _M[0][0]);
      M2h[1][6] = 0.5 * (_M[1][1] + _M[1][0]);
      
      M2h[0][7] = 0.5 * (_M[0][2] + _M[0][3]);
      M2h[1][7] = 0.5 * (_M[1][2] + _M[1][3]);
      
      //The position of the midpoint depends on the mapping T_2h, which depends on the
      //local hierarchical space 
      if (_femtype ==0) {
        //We place the midpoint to the position that is induced by the bilinear mapping 
        //m_p = T_2h(0.5, 0.5)
        M2h[0][8] = 0.25 * (_M[0][0] + _M[0][1] + _M[0][2] + _M[0][3]);
        M2h[1][8] = 0.25 * (_M[1][0] + _M[1][1] + _M[1][2] + _M[1][3]);
        
      } else if ((_femtype==1)||(_femtype==3)) { 
        
        M2h[0][8] = 0.5*_M[0][0] + 0.5*_M[0][3];
        M2h[1][8] = 0.5*_M[1][0] + 0.5*_M[1][3];
      } else {
        M2h[0][8] = 0.5*_M[0][1] + 0.5*_M[0][2];
        M2h[1][8] = 0.5*_M[1][1] + 0.5*_M[1][2];
        
      }
      
      
      //As we will need to integrate standard and hierarchical basis functions against each other,
      //and they rely on different mappings T_h: hatT to T, T_2h: hatT to T respectively, we have
      //to find the local quadrature point for T_2h that maps a point xhat to the physical point 
      //Th(q), i.e. we have to calculate for each integration point T_2h^{-1}(T_h(q))
      //This is done using a Newton method.
      Point<dim> MappedIP;   
      MappedIP[0]=0.;
      MappedIP[1]=0.;
      
      //First map a point q by Th
      for (unsigned int i=0; i<_dofs_per_cell; ++i) {
        MappedIP[0] += _M[0][i] * _shape_values(i,q_point);
        MappedIP[1] += _M[1][i] * _shape_values(i,q_point);
      }
      
      //Now, we calculate the inverse mapping
      double s_xh=0.5; 
      double s_yh=0.5;
      
      //If the cell is Cartesian, the inverse mapping is easily calculated
      //by the following formula
      //In the general case this position serves at initial point for a
      //Newton iteration
      if ((fabs(_M[0][1]-_M[0][0])>1.e-8)&&(fabs(_M[1][2]-_M[1][0])>1.e-8)) {
        s_xh=(MappedIP[0]- _M[0][0]) / (_M[0][1]-_M[0][0]);
        s_yh=(MappedIP[1]- _M[1][0]) / (_M[1][2]-_M[1][0]);
      }
      
      // Newton method to solve MappedIP - T_2h(xhat, yhat) = 0
      //delta_k = -(GradT2h)^(-1) (x-T_2h(xhat_k)) ,  x_k+1 = x_k + delta_k
      //Calculate res = x-T_2h(xhat)
      //First we have to evaluate the shape functions for the mapping T2h in
      //the Newton point s_xh, s_yh
      ComputeStandardReferenceShapeValuesAndGradients(shape_values_Newton, 
							shape_gradients_tilde_Newton, tmp_shape_values_Newton, 
							tmp_shape_gradients_tilde_Newton,s_xh, s_yh);
	
      for (unsigned int i=0; i<dim; i++) {
        res[i] = MappedIP[i];
        for (unsigned int k=0; k<_dofs_per_cell; ++k) {
          res[i] -= M2h[i][k]*shape_values_Newton[k];
        }
      }
      
      int newton_iter=0;
      //Newton
      while ((res.l2_norm()>1.e-8)&&(newton_iter<50)) {
        
        //Calculate gradient GradT2h for Newton step
        GradT2h=0.;
        for (unsigned int i=0; i<dim; i++)
          for (unsigned int j=0; j<dim; j++)
            for (unsigned int k=0; k<_dofs_per_cell; k++)
                GradT2h[i][j] -= M2h[i][k]*shape_gradients_tilde_Newton[k][j];
        
        
        //Inverse gradient for Newton;
        GradT2h.gauss_jordan();
        
        //Newton step
        const Vector<double> res2 = res;
        
        GradT2h.vmult(delta, res2, false);
        
        s_xh -= delta[0];
        s_yh -= delta[1];
        
        //Evaluate the shape functions for the mapping T2h in
        //the new Newton point s_xh, s_yh to compute new residual
        ComputeStandardReferenceShapeValuesAndGradients(shape_values_Newton, 
							  shape_gradients_tilde_Newton, tmp_shape_values_Newton, 
							  tmp_shape_gradients_tilde_Newton, s_xh, s_yh);
	  
        for (unsigned int i=0; i<dim; i++) {
          res[i] = MappedIP[i];
          for (unsigned int k=0; k<_dofs_per_cell; ++k) {
            res[i] -= M2h[i][k]*shape_values_Newton[k];
          }
        }
        
        newton_iter++;
      }
      		
      //Now that we have the right integration point s_hx, s_yh for the hierarchical basis functions,
      //we can compute the shape functions
      if (_femtype==0) 
        {
          for (int x=0;x<3;++x)
            for (int y=0;y<3;++y)
            {
              
              
              //If hierarchical basis, basis functions for (x,y)=(0,0), (0,2), (2,0), (2,2) from V2h 
              if ((x%2==1)||(y%2==1)) {
                tmp_shape_values[3*y+x] = L(x,s_x) * L(y,s_y);
                tmp_shape_values_standard[3*y+x] = L(x,s_x) * L(y,s_y);
                
                tmp_shape_gradients_tilde_standard[3*y+x][0] =  DL(x,s_x)*L(y,s_y);
                tmp_shape_gradients_tilde_standard[3*y+x][1] =  L(x,s_x)*DL(y,s_y);
                tmp_shape_gradients_tilde[3*y+x][0] =  DL(x,s_x)*L(y,s_y);
                tmp_shape_gradients_tilde[3*y+x][1] =  L(x,s_x)*DL(y,s_y);
                
              } else {
                
                tmp_shape_values[3*y+x] = L2h(x,s_xh) * L2h(y,s_yh);
                tmp_shape_values_standard[3*y+x] = L(x,s_x) * L(y,s_y);
                
                tmp_shape_gradients_tilde_standard[3*y+x][0] =  DL(x,s_x)*L(y,s_y);
                tmp_shape_gradients_tilde_standard[3*y+x][1] =  L(x,s_x)*DL(y,s_y);
                tmp_shape_gradients_tilde[3*y+x][0] =  DL2h(x,s_xh)*L2h(y,s_yh);
                tmp_shape_gradients_tilde[3*y+x][1] =  L2h(x,s_xh)*DL2h(y,s_yh);
                
              }
            }
          
        } else if ((_femtype==1)||(_femtype==11)) { 
        
        for (int x=0;x<3;++x)
          for (int y=0;y<3;++y)
            {
            //Take values from above
            tmp_shape_gradients_tilde_standard[3*y+x][0] = tmp_shape_gradients_tilde[3*y+x][0];
            tmp_shape_gradients_tilde_standard[3*y+x][1] = tmp_shape_gradients_tilde[3*y+x][1];
            tmp_shape_values_standard[3*y+x] = tmp_shape_values[3*y+x]; 
            
            if ((x!=1)&&(y!=1)) { 
              //Change only shape functions of the outer vertices
              
              //These are the only lines, where we differentiate between type 1 and 11
              //The difference are the basis functions phi_2h to be used
              //1 is based on the diagonal line from bottom left to top right, 
              //11 on the one from bottom right to top left
              if (_femtype==1) {
                tmp_shape_values[3*y+x] = E1(x/2,y/2, s_xh, s_yh);
                tmp_shape_gradients_tilde[3*y+x][0] = 0.5*E1x(x/2,y/2,s_xh, s_yh);
                tmp_shape_gradients_tilde[3*y+x][1] = 0.5*E1y(x/2,y/2,s_xh,s_yh);
              } else if (_femtype==11) {
                tmp_shape_values[3*y+x] = E0(x/2,y/2, s_xh, s_yh);
                tmp_shape_gradients_tilde[3*y+x][0] = 0.5*E0x(x/2,y/2,s_xh, s_yh);
                tmp_shape_gradients_tilde[3*y+x][1] = 0.5*E0y(x/2,y/2,s_xh,s_yh);
              }
            }
            }         
        
        
      } else if (_femtype==2) {
        
        for (int x=0;x<3;++x)
          for (int y=0;y<3;++y)
            {
            
              tmp_shape_gradients_tilde_standard[3*y+x][0] = tmp_shape_gradients_tilde[3*y+x][0];
              tmp_shape_gradients_tilde_standard[3*y+x][1] = tmp_shape_gradients_tilde[3*y+x][1];
              tmp_shape_values_standard[3*y+x] = tmp_shape_values[3*y+x]; 
            
              if ((x%2!=1)&&(y%2!=1)) {
                tmp_shape_values[3*y+x] = E0(x/2,y/2, s_xh, s_yh);
                tmp_shape_gradients_tilde[3*y+x][0] = 0.5*E0x(x/2,y/2,s_xh, s_yh);
                tmp_shape_gradients_tilde[3*y+x][1] = 0.5*E0y(x/2,y/2,s_xh, s_yh);
              
              }
            }         
        
      } else if (_femtype==3) {
        
        for (int x=0;x<3;++x)
          for (int y=0;y<3;++y)
            {
            
              tmp_shape_gradients_tilde_standard[3*y+x][0] = tmp_shape_gradients_tilde[3*y+x][0];
              tmp_shape_gradients_tilde_standard[3*y+x][1] = tmp_shape_gradients_tilde[3*y+x][1];
              tmp_shape_values_standard[3*y+x] = tmp_shape_values[3*y+x]; 
            
            
              if ((x!=1)&&(y!=1)) {
              
                tmp_shape_values[3*y+x] = E1(x/2,y/2, s_xh, s_yh);
                tmp_shape_gradients_tilde[3*y+x][0] = 0.5*E1x(x/2,y/2,s_xh, s_yh);
                tmp_shape_gradients_tilde[3*y+x][1] = 0.5*E1y(x/2,y/2,s_xh, s_yh);
              }
            }         
      }
      
      //Renumbering
      _shape_values(0,q_point) = tmp_shape_values[0];
      _shape_values(1,q_point) = tmp_shape_values[2];
      _shape_values(2,q_point) = tmp_shape_values[6];
      _shape_values(3,q_point) = tmp_shape_values[8];
      _shape_values(4,q_point) = tmp_shape_values[3];
      _shape_values(5,q_point) = tmp_shape_values[5];
      _shape_values(6,q_point) = tmp_shape_values[1];
      _shape_values(7,q_point) = tmp_shape_values[7];
      _shape_values(8,q_point) = tmp_shape_values[4];
      
      _shape_gradients_tilde(0,q_point) = tmp_shape_gradients_tilde[0];
      _shape_gradients_tilde(1,q_point) = tmp_shape_gradients_tilde[2];
      _shape_gradients_tilde(2,q_point) = tmp_shape_gradients_tilde[6];
      _shape_gradients_tilde(3,q_point) = tmp_shape_gradients_tilde[8];
      _shape_gradients_tilde(4,q_point) = tmp_shape_gradients_tilde[3];
      _shape_gradients_tilde(5,q_point) = tmp_shape_gradients_tilde[5];
      _shape_gradients_tilde(6,q_point) = tmp_shape_gradients_tilde[1];
      _shape_gradients_tilde(7,q_point) = tmp_shape_gradients_tilde[7];
      _shape_gradients_tilde(8,q_point) = tmp_shape_gradients_tilde[4];
      
      _shape_values_standard(0,q_point) = tmp_shape_values_standard[0];
      _shape_values_standard(1,q_point) = tmp_shape_values_standard[2];
      _shape_values_standard(2,q_point) = tmp_shape_values_standard[6];
      _shape_values_standard(3,q_point) = tmp_shape_values_standard[8];
      _shape_values_standard(4,q_point) = tmp_shape_values_standard[3];
      _shape_values_standard(5,q_point) = tmp_shape_values_standard[5];
      _shape_values_standard(6,q_point) = tmp_shape_values_standard[1];
      _shape_values_standard(7,q_point) = tmp_shape_values_standard[7];
      _shape_values_standard(8,q_point) = tmp_shape_values_standard[4];
      
      _shape_gradients_tilde_standard(0,q_point) = tmp_shape_gradients_tilde_standard[0];
      _shape_gradients_tilde_standard(1,q_point) = tmp_shape_gradients_tilde_standard[2];
      _shape_gradients_tilde_standard(2,q_point) = tmp_shape_gradients_tilde_standard[6];
      _shape_gradients_tilde_standard(3,q_point) = tmp_shape_gradients_tilde_standard[8];
      _shape_gradients_tilde_standard(4,q_point) = tmp_shape_gradients_tilde_standard[3];
      _shape_gradients_tilde_standard(5,q_point) = tmp_shape_gradients_tilde_standard[5];
      _shape_gradients_tilde_standard(6,q_point) = tmp_shape_gradients_tilde_standard[1];
      _shape_gradients_tilde_standard(7,q_point) = tmp_shape_gradients_tilde_standard[7];
      _shape_gradients_tilde_standard(8,q_point) = tmp_shape_gradients_tilde_standard[4];
      
      
      for (int i=0; i<4; i++) {
        //F2h only depends on the dofs 0 to 3 that are the dofs from V2h
        F2h[0][0] += _shape_gradients_tilde(i,q_point)[0] * _M[0][i];
        F2h[0][1] += _shape_gradients_tilde(i,q_point)[1] * _M[0][i];
        F2h[1][0] += _shape_gradients_tilde(i,q_point)[0] * _M[1][i];
        F2h[1][1] += _shape_gradients_tilde(i,q_point)[1] * _M[1][i];
        
      }
      
      F2h_inv_transpose.clear();
      F2h_inv_transpose = transpose(invert(F2h));
      
      //Grad phi = F^{-T} gradhat phihat
      //We scale the hierarchical basis functions with F2h, the standard basis functions
      //with F
      for (int i=0; i<9;i++)
        {
          _shape_gradients(i,q_point) = F_inv_transpose *  _shape_gradients_tilde(i,q_point);
          if ( i<4)
            _shape_gradients(i,q_point) = F2h_inv_transpose *  _shape_gradients_tilde(i,q_point);
          
        }
      
      }
    }
}

//Get specific values in quadrature points
template <int dim,int spacedim>
double LocModFEValues<dim,spacedim>::shape_value(const unsigned int function_no,
                                                 const unsigned int point_no)
{
  typedef FEValuesBase<dim,spacedim> FVB;
  Assert(_update_flags & update_values, typename FVB::ExcAccessToUninitializedField("update_values"));
  return _shape_values(function_no,point_no);
}

template <int dim,int spacedim>
double LocModFEValues<dim,spacedim>::shape_value_standard(const unsigned int function_no,
                                                 const unsigned int point_no)
{
  typedef FEValuesBase<dim,spacedim> FVB;
  Assert(_update_flags & update_values, typename FVB::ExcAccessToUninitializedField("update_values"));
  return _shape_values_standard(function_no,point_no);
}

template <int dim,int spacedim>
Tensor<1,dim> LocModFEValues<dim,spacedim>::shape_grad(const unsigned int function_no,
                                                       const unsigned int point_no)
{
  typedef FEValuesBase<dim,spacedim> FVB;
  Assert(_update_flags & update_gradients, typename FVB::ExcAccessToUninitializedField("update_gradients"));
  return _shape_gradients(function_no,point_no);
}


//Get the whole vector of values in all quadrature points
template <int dim,int spacedim>
void LocModFEValues<dim,spacedim>::get_function_values(const typename DoFHandler<dim,spacedim>::cell_iterator &cell,
						       const Vector<double>& fe_function,
						       std::vector<Vector<double> > &values)
{
  for (unsigned int q=0; q<values.size(); ++q)
    values[q](0) = 0.0;

  std::vector<unsigned int> local_dof_indices (9);
  cell->get_dof_indices (local_dof_indices);

  for (unsigned int q=0; q<values.size(); ++q)
    {
      for (int i=0;i<9;i++)
      values[q](0) +=  _shape_values(i,q) * fe_function(local_dof_indices[i]);
    }
}


template <int dim,int spacedim>
void LocModFEValues<dim,spacedim>
::get_function_gradients(const typename DoFHandler<dim,spacedim>::cell_iterator &cell, 
			 const Vector<double>& fe_function, 
			 std::vector< std::vector<Tensor<1,dim> > > &values)
{
  
  for (unsigned int q=0; q<values.size(); ++q)
    for (unsigned int d=0; d<dim; ++d)
      values[q][0][d] = 0.0;
  
  std::vector<unsigned int> local_dof_indices (9);
  cell->get_dof_indices (local_dof_indices);
  
  for (int i=0;i<9;i++)
    {
      double fe_dof_i=fe_function(local_dof_indices[i]);
      
      for (unsigned int q=0; q<values.size(); ++q)
      {
        values[q][0][0] +=  _shape_gradients(i,q)[0] * fe_dof_i;
        values[q][0][1] +=  _shape_gradients(i,q)[1] * fe_dof_i;
      }
      
    }
  
  
}



/****************Class LocModFE*****************************************/

//Constructor
template <int dim>
LocModFE<dim>::LocModFE ()
{
  //Standard value (can be changed from outside later)
  _hierarchical = false;
  chi = new LevelSet<dim>;
}


// Find cut with Newton-scheme in terms of s in [0,1], 
// the cut position is then (1-s) x1 + s x2
template <int dim>
double LocModFE<dim>::find_cut (Point<dim> x1, Point<dim> x2)
{
  
  assert(chi->domain(x1)!=chi->domain(x2));
  
  Point<dim> HX;
  HX[0] = x2[0] - x1[0];
  HX[1] = x2[1] - x1[1];
  
  //Length of the line
  double H = std::sqrt(HX[0]*HX[0] + HX[1]*HX[1]);
  double s = 0.5; // initial guess
  
  // Actual iterate
  Point<dim> x;
  x[0] = (1.0 - s) * x1[0] + s * x2[0];
  x[1] = (1.0 - s) * x1[1] + s * x2[1];
  
  //Residual
  double res=chi->dist(x);
  
  int iter=0;
  for (iter=0;(iter<20)&&(fabs(res)>1.e-14*H);++iter)
    {
      //Derivatived for Newton
      Point<dim> D(chi->dist_x(x), chi->dist_y(x));
      assert((D[0]*HX[0] + D[1]*HX[1])!=0);
      //Newton update
      s-= res/(D[0]*HX[0] + D[1]*HX[1]);
      
      //Corresponding point
      x[0] = (1.0 - s) * x1[0] + s * x2[0];
      x[1] = (1.0 - s) * x1[1] + s * x2[1];
      
      //New residual
      res=chi->dist(x);
    }
  if (iter==19) 
    {
      cerr << "FindCut Newton did not converge: " << x1 << " " << x2 << endl;
      abort();
    }
  
  //Assert that cut goes through the line
  assert(s<1.0+1.e-6*H); 
  assert(s>   -1.e-6*H); 
  return s;
  
}

//Computes cut position in terms of s and t of the lines (x0,y0) + s(x1-x0, y1-y0)
// and (x2, y2) + t (x3-x2, y3-y2)
template <int dim>
void LocModFE<dim>::FindCutOfTwoLines(double &s, double& t, double x0, double y0,
				      double x1, double y1, double x2, double y2, 
				      double x3, double y3)
{
  
  //Check, if the lines are parallel
  if (fabs((x1-x0)*(y3-y2) - (x3-x2)*(y1-y0))>1.e-12) {
    
    s= ((y3-y2)*(x2-x0) + (y0-y2)*(x3-x2)) / ((x1-x0)*(y3-y2) - (x3-x2)*(y1-y0));
    t= ((x1-x0)*(y0-y2) + (y1-y0)*(x2-x0)) / ((x1-x0)*(y3-y2) - (x3-x2)*(y1-y0));
    
  } else {
    cout << " Problem to find cut of two lines, as lines (" << x0 << "," <<y0<< ") + s(" 
	 << x1 -x0<< "," << y1-y0 << ") and (" << x2 <<"," << y2 <<") + t (" << x3-x2<< "," << y3-y2 
	 << ") are parallel." << endl;
    s=0.5; t=0.5;
  }
  
}


//Set patch and vertex colors
template <int dim>
void LocModFE<dim>::set_material_ids (const DoFHandler<dim> &dof_handler,
				      const Triangulation<dim> &triangulation)
{
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  
  cell_colors.reinit(triangulation.n_active_cells());
  node_colors.reinit(triangulation.n_used_vertices());
  
  
  unsigned int subdom1_counter;
  
  for (unsigned int cell_counter = 0; cell!=endc; ++cell, cell_counter++)
    { 
      subdom1_counter = 0;
      
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        //First determine the sub-domain of the four outer vertices
        double chi_local = chi->domain(cell->vertex(v));
        node_colors[cell->vertex_index(v)] = chi_local;
        
        if (chi_local > 0) subdom1_counter ++;			
      }
      
      //Based on the colors of the vertices, specify a color for the patches
      //(0 stands for an interface patch)
      if (subdom1_counter == 4)
        cell_colors[cell_counter] = 1;
      else if (subdom1_counter == 0)
        cell_colors[cell_counter] = -1;
      else 
        cell_colors[cell_counter] = 0;
      
    }	
}


// Initialize the lcoal variables specific to LocModFE
// Specifically the different fem_types are computed depending
// on the cut of the interface, the local map T_P(parametrised in M)
// and the discrete level set function (parametrised by its values in the 
// vertices 
template <int dim>
void LocModFE<dim>::init_FEM (const typename DoFHandler<dim>::active_cell_iterator &cell,
			      unsigned int cell_counter, 
			      FullMatrix<double> &M,
			      const unsigned int   dofs_per_cell,
			      unsigned int &femtype_int,
			      std::vector<double> &LocalDiscChi,
			      std::vector<int>& NodesAtInterface)
{
  
  std::string _femtype;
  
  //If the cut is very close to a vertex, we move it by a distance smaller eps
  //onto it, in order to avoid unnecessarily large anisotropies   
  double eps = 1.e-6;
  
  //Init standard position of the vertices for a Q1 patch element
  FullMatrix<double> T(dim, dofs_per_cell);
  for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      T[0][v] = cell->vertex(v)[0];
      T[1][v] = cell->vertex(v)[1];
    }
  
  T[0][4] = 0.5 * (T[0][0] + T[0][2]);
  T[1][4] = 0.5 * (T[1][0] + T[1][2]);
  
  T[0][5] = 0.5 * (T[0][1] + T[0][3]);
  T[1][5] = 0.5 * (T[1][1] + T[1][3]);
  
  T[0][6] = 0.5 * (T[0][1] + T[0][0]);
  T[1][6] = 0.5 * (T[1][1] + T[1][0]);
  
  T[0][7] = 0.5 * (T[0][2] + T[0][3]);
  T[1][7] = 0.5 * (T[1][2] + T[1][3]);
  
  T[0][8] = 0.25 * (T[0][0] + T[0][1] + T[0][2] + T[0][3]);
  T[1][8] = 0.25 * (T[1][0] + T[1][1] + T[1][2] + T[1][3]);
  
  
  NodesAtInterface.clear();
  LocalDiscChi.resize(dofs_per_cell);
  
  int domain = cell_colors[cell_counter];
  
  double s,t;
  
  for (int i = 0; i < 9; i++) {
    LocalDiscChi[i] = domain;
  }
  
  //Nodes belonging to each of the four edges
  int e2n[4][3] = { { 0, 6, 1 }, { 1, 5, 3 }, { 3, 7, 2 }, { 2, 4, 0 } };
  
  //Init map T_P (M)
  for (unsigned int n = 0; n < T.n(); ++n)
    for (unsigned int m = 0; m < T.m(); ++m)
      M(m, n) = T(m, n);
  
  //Variables for the positions, where the interface cuts outer edges
  double pos_f2s = 0;
  double pos_s2f = 0;
  
  
  //If a patch is cut by the interface (color 0), we compute the position 
  //of the cut
  if (cell_colors[cell_counter] == 0)
    {
      int ro[4] = {0,1,3,2};
      //f2s and s2f denote the edges that are cut
      int f2s = -1;
      int s2f = -1;
      int node_colors_local[4];
      
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        node_colors_local[v] = node_colors[cell->vertex_index(v)];
      }
      
      
      for (int i=0; i<4; i++)
      {
        if ((node_colors_local[ro[i]] == -1) && (node_colors_local[ro[(i+1)%4]] == 1))
          f2s = i;
        else if ((node_colors_local[ro[i]] == 1) && (node_colors_local[ro[(i+1)%4]] == -1))
          s2f = i;
      }
      
      Point<dim> x1(cell->vertex(ro[f2s])[0],cell->vertex(ro[f2s])[1]);
      Point<dim> x2(cell->vertex(ro[(f2s+1)%4])[0],cell->vertex(ro[(f2s+1)%4])[1]);
      
      //Local position pos_s2f in (0,1), where edge f2s is cut
      pos_f2s = find_cut(x1,x2);
      
      Point<dim> x3(cell->vertex(ro[s2f])[0],cell->vertex(ro[s2f])[1]);
      Point<dim> x4(cell->vertex(ro[(s2f+1)%4])[0],cell->vertex(ro[(s2f+1)%4])[1]);
      
      //Local position pos_s2f in (0,1), where edge f2s is cut
      pos_s2f = find_cut(x3,x4);
      
      std::vector<unsigned int> local_dof_indices (dofs_per_cell);
      cell->get_dof_indices (local_dof_indices);
      
      
      // (I) Define FemType: q2 (not cut), h (horizontal cut), v (vertical cut), 
      // r0, r1, r2, r3 (Cut through adjacent edges that leaved vertex 0/1/2/3 isolated), 
      // v0, v1, v2, v3 (cut through vertex 0/1/2/3 and edge), 
      // d02 or d13 (diagonal cut through two vertices)
      
      assert(f2s != s2f);
      int splittype = (f2s - s2f + 4) % 4;
      if (splittype == 2) // Cut through opposite edges
      {
        assert(f2s % 2 == s2f % 2);
        if ((f2s == 1) || (s2f == 1))
          _femtype = "h"; //horizontal cut
        else if ((f2s == 0) || (s2f == 0))
          _femtype = "v"; //vertical cut
        else
          abort();
      } 
      else //Cut through adjacent edges
      {
        if (((f2s == 0) && (s2f == 3)) || ((f2s == 3) && (s2f == 0)))
          _femtype = "r0";//Vertex 0 is on one side, 1 to 3 on the other
        else if (((f2s == 1) && (s2f == 0)) || ((f2s == 0) && (s2f == 1)))
          _femtype = "r1";//Vertex 1 is left alone
        else if (((f2s == 2) && (s2f == 1)) || ((f2s == 1) && (s2f == 2)))
          _femtype = "r2";//Vertex 2 is left alone
        else if (((f2s == 3) && (s2f == 2)) || ((f2s == 2) && (s2f == 3)))
          _femtype = "r3";//Vertex 3 is left alone
        else
          abort();
      }
      
      
      // Define indicators which indicate if cut goes through
      // vertex (up to eps). If the cut position on an edge is 
      // closer to a vertex than eps*h, we consider that the cut
      // goes through the vertex 
      int vf2s = -1;
      int vs2f = -1;
      
      int vi= -1;
      int vj= -1;
      if (pos_f2s > 1. - eps) {
        vf2s = 1;
      }
      if (pos_f2s < eps) {
        vf2s = 0;
      }
      if (pos_s2f > 1. - eps) {
        vs2f = 1;
      }
      if (pos_s2f < eps) {
        vs2f = 0;
      }
      
      // f2s+vf2s stands now for the vertex that cut goes through (if
      // any)
      
      if (vf2s + vs2f > -2) {
      // at least one cut through vertex
      
      if ((vf2s < 0) && (vs2f >= 0)) {
        // only one cut through vertex i=s2f+vs2f -> Element v_i
        int v = (s2f + vs2f) % 4;
        if (v == 0)
          _femtype = "v0";//Cut through vertex 0
        if (v == 1)
          _femtype = "v1";//Cut through vertex 1
        if (v == 2)
          _femtype = "v2";//Cut through vertex 2
        if (v == 3)
          _femtype = "v3";//Cut through vertex 3
        
      }
      
      if ((vf2s >= 0) && (vs2f < 0)) {
        // only one cut through vertex i=f2s+vf2s -> Element v_i
        int v = (f2s + vf2s) % 4;
        if (v == 0)
          _femtype = "v0";
        if (v == 1)
          _femtype = "v1";
        if (v == 2)
          _femtype = "v2";
        if (v == 3)
          _femtype = "v3";
      } 
      
      if ((vf2s >= 0) && (vs2f >= 0)) {
        // Cut through vertices i=f2s+vf2s and j=s2f+vs2f
        vi = (f2s + vf2s) % 4;
        vj = (s2f + vs2f) % 4;
        int dist = vi - vj;
        
        if (abs(dist) != 2)
          _femtype = "q1"; // Cell only touched by interface, set regular femtype Q1
        else if (vi % 2 == 0)
          _femtype = "d02";//Interface goes through the two vertices 0 and 2
        else
          _femtype = "d13";
      }
    }
      
      
    // Move degrees of freedom 
    // (i) on cut edges: Set edge midpoint to the point, where the cut goes through
    if ((_femtype != "q1") && (_femtype != "d02") && (_femtype != "d13")) {
	
      for (int j = 0; j < 2; ++j) {
        if ((eps < pos_f2s) && (pos_f2s < 1. - eps)) {
          M(j, e2n[f2s][1]) = pos_f2s * M(j, e2n[f2s][2]) + (1.0 - pos_f2s) * M(j, e2n[f2s][0]);
        }
        if ((eps < pos_s2f) && (pos_s2f < 1. - eps)) {
          M(j, e2n[s2f][1]) = pos_s2f * M(j, e2n[s2f][2]) + (1.0 - pos_s2f) * M(j, e2n[s2f][0]);
        }
      }
      
      //(ii) Midpoint 
      // Move to diagonal line for hierarchical approach and interface!
      if (_femtype == "h") {
        //On the way, we also define some useful vertex lists
        //
        //The list NodesAtInterface contains all local indices that are at the interface. 
        //This is useful for example for visualisation 
        //or to eliminate interface dofs from the integration on one side for a dG type approach
        //that might be useful to avoid a feedback from certain variables to the other side
        NodesAtInterface.push_back(4);
        NodesAtInterface.push_back(8);
        NodesAtInterface.push_back(5);
        //The discrete level set function is needed to determine which sub-cells lie in which 
        //sub-domain. If the continuous level set function would be used for this, certain sub-cells
        //would partly lie in both subdomains and quadrature would be inaccurate
        LocalDiscChi[4] = 0.;
        LocalDiscChi[8] = 0.;
        LocalDiscChi[5] = 0.;
        double domain_below = node_colors[cell->vertex_index(0)];
        double domain_above  = node_colors[cell->vertex_index(2)];
        LocalDiscChi[0] = domain_below;
        LocalDiscChi[6] = domain_below;
        LocalDiscChi[1] = domain_below;
        LocalDiscChi[2] = domain_above;
        LocalDiscChi[7] = domain_above;
        LocalDiscChi[3] = domain_above;
        
        
        if (!_hierarchical) {
          //Move the midpoint to the centre of the horizontal line
          for (int j = 0; j < 2; ++j)
            M(j, 8) = 0.5 * M(j, 4) + 0.5 * M(j, 5);
          
          //Depending on the cut positions, different reference elements yield
          //nicer elements (h1: femtype_int 3 (quadrilaterals are divided by diagonal
          //from bottom left (bl) to top right (tr), h2: femtype_int 2 by diagonal bottom right/top left)
          if (pos_f2s+pos_s2f < 1.) 
            _femtype ="h1";
          else _femtype ="h2";

        } else {
          
          //In the hierarchical case, the midpoint needs to lie on the respective 
          //diagonal line                              
          if (pos_f2s + pos_s2f < 1.) {
            
            //Use diagonal bl to tr (Vertex 0 to 3)
            //Compute cut with horizontal line (Vertex 4 to 5)
            FindCutOfTwoLines(s,t, M(0,0), M(1,0), M(0,3), M(1,3), M(0,4), M(1,4), M(0,5), M(1,5));
            for (int i=0; i<dim; i++) {
              M(i,8) = M(i,0) + s*(M(i,3)-M(i,0));
            }
            
            _femtype = "h1";
	      
          } else {
            
            //Use diagonal br to tl (Vertex 1 to 2)
            //Compute cut with horizontal line (Vertex 6 to 7)
            FindCutOfTwoLines(s,t, M(0,1), M(1,1), M(0,2), M(1,2), M(0,4), M(1,4), M(0,5), M(1,5));
            for (int i=0; i<dim; i++) {
              M(i,8) = M(i,1) + s*(M(i,2)-M(i,1));
            }
            
            _femtype = "h2";
          }
        }
      }
      
      if (_femtype == "v") {
        
        NodesAtInterface.push_back(6);
        NodesAtInterface.push_back(8);
        NodesAtInterface.push_back(7);
        LocalDiscChi[6] = 0.;
        LocalDiscChi[8] = 0.;
        LocalDiscChi[7] = 0.;
        double domain_left = node_colors[cell->vertex_index(0)];
        double domain_right = node_colors[cell->vertex_index(1)];
        LocalDiscChi[0] = domain_left;
        LocalDiscChi[4] = domain_left;
        LocalDiscChi[2] = domain_left;
        LocalDiscChi[1] = domain_right;
        LocalDiscChi[5] = domain_right;
        LocalDiscChi[3] = domain_right;
        
        if (!_hierarchical) {
          
          //In the non-hierarchical case, the centre of the vertical line
          //is a good choice for the midpoint
          for (int j = 0; j < 2; ++j)
            M(j, 8) = 0.5 * M(j, 6) + 0.5 * M(j, 7);
          
          if (pos_f2s+pos_s2f > 1.) //Changed!
            _femtype = "v_1";//femtype_int 3 (Diagonal bl to tr)
          else 
            _femtype = "v_2";//femtype_int 2 (Diagonal br to tl)
          
        } else {
          //In the hierarchical case, the midpoint has to lie again on one of the 
          //diagonal lines of the patch
          
          if (pos_f2s + pos_s2f > 1.) {
            //Use diagonal bl to tr (Vertex 0 to 3)
            //Compute cut with horizontal line (Vertex 6 to 7)
            FindCutOfTwoLines(s,t, M(0,0), M(1,0), M(0,3), M(1,3), M(0,6), M(1,6), M(0,7), M(1,7));
            for (int i=0; i<dim; i++) {
              M(i,8) = M(i,0) + s*(M(i,3)-M(i,0));
            }
            
            _femtype = "v_1";
            
          } else {
            //Use diagonal br to tl (Vertex 1 to 2)
            //Compute cut with vertical line (Vertex 6 to 7)
            FindCutOfTwoLines(s,t, M(0,1), M(1,1), M(0,2), M(1,2), M(0,6), M(1,6), M(0,7), M(1,7));
            for (int i=0; i<dim; i++) {
              M(i,8) = M(i,1) + s*(M(i,2)-M(i,1));
            }
            
            _femtype = "v_2";
          }
        }
        
      }
      
      if (_femtype == "r0") {
        NodesAtInterface.push_back(6);
        NodesAtInterface.push_back(4);
        LocalDiscChi[6] = 0.;
        LocalDiscChi[4] = 0.;
        double domain_left = node_colors[cell->vertex_index(0)];
        double domain_right = node_colors[cell->vertex_index(1)];
        LocalDiscChi[0] = domain_left;
        LocalDiscChi[1] = domain_right;
        LocalDiscChi[8] = domain_right;
        LocalDiscChi[5] = domain_right;
        LocalDiscChi[2] = domain_right;
        LocalDiscChi[7] = domain_right;
        LocalDiscChi[3] = domain_right;
      }
      if (_femtype == "r1") {
        NodesAtInterface.push_back(6);
        NodesAtInterface.push_back(5);
        LocalDiscChi[6] = 0.;
        LocalDiscChi[5] = 0.;
        double domain_left = node_colors[cell->vertex_index(2)];
        double domain_right = node_colors[cell->vertex_index(1)];
        LocalDiscChi[0] = domain_left;
        LocalDiscChi[4] = domain_left;
        LocalDiscChi[8] = domain_left;
        LocalDiscChi[2] = domain_left;
        LocalDiscChi[7] = domain_left;
        LocalDiscChi[3] = domain_left;
        LocalDiscChi[1] = domain_right;
      }
      if (_femtype == "r2") {
        NodesAtInterface.push_back(5);
        NodesAtInterface.push_back(7);
        LocalDiscChi[5] = 0.;
        LocalDiscChi[7] = 0.;
        double domain_left = node_colors[cell->vertex_index(0)];
        double domain_right = node_colors[cell->vertex_index(3)];
        LocalDiscChi[0] = domain_left;
        LocalDiscChi[6] = domain_left;
        LocalDiscChi[1] = domain_left;
        LocalDiscChi[4] = domain_left;
        LocalDiscChi[8] = domain_left;
        LocalDiscChi[2] = domain_left;
        LocalDiscChi[3] = domain_right;
      }
      if (_femtype == "r3") {
        NodesAtInterface.push_back(4);
        NodesAtInterface.push_back(7);
        LocalDiscChi[4] = 0.;
        LocalDiscChi[7] = 0.;
        double domain_left = node_colors[cell->vertex_index(2)];
        double domain_right = node_colors[cell->vertex_index(1)];
        LocalDiscChi[2] = domain_left;
        LocalDiscChi[0] = domain_right;
        LocalDiscChi[6] = domain_right;
        LocalDiscChi[1] = domain_right;
        LocalDiscChi[8] = domain_right;
        LocalDiscChi[5] = domain_right;
        LocalDiscChi[3] = domain_right;
      }
      
      if (!_hierarchical) {
        //A good position for the midpoint is the "mean" of the midpoints of the edges
        if ((_femtype == "r0") || (_femtype == "r1") || (_femtype == "r2") || (_femtype == "r3")) {
          
          for (int j = 0; j < dim; ++j)
            M(j, 8) = 0.25 * (M(j, 4) + M(j, 5) + M(j, 6) + M(j, 7));
          
          }
        
         } else {
        
        double s,t;
        if ((_femtype == "r0") || (_femtype == "r2")) {
          //The midpoint has to be on the diagonal line from bottom right to top left
          //A maximum angle condition is guaranteed, if we use the cut point of this diagonal
          //with the interior horizontal line.      
          FindCutOfTwoLines(s,t, M(0,1), M(1,1), M(0,2), M(1,2), M(0,4), M(1,4), M(0,5), M(1,5));
          for (int j = 0; j < dim; ++j) {
            M(j, 8) = M(j,1) + s*(M(j,2)-M(j,1));
          }
        }
        if ((_femtype == "r1") || (_femtype == "r3")) {      
          //The midpoint has to be on the diagonal line from bottom left to top right
          //A maximum angle condition is guaranteed, if we use the cut point of this diagonal
          //with the interior horizontal line. 
          FindCutOfTwoLines(s,t, M(0,0), M(1,0), M(0,3), M(1,3), M(0,4), M(1,4), M(0,5), M(1,5));
          for (int j = 0; j < dim; ++j) {
            M(j, 8) = M(j,0) + s*(M(j,3)-M(j,0));
          }
          
        }
        
      }
      
      s=0.5;
      // If interface through vertex set midpoint onto interface line
      if (_femtype == "v0") {
        
        if ((s2f == 1) || (f2s == 1)) {
          // Interface through 0/5
          
          if (_hierarchical) {
            //The midpoint has to be on the diagonal line from bottom right to top left (1 to 2)
            //Find Cut with the discrete interface line from Vertex 0 to 5 
            FindCutOfTwoLines(s,t, M(0,1), M(1,1), M(0,2), M(1,2), M(0,0), M(1,0), M(0,5), M(1,5));
            for (int j = 0; j < dim; ++j) {
               M(j, 8) = M(j,1) + s*(M(j,2)-M(j,1));
            }
            
          } else {
            //Otherwise, we have more freedom to choose the midpoint
            //A better possibility (in terms of the interior angles) could be the centre 
            //of the line 0/5:
            
            for (int j = 0; j < 2; ++j)
               M(j, 8) = 0.5 * (M(j, 0) + M(j, 5));
          }
          
          NodesAtInterface.push_back(0);
          NodesAtInterface.push_back(8);
          NodesAtInterface.push_back(5);
          LocalDiscChi[0] = 0.;
          LocalDiscChi[8] = 0.;
          LocalDiscChi[5] = 0.;
          double domain_left = node_colors[cell->vertex_index(2)];
          double domain_right = node_colors[cell->vertex_index(1)];
          LocalDiscChi[4] = domain_left;
          LocalDiscChi[2] = domain_left;
          LocalDiscChi[7] = domain_left;
          LocalDiscChi[3] = domain_left;
          LocalDiscChi[6] = domain_right;
          LocalDiscChi[1] = domain_right;
                  
        }
        else if ((s2f == 2) || (f2s == 2)) {
          // Interface through 0/7
          
          if (_hierarchical) {
            //The midpoint has to be on the diagonal line from bottom right to top left (1 to 2)
            //Find Cut with the discrete interface line from Vertex 0 to 7
            FindCutOfTwoLines(s,t, M(0,1), M(1,1), M(0,2), M(1,2), M(0,0), M(1,0), M(0,7), M(1,7));
            for (int j = 0; j < dim; ++j) {
            M(j, 8) = M(j,1) + s*(M(j,2)-M(j,1));
            }
            
          } else {
            for (int j = 0; j < 2; ++j)
            M(j, 8) = 0.5 * (M(j, 0) + M(j, 7));
          }
          
          NodesAtInterface.push_back(0);
          NodesAtInterface.push_back(8);
          NodesAtInterface.push_back(7);
          LocalDiscChi[0] = 0.;
          LocalDiscChi[8] = 0.;
          LocalDiscChi[7] = 0.;
          double domain_left = node_colors[cell->vertex_index(2)];
          double domain_right = node_colors[cell->vertex_index(1)];
          LocalDiscChi[4] = domain_left;
          LocalDiscChi[2] = domain_left;
          LocalDiscChi[6] = domain_right;
          LocalDiscChi[1] = domain_right;
          LocalDiscChi[5] = domain_right;
          LocalDiscChi[3] = domain_right;
          
        }
        else 
        {
            //Cut goes through vertex and line adjacent to it
            //Element is not cut, but the interface lies partly on the elements boundary
            _femtype = "q1i";
            
            for (int j = 0; j < 2; ++j) {
                M(j, 8) = 0.25 * (M(j, 4) + M(j, 5) + M(j, 6) + M(j, 7));
            }
            
            //Set basic material id, which is -1 or 1 
            int basic_domain = node_colors[cell->vertex_index(3)];
            
            for (int i = 0; i < 9; i++) {
                LocalDiscChi[i] = basic_domain;
            }
            
            if ( ((f2s==3)&&(eps<pos_f2s)&&(pos_f2s<1.-eps)) 
                 || ((s2f==3)&&(eps<pos_s2f)&&(pos_s2f<1.-eps)))  {
            
                LocalDiscChi[0] = 0.;
                LocalDiscChi[4] = 0.;
                NodesAtInterface.push_back(0);
                NodesAtInterface.push_back(4);
          }
            
            if ( ((f2s==0)&&(eps<pos_f2s)&&(pos_f2s<1.-eps))
               || ((s2f==0)&&(eps<pos_s2f)&&(pos_s2f<1.-eps)))  {
            
                LocalDiscChi[0] = 0.;
                LocalDiscChi[6] = 0.;
                NodesAtInterface.push_back(0);
                NodesAtInterface.push_back(6);
          }
          }
      }
      
      if (_femtype == "v1") {
        
        if ((s2f == 2) || (f2s == 2)) {
          // Interface through 1/7
          if (_hierarchical) {
            //The midpoint has to be on the diagonal line from bottom left to top right (0 to 3)
            //Find Cut with the discrete interface line from Vertex 1 to 7
            FindCutOfTwoLines(s,t, M(0,0), M(1,0), M(0,3), M(1,3), M(0,1), M(1,1), M(0,7), M(1,7));
            for (int j = 0; j < dim; ++j) {
                M(j, 8) = M(j,0) + s*(M(j,3)-M(j,0));
            }
            
            
          } else {
            for (int j = 0; j < 2; ++j)
                M(j, 8) = 0.5 * (M(j, 1) + M(j, 7));
          } 
            
          NodesAtInterface.push_back(1);
          NodesAtInterface.push_back(8);
          NodesAtInterface.push_back(7);
          LocalDiscChi[1] = 0.;
          LocalDiscChi[8] = 0.;
          LocalDiscChi[7] = 0.;
          double domain_left = node_colors[cell->vertex_index(0)];
          double domain_right = node_colors[cell->vertex_index(3)];
          LocalDiscChi[0] = domain_left;
          LocalDiscChi[6] = domain_left;
          LocalDiscChi[4] = domain_left;
          LocalDiscChi[2] = domain_left;
          LocalDiscChi[5] = domain_right;
          LocalDiscChi[3] = domain_right;
          
        } else if ((s2f == 3) || (f2s == 3)) {
          // Interface through 1/4
          if (_hierarchical) {  
            //The midpoint has to be on the diagonal line from bottom left to top right (0 to 3)
            //Find Cut with the discrete interface line from Vertex 1 to 4
            FindCutOfTwoLines(s,t, M(0,0), M(1,0), M(0,3), M(1,3), M(0,1), M(1,1), M(0,4), M(1,4));
            for (int j = 0; j < dim; ++j) {
            M(j, 8) = M(j,0) + s*(M(j,3)-M(j,0));
            }
            
          } else {
            
            for (int j = 0; j < 2; ++j)
            M(j, 8) = 0.5 * (M(j, 1) + M(j, 4));
          }
          
          NodesAtInterface.push_back(1);
          NodesAtInterface.push_back(4);
          NodesAtInterface.push_back(8);
          LocalDiscChi[1] = 0.;
          LocalDiscChi[4] = 0.;
          LocalDiscChi[8] = 0.;
          
          double domain_left = node_colors[cell->vertex_index(0)];
          double domain_right = node_colors[cell->vertex_index(3)];
          LocalDiscChi[0] = domain_left;
          LocalDiscChi[6] = domain_left;
          LocalDiscChi[5] = domain_right;
          LocalDiscChi[2] = domain_right;
          LocalDiscChi[7] = domain_right;
          LocalDiscChi[3] = domain_right;
          
        } else {
          
          _femtype = "q1i";
          for (int j = 0; j < 2; ++j) {
            M(j, 8) = 0.25 * (M(j, 4) + M(j, 5) + M(j, 6) + M(j, 7));
          }
          
          //Set basic material id, which is -1 or 1 
          int basic_domain = node_colors[cell->vertex_index(2)];
          
          for (int i = 0; i < 9; i++) {
            LocalDiscChi[i] = basic_domain;
          }
          
          
          if ( ((f2s==1)&&(eps<pos_f2s)&&(pos_f2s<1.-eps)) 
             || ((s2f==1)&&(eps<pos_s2f)&&(pos_s2f<1.-eps)))  {
            
            LocalDiscChi[1] = 0.;
            LocalDiscChi[5] = 0.;
            NodesAtInterface.push_back(1);
            NodesAtInterface.push_back(5);
          }
          if ( ((f2s==0)&&(eps<pos_f2s)&&(pos_f2s<1.-eps)) 
             || ((s2f==0)&&(eps<pos_s2f)&&(pos_s2f<1.-eps)))  {
            
            LocalDiscChi[6] = 0.;
            LocalDiscChi[1] = 0.;
            NodesAtInterface.push_back(6);
            NodesAtInterface.push_back(1);
          }
        }
      }
      
      if (_femtype == "v2") {
        if ((s2f == 0) || (f2s == 0)) {
          // Interface through 3/6
          if (_hierarchical) { 
            //The midpoint has to be on the diagonal line from bottom right to top left (1 to 2)
            //Find Cut with the discrete interface line from Vertex 3 to 6
            FindCutOfTwoLines(s,t, M(0,1), M(1,1), M(0,2), M(1,2), M(0,3), M(1,3), M(0,6), M(1,6));
            for (int j = 0; j < dim; ++j) {
            M(j, 8) = M(j,1) + s*(M(j,2)-M(j,1));
            }
            
          } else {
            
            for (int j = 0; j < 2; ++j)
            M(j, 8) = 0.5 * (M(j, 3) + M(j, 6));
          }
          
          NodesAtInterface.push_back(6);
          NodesAtInterface.push_back(8);
          NodesAtInterface.push_back(3);
          LocalDiscChi[6] = 0.;
          LocalDiscChi[8] = 0.;
          LocalDiscChi[3] = 0.;
          double domain_left = node_colors[cell->vertex_index(2)];
          double domain_right = node_colors[cell->vertex_index(1)];
          LocalDiscChi[0] = domain_left;
          LocalDiscChi[4] = domain_left;
          LocalDiscChi[2] = domain_left;
          LocalDiscChi[7] = domain_left;
          LocalDiscChi[1] = domain_right;
          LocalDiscChi[5] = domain_right;
          
        } else if ((s2f == 3) || (f2s == 3)) {
          // Interface through 3/4
          if (_hierarchical) {  
            //The midpoint has to be on the diagonal line from bottom right to top left (1 to 2)
            //Find Cut with the discrete interface line from Vertex 3 to 4
            FindCutOfTwoLines(s,t, M(0,1), M(1,1), M(0,2), M(1,2), M(0,3), M(1,3), M(0,4), M(1,4));
            for (int j = 0; j < dim; ++j) {
              M(j, 8) = M(j,1) + s*(M(j,2)-M(j,1));
            }
            
          } else { 
            for (int j = 0; j < 2; ++j)
               M(j, 8) = 0.5 * (M(j, 3) + M(j, 4));
          }
          
          NodesAtInterface.push_back(4);
          NodesAtInterface.push_back(8);
          NodesAtInterface.push_back(3);
          
          LocalDiscChi[4] = 0.;
          LocalDiscChi[8] = 0.;
          LocalDiscChi[3] = 0.;
          double domain_left = node_colors[cell->vertex_index(2)];
          double domain_right = node_colors[cell->vertex_index(1)];
          LocalDiscChi[2] = domain_left;
          LocalDiscChi[7] = domain_left;
          LocalDiscChi[0] = domain_right;
          LocalDiscChi[6] = domain_right;
          LocalDiscChi[1] = domain_right;
          LocalDiscChi[5] = domain_right;
          
        } else {
          
          _femtype = "q1i";
          for (int j = 0; j < 2; ++j) {
            M(j, 8) = 0.25 * (M(j, 4) + M(j, 5) + M(j, 6) + M(j, 7));
          }
          
          //Set basic material id, which is -1 or 1 
          int basic_domain = node_colors[cell->vertex_index(0)];
          
          for (int i = 0; i < 9; i++) {
            LocalDiscChi[i] = basic_domain;
          }
          
          if ( ((f2s==1)&&(eps<pos_f2s)&&(pos_f2s<1.-eps)) 
             || ((s2f==1)&&(eps<pos_s2f)&&(pos_s2f<1.-eps)))  {
            
            LocalDiscChi[5] = 0.;
            LocalDiscChi[3] = 0.;
            NodesAtInterface.push_back(5);
            NodesAtInterface.push_back(3);
          }
          if ( ((f2s==2)&&(eps<pos_f2s)&&(pos_f2s<1.-eps)) 
             || ((s2f==2)&&(eps<pos_s2f)&&(pos_s2f<1.-eps)))  {
                         
            LocalDiscChi[7] = 0.;
            LocalDiscChi[3] = 0.;
            NodesAtInterface.push_back(7);
            NodesAtInterface.push_back(3);
          }
        }
      }
      
      
      if (_femtype == "v3") {
        if ((s2f == 0) || (f2s == 0)) {
          // Interface through 2/6
          if (_hierarchical) {  
            //The midpoint has to be on the diagonal line from bottom left to top right (0 to 3)
            //Find Cut with the discrete interface line from Vertex 2 to 6
            FindCutOfTwoLines(s,t, M(0,0), M(1,0), M(0,3), M(1,3), M(0,2), M(1,2), M(0,6), M(1,6));
            for (int j = 0; j < dim; ++j) {
               M(j, 8) = M(j,0) + s*(M(j,3)-M(j,0));
            }
            
          } else {
            
            for (int j = 0; j < 2; ++j)
              M(j, 8) = 0.5 * (M(j, 2) + M(j, 6));
          }
            
          NodesAtInterface.push_back(6);
          NodesAtInterface.push_back(8);
          NodesAtInterface.push_back(2);
          LocalDiscChi[6] = 0.;
          LocalDiscChi[8] = 0.;
          LocalDiscChi[2] = 0.;
          double domain_left = node_colors[cell->vertex_index(0)];
          double domain_right = node_colors[cell->vertex_index(3)];
          LocalDiscChi[0] = domain_left;
          LocalDiscChi[4] = domain_left;
          LocalDiscChi[1] = domain_right;
          LocalDiscChi[5] = domain_right;
          LocalDiscChi[7] = domain_right;
          LocalDiscChi[3] = domain_right;
          
        } else if ((s2f == 1) || (f2s == 1)) {
          // Interface through 2/5
          if (_hierarchical) {  
            //The midpoint has to be on the diagonal line from bottom left to top right (0 to 3)
            //Find Cut with the discrete interface line from Vertex 2 to 5
            FindCutOfTwoLines(s,t, M(0,0), M(1,0), M(0,3), M(1,3), M(0,2), M(1,2), M(0,5), M(1,5));
            for (int j = 0; j < dim; ++j) {
            M(j, 8) = M(j,0) + s*(M(j,3)-M(j,0));
            }
            
          } else {
            for (int j = 0; j < 2; ++j)
                M(j, 8) = 0.5 * (M(j, 5) + M(j, 2));
          }
          
          NodesAtInterface.push_back(8);
          NodesAtInterface.push_back(5);
          NodesAtInterface.push_back(2);
          LocalDiscChi[8] = 0.;
          LocalDiscChi[5] = 0.;
          LocalDiscChi[2] = 0.;
          double domain_left = node_colors[cell->vertex_index(0)];
          double domain_right = node_colors[cell->vertex_index(3)];
          LocalDiscChi[0] = domain_left;
          LocalDiscChi[6] = domain_left;
          LocalDiscChi[1] = domain_left;
          LocalDiscChi[4] = domain_left;
          LocalDiscChi[7] = domain_right;
          LocalDiscChi[3] = domain_right;
          
        } else {
          _femtype = "q1i";
          
          for (int j = 0; j < 2; ++j) {
            M(j, 8) = 0.25 * (M(j, 4) + M(j, 5) + M(j, 6) + M(j, 7));
          }
          
          //Set basic material id, which is -1 or 1 
          int basic_domain = node_colors[cell->vertex_index(1)];
          
          for (int i = 0; i < 9; i++) {
            LocalDiscChi[i] = basic_domain;
          }
          
          if ( ((f2s==3)&&(eps<pos_f2s)&&(pos_f2s<1.-eps)) 
             || ((s2f==1)&&(eps<pos_s2f)&&(pos_s2f<1.-eps)))  {
            
            LocalDiscChi[4] = 0.;
            LocalDiscChi[2] = 0.;
            NodesAtInterface.push_back(4);
            NodesAtInterface.push_back(2);
          }
          if ( ((f2s==2)&&(eps<pos_f2s)&&(pos_f2s<1.-eps)) 
             || ((s2f==2)&&(eps<pos_s2f)&&(pos_s2f<1.-eps)))  {
            
            LocalDiscChi[2] = 0.;
            LocalDiscChi[7] = 0.;
            NodesAtInterface.push_back(2);
            NodesAtInterface.push_back(7);
          }
        }
        
      }
      
      
      }
      
      if (_femtype == "d02") {
      NodesAtInterface.push_back(0);
      NodesAtInterface.push_back(8);
      NodesAtInterface.push_back(3);
      LocalDiscChi[0] = 0.;
      LocalDiscChi[8] = 0.;
      LocalDiscChi[3] = 0.;
      double domain_left = node_colors[cell->vertex_index(2)];
      double domain_right = node_colors[cell->vertex_index(1)];
      LocalDiscChi[4] = domain_left;
      LocalDiscChi[2] = domain_left;
      LocalDiscChi[7] = domain_left;
      LocalDiscChi[6] = domain_right;
      LocalDiscChi[1] = domain_right;
      LocalDiscChi[5] = domain_right;
      
      }
      
      if (_femtype == "d13") {
      NodesAtInterface.push_back(1);
      NodesAtInterface.push_back(8);
      NodesAtInterface.push_back(2);
      LocalDiscChi[1] = 0.;
      LocalDiscChi[8] = 0.;
      LocalDiscChi[2] = 0.;
      double domain_left = node_colors[cell->vertex_index(0)];
      double domain_right = node_colors[cell->vertex_index(3)];
      LocalDiscChi[0] = domain_left;
      LocalDiscChi[6] = domain_left;
      LocalDiscChi[4] = domain_left;
      LocalDiscChi[5] = domain_right;
      LocalDiscChi[7] = domain_right;
      LocalDiscChi[3] = domain_right;
      
      }
      
      
      if (_femtype == "q1") {
      
      //Set basic material id, which is -1 or 1. It might be that one or two neighbouring 
      //outer vertices lie on the 
      //interface, the value 0 will be set below. The vertex e2n[(vi+2)%4][0]
      //is guaranteed to lie in the interior as it is two edges away from vi/vj
      int basic_domain= node_colors[cell->vertex_index(e2n[(vi+2)%4][0])];
      
      for (int i = 0; i < 9; i++) {
        LocalDiscChi[i] = basic_domain;
      }
      
      if ((vi>=0)&&(vj>=0)) {
        
        if (vi==(vj+1)%4) {
          //Cut through line "vj"
          NodesAtInterface.push_back(e2n[vj][0]);
          NodesAtInterface.push_back(e2n[vj][1]);
          NodesAtInterface.push_back(e2n[vj][2]);
          LocalDiscChi[e2n[vj][0]]=0.;
          LocalDiscChi[e2n[vj][1]]=0.;
          LocalDiscChi[e2n[vj][2]]=0.;
        }
        
        if (vj==(vi+1)%4) {
          //Cut through line "vi"
          NodesAtInterface.push_back(e2n[vi][0]);
          NodesAtInterface.push_back(e2n[vi][1]);
          NodesAtInterface.push_back(e2n[vi][2]);
          LocalDiscChi[e2n[vi][0]]=0.;
          LocalDiscChi[e2n[vi][1]]=0.;
          LocalDiscChi[e2n[vi][2]]=0.;
        } 
        
        if (vi==vj) {                  
          NodesAtInterface.push_back(e2n[vi][0]);
          LocalDiscChi[e2n[vi][0]]=0.;
        }
      }
      
      }
      
    } // end (cell_colors[cell_counter] == 0)
  else _femtype = "q1";
  
  
  if (_femtype == "q1" || _femtype == "q1i")
    femtype_int = 0;
  else if (_femtype == "v1" || _femtype == "v3" )
    femtype_int = 1; // cut through vertex
  else if (_femtype == "v0" || _femtype == "v2" )
    femtype_int = 11; // cut through vertex
  else if (_femtype == "r0" || _femtype == "r2" ||  _femtype == "h2" ||
           _femtype == "v_2" || _femtype == "d13")
    femtype_int = 2; 
  else if (_femtype == "r1" || _femtype == "r3" || _femtype == "h1" ||
                   _femtype == "v_1"|| _femtype == "d02")
    femtype_int = 3; 
  else 
    {
      std::cout << "Stop. No such femtype." << std::endl;
      abort();
    }
  
  
  bool bool_print_output = false;
  
  if (bool_print_output)
    {
      std::cout << "Output: " 
                << _femtype 
                << std::endl;
      
      std::cout << "LocalDiscChi: "; 
      for (unsigned int i=0; i<LocalDiscChi.size();i++)
      std::cout << LocalDiscChi[i] << " ";
      
      std::cout << std::endl;
      
      std::cout << "NodesAtInterface: ";
      for (unsigned int i=0; i<NodesAtInterface.size();i++)
      std::cout << NodesAtInterface[i] << " ";
      
      std::cout << std::endl;
    }
  
  
}


//Initialise the four different quadrature formulas
template <int dim>
void LocModFE<dim>::initialize_quadrature()
{
  
  std::vector<Point<dim> > quad_points;
  std::vector<double>      quad_weights;
  
  
  int n_points=0;
  Point<dim> qp;
  
  //Quadrature formulas fot type 0:
  //4 point Gauss formula on each of the subquads
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      
      qp[0] = 0.25+0.5*i-sqrt(1./48.); 
      qp[1] = 0.25+0.5*j-sqrt(1./48.);
      quad_points.push_back(qp);
      quad_weights.push_back(1./16.);

      qp[0] = 0.25+0.5*i+sqrt(1./48.); 
      qp[1] = 0.25+0.5*j-sqrt(1./48.);
      quad_points.push_back(qp);
      quad_weights.push_back(1./16.);

      qp[0] = 0.25+0.5*i-sqrt(1./48.); 
      qp[1] = 0.25+0.5*j+sqrt(1./48.);
      quad_points.push_back(qp);
      quad_weights.push_back(1./16.);

      qp[0] = 0.25+0.5*i+sqrt(1./48.); 
      qp[1] = 0.25+0.5*j+sqrt(1./48.);
      quad_points.push_back(qp);
      quad_weights.push_back(1./16.);
      
    }
  }
  
  n_points=16;
  
  Quadrature0 = new Quadrature<dim>(n_points);
  Quadrature0->initialize(quad_points,quad_weights);
  
  quad_points.clear();
  quad_weights.clear();

  //Type 2:
  //3 point Gauss formula on each of the subtriangles
  /*Here, we have 4 times the two triangles
   *---*
   |\  |
   | \ |
   |  \|
   *---* */
  
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      
      qp[0] = 1./12.+0.5*i; 
      qp[1] = 1./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 1./12.+0.5*i; 
      qp[1] = 4./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 4./12.+0.5*i; 
      qp[1] = 1./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 5./12.+0.5*i; 
      qp[1] = 2./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 2./12.+0.5*i; 
      qp[1] = 5./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 5./12.+0.5*i; 
      qp[1] = 5./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);
      
    }
  }
  
  Quadrature2 = new Quadrature<dim>(24);
  Quadrature2->initialize(quad_points,quad_weights);
  
  quad_points.clear();
  quad_weights.clear();
  
  //Type 3:
  //3 point Gauss formula on each of the subtriangles
  /*Here, we have 4 times
   *---*
   |  /|
   | / |
   |/  |
   *---* */
  
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      
      qp[0] = 1./12.+0.5*i; 
      qp[1] = 2./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 1./12.+0.5*i; 
      qp[1] = 5./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 4./12.+0.5*i; 
      qp[1] = 5./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 2./12.+0.5*i; 
      qp[1] = 1./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 5./12.+0.5*i; 
      qp[1] = 1./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);

      qp[0] = 5./12.+0.5*i; 
      qp[1] = 4./12.+0.5*j;
      quad_points.push_back(qp);
      quad_weights.push_back(1./24.);
      
    }
  }
  
  Quadrature3 = new Quadrature<dim>(24);
  Quadrature3->initialize(quad_points,quad_weights);
  
  quad_points.clear();
  quad_weights.clear();
  
  //Type 1:
  //Lower left and upper right as type 2, else type 3
  //3 point Gauss formula on each of the subtriangles
  
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      
       if (i!=j) {
          qp[0] = 1./12.+0.5*i; 
          qp[1] = 1./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 1./12.+0.5*i; 
          qp[1] = 4./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 4./12.+0.5*i; 
          qp[1] = 1./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 5./12.+0.5*i; 
          qp[1] = 2./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 2./12.+0.5*i; 
          qp[1] = 5./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 5./12.+0.5*i; 
          qp[1] = 5./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          
      } else {  
	
          qp[0] = 1./12.+0.5*i; 
          qp[1] = 2./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 1./12.+0.5*i; 
          qp[1] = 5./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 4./12.+0.5*i; 
          qp[1] = 5./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 2./12.+0.5*i; 
          qp[1] = 1./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 5./12.+0.5*i; 
          qp[1] = 1./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
          qp[0] = 5./12.+0.5*i; 
          qp[1] = 4./12.+0.5*j;
          quad_points.push_back(qp);
          quad_weights.push_back(1./24.);
       }
    }
  }

  Quadrature1 = new Quadrature<dim>(24);
  Quadrature1->initialize(quad_points,quad_weights);
  
}

//Assigne one of the four formulas defined by the previous function
template <int dim>
Quadrature<dim> LocModFE<dim>::compute_quadrature (int femtype)
{
  
  if (femtype==0) return *Quadrature0;
  else if ((femtype==1)||(femtype==11)) return *Quadrature1; 
  else if (femtype==2) return *Quadrature2;
  else if (femtype==3) return *Quadrature3;
  
  // Default
  return *Quadrature0;
}


//Map local integration point to physical one, for example to set data 
template <int dim>
void LocModFE<dim>::ComputePhysicalIntegrationPoint(Point<dim>& IntPoint, 
						    LocModFEValues<dim>& fe_values, 
						    FullMatrix<double>& M, 
						    int dofs_per_cell, 
						    int q)
{
  IntPoint[0]=0.;
  IntPoint[1]=0.;
  for (int i=0; i<dofs_per_cell; ++i) {
    
    if (_hierarchical) {
      IntPoint[0] += M[0][i] * fe_values.shape_value_standard(i,q);
      IntPoint[1] += M[1][i] * fe_values.shape_value_standard(i,q);
    } else {
      IntPoint[0] += M[0][i] * fe_values.shape_value(i,q);
      IntPoint[1] += M[1][i] * fe_values.shape_value(i,q);
    }
  }
  
}

//Compute discrete level set function in quadrature point
template <int dim>
void LocModFE<dim>::ComputeLocalDiscChi(double &ChiValue, 
					int q, 
					LocModFEValues<dim>& fe_values, 
					int dofs_per_cell, 
					std::vector<double> LocalDiscChi)
{
  
  ChiValue=0.;
  for (int i=0; i<dofs_per_cell; ++i) {
    if (_hierarchical) ChiValue += LocalDiscChi[i] * fe_values.shape_value_standard(i,q);
    else ChiValue += LocalDiscChi[i] * fe_values.shape_value(i,q);
  }
  
}



// Modify the vtk-output to visualise the sub-cells of the patches 
template <int dim>
void
LocModFE<dim>::plot_vtk (const DoFHandler<dim> &dof_handler,
			 const FiniteElement<dim> &fe,
			 const Vector<double>& solution,
			 const unsigned int refinement_cycle,
			 const std::string output_basis_filename) 
{
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  std::vector<Point<dim>> mesh_(solution.size());
  std::vector<double> solution_(solution.size());
  std::vector<int> types_;
  std::vector<std::vector<int> > cells_;
  
  
  const unsigned int   dofs_per_cell   = fe.dofs_per_cell; 
  std::vector<double> LocalDiscChi;
  std::vector<int> NodesAtInterface;
  unsigned int cell_counter = 0;
  unsigned int entries_in_cell_vertex_numbers = 0;

  //Define auxiliary quadrature formula consisting of vertices  
  Quadrature<dim> VertexFormula;
  std::vector<Point<dim> > quad_points;
  std::vector<double> quad_weights;
  Point<dim> vertex;
  
  vertex[0]=0.; vertex[1]=0.;
  quad_points.push_back(vertex);
  quad_weights.push_back(1.);
  
  vertex[0]=1.; vertex[1]=0.;
  quad_points.push_back(vertex);
  quad_weights.push_back(1.);
  
  vertex[0]=0.; vertex[1]=1.;
  quad_points.push_back(vertex);
  quad_weights.push_back(1.);
  
  vertex[0]=1.; vertex[1]=1.;
  quad_points.push_back(vertex);
  quad_weights.push_back(1.);
  
  vertex[0]=0.; vertex[1]=0.5;
  quad_points.push_back(vertex);
  quad_weights.push_back(1.);
  
  vertex[0]=1.; vertex[1]=0.5;
  quad_points.push_back(vertex);
  quad_weights.push_back(1.);
  
  vertex[0]=0.5; vertex[1]=0.;
  quad_points.push_back(vertex);
  quad_weights.push_back(1.);
  
  vertex[0]=0.5; vertex[1]=1.;
  quad_points.push_back(vertex);
  quad_weights.push_back(1.);
  
  vertex[0]=0.5; vertex[1]=0.5;
  quad_points.push_back(vertex);
  quad_weights.push_back(1.);
  
  VertexFormula.initialize(quad_points,quad_weights);
  
  
  LocModFEValues<dim> fe_values (fe, VertexFormula,
				 _hierarchical, update_values | update_quadrature_points  |
				 update_JxW_values | update_gradients);
  
  FullMatrix<double> M(dim, dofs_per_cell);
  
  for (; cell!=endc; ++cell, cell_counter++)
    { 
      std::vector<unsigned int> local_dof_indices (dofs_per_cell);
      
      unsigned int femtype = 0;
      init_FEM (cell,cell_counter,M,dofs_per_cell,femtype, LocalDiscChi, NodesAtInterface);
      
      //Get values of solution
      Vector<double> dof_values(dofs_per_cell);
      cell->get_dof_values(solution,dof_values);
      cell->get_dof_indices (local_dof_indices);
      
      if (_hierarchical) {
	
            fe_values.SetFemtypeAndQuadrature(VertexFormula, femtype, M);
            
            std::vector<double> J;
            fe_values.reinit(J);
                
            for (unsigned int q_point=0; q_point<VertexFormula.size(); ++q_point)
              {
                double sol_in_q_point=0;
                
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                      sol_in_q_point+= dof_values[i]
                    * fe_values.shape_value(i,q_point);
                    }
                  
                 solution_[local_dof_indices[q_point]] = sol_in_q_point;
              }

      } else {
        
        //non-hierarchical: Values in vertices are dofs
        for (unsigned int i=0; i<dofs_per_cell; ++i)
    	  {
	        solution_[local_dof_indices[i]] = dof_values[i];
	      }
	
	
      }
      
   	Point<dim> tmp_p;
	
	for (unsigned int i=0; i<dofs_per_cell; ++i)
	  {
	    tmp_p[0] = M[0][i];
	    tmp_p[1] = M[1][i];
	    mesh_[local_dof_indices[i]] = tmp_p;
	  }
	
	//Now define variables specifying the mesh
	std::vector<int> cell_vertex_numbers;
	switch (femtype)
	  {
	  case 0 : 
	    for (int i=0;i<4;++i)
	    types_.push_back(9); 
	    
	    cell_vertex_numbers.push_back(4);
	    cell_vertex_numbers.push_back(local_dof_indices[0]);
	    cell_vertex_numbers.push_back(local_dof_indices[6]);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[4]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(4);
	    cell_vertex_numbers.push_back(local_dof_indices[6]);
	    cell_vertex_numbers.push_back(local_dof_indices[1]);
	    cell_vertex_numbers.push_back(local_dof_indices[5]);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(4);
	    cell_vertex_numbers.push_back(local_dof_indices[4]);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[7]);
	    cell_vertex_numbers.push_back(local_dof_indices[2]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(4);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[5]);
	    cell_vertex_numbers.push_back(local_dof_indices[3]);
	    cell_vertex_numbers.push_back(local_dof_indices[7]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    // 20 = 4 * 5
	    entries_in_cell_vertex_numbers += 20;
	    
	    break; 
	default :  //Case 1 and 11 
	  for (int i=0;i<8;++i)
	    types_.push_back(5);
	  
	  cell_vertex_numbers.push_back(3);
	  cell_vertex_numbers.push_back(local_dof_indices[0]);
	  cell_vertex_numbers.push_back(local_dof_indices[6]);
	  cell_vertex_numbers.push_back(local_dof_indices[8]);
	  
	  cells_.push_back(cell_vertex_numbers);
	  cell_vertex_numbers.clear();
	  
	  cell_vertex_numbers.push_back(3);
	  cell_vertex_numbers.push_back(local_dof_indices[0]);
	  cell_vertex_numbers.push_back(local_dof_indices[8]);
	  cell_vertex_numbers.push_back(local_dof_indices[4]);
	  
	  cells_.push_back(cell_vertex_numbers);
	  cell_vertex_numbers.clear();
	  
	  cell_vertex_numbers.push_back(3);
	  cell_vertex_numbers.push_back(local_dof_indices[6]);
	  cell_vertex_numbers.push_back(local_dof_indices[1]);
	  cell_vertex_numbers.push_back(local_dof_indices[8]);
	  
	  cells_.push_back(cell_vertex_numbers);
	  cell_vertex_numbers.clear();
	  
	  cell_vertex_numbers.push_back(3);
	  cell_vertex_numbers.push_back(local_dof_indices[1]);
	  cell_vertex_numbers.push_back(local_dof_indices[5]);
	  cell_vertex_numbers.push_back(local_dof_indices[8]);
	  
	  cells_.push_back(cell_vertex_numbers);
	  cell_vertex_numbers.clear();
	  
	  cell_vertex_numbers.push_back(3);
	  cell_vertex_numbers.push_back(local_dof_indices[8]);
	  cell_vertex_numbers.push_back(local_dof_indices[5]);
	  cell_vertex_numbers.push_back(local_dof_indices[3]);
	  
	  cells_.push_back(cell_vertex_numbers);
	  cell_vertex_numbers.clear();
	  
	  cell_vertex_numbers.push_back(3);
	  cell_vertex_numbers.push_back(local_dof_indices[8]);
	  cell_vertex_numbers.push_back(local_dof_indices[3]);
	  cell_vertex_numbers.push_back(local_dof_indices[7]);
	  
	  cells_.push_back(cell_vertex_numbers);
	  cell_vertex_numbers.clear();
	  
	  cell_vertex_numbers.push_back(3);
	  cell_vertex_numbers.push_back(local_dof_indices[8]);
	  cell_vertex_numbers.push_back(local_dof_indices[7]);
	  cell_vertex_numbers.push_back(local_dof_indices[2]);
	  
	  cells_.push_back(cell_vertex_numbers);
	  cell_vertex_numbers.clear();
	  
	  cell_vertex_numbers.push_back(3);
	  cell_vertex_numbers.push_back(local_dof_indices[8]);
	  cell_vertex_numbers.push_back(local_dof_indices[2]);
	  cell_vertex_numbers.push_back(local_dof_indices[4]);
	  
	  cells_.push_back(cell_vertex_numbers);
	  cell_vertex_numbers.clear();
	  
	  // 8 * 4 = 32
	  entries_in_cell_vertex_numbers += 32;
	  
	  break; 
	  case 2 : 
	    for (int i=0;i<8;++i)
	      types_.push_back(5); 

	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[0]);
	    cell_vertex_numbers.push_back(local_dof_indices[6]);
	    cell_vertex_numbers.push_back(local_dof_indices[4]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[6]);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[4]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[6]);
	    cell_vertex_numbers.push_back(local_dof_indices[1]);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[1]);
	    cell_vertex_numbers.push_back(local_dof_indices[5]);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[5]);
	    cell_vertex_numbers.push_back(local_dof_indices[7]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[5]);
	    cell_vertex_numbers.push_back(local_dof_indices[3]);
	    cell_vertex_numbers.push_back(local_dof_indices[7]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[7]);
	    cell_vertex_numbers.push_back(local_dof_indices[2]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[2]);
	    cell_vertex_numbers.push_back(local_dof_indices[4]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    // 8 * 4 = 32
	    entries_in_cell_vertex_numbers += 32;
	    
	    break; 
	  case 3 : 
	    for (int i=0;i<8;++i)
	      types_.push_back(5); 
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[0]);
	    cell_vertex_numbers.push_back(local_dof_indices[6]);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[0]);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[4]);
	  
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[6]);
	    cell_vertex_numbers.push_back(local_dof_indices[1]);
	    cell_vertex_numbers.push_back(local_dof_indices[5]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();

	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[6]);
	    cell_vertex_numbers.push_back(local_dof_indices[5]);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[5]);
	    cell_vertex_numbers.push_back(local_dof_indices[3]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[3]);
	    cell_vertex_numbers.push_back(local_dof_indices[7]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[8]);
	    cell_vertex_numbers.push_back(local_dof_indices[7]);
	    cell_vertex_numbers.push_back(local_dof_indices[4]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    cell_vertex_numbers.push_back(3);
	    cell_vertex_numbers.push_back(local_dof_indices[7]);
	    cell_vertex_numbers.push_back(local_dof_indices[2]);
	    cell_vertex_numbers.push_back(local_dof_indices[4]);
	    
	    cells_.push_back(cell_vertex_numbers);
	    cell_vertex_numbers.clear();
	    
	    // 8 * 4 = 32
	    entries_in_cell_vertex_numbers += 32;
	    
	    break; 
	  }
	
	

    } // end cell
  
  
  //Write to vtk file
  std::ostringstream filename;
  filename << output_basis_filename
	   << Utilities::int_to_string (refinement_cycle, 4)
	   << ".vtk";
  
  
  std::ofstream out (filename.str().c_str());
  out<<"# vtk DataFile Version 3.0"<<std::endl;
  out<<"#vtk This file was generated by the deal.II library"<<std::endl;
  out<<"ASCII"<<std::endl;
  out<<"DATASET UNSTRUCTURED_GRID"<<std::endl<<std::endl;
  
  unsigned int m = mesh_.size();
  
  //Vertices
  out<<"POINTS "<< m <<" double"<<std::endl;
  
  for (unsigned int q=0; q<m; ++q)
    {
      out << mesh_[q];
      if (dim==2)
        out<<" 0"<<std::endl;
      else
        {
          std::cout<<"Only 2d case is implemented in this code"<<std::endl;
          assert(0);
        }
    }
  
  //Sub-cells
  out<<std::endl<<"CELLS "<< types_.size() <<" " << entries_in_cell_vertex_numbers <<std::endl;
  
  for (unsigned int i=0;i<cells_.size(); i++)
    {
      for (unsigned int k=0;k<cells_[i].size(); k++)
	    out << cells_[i][k] << " ";
      
        out << std::endl;
    }
  
  //Type: Triangles/quadrilaterals
  out<<std::endl<<"CELL_TYPES" <<" "<< types_.size() <<std::endl;
  
  for (unsigned int q=0; q<types_.size(); ++q)
    out<< types_[q] << " ";
  
  
  //Values of the solution
  out<<std::endl<<std::endl<<"POINT_DATA " << m <<std::endl;
  out<<"SCALARS solution double 1"<<std::endl;
  out<<"LOOKUP_TABLE default"<<std::endl;
  
  if (m!=solution_.size())
    {
      std::cout<<"solution- and mesh-vector don't match"<<std::endl;
      assert(0);
    }
  
  for (unsigned int q=0; q<solution_.size(); ++q)
    out << solution_[q] << std::endl;
  
}

// Computes the local L2 norm \| u -u_h\|_K 
// or H1 seminorm \| nabla (u -u_h)\|_K n each cell K
// The basic function has been copied 
// from the deal.II library and then slightly modified. 
// See also possibly step-7 
template <int dim>
void LocModFE<dim>::integrate_difference_norms (const DoFHandler<dim> &dof,
						const FiniteElement<dim> &fe,
						const Vector<double> &fe_function,
						ManufacturedSolution<dim> exact_solution,
						Vector<float> &difference,
						const Function<dim> *weight,
						const std::string norm_string)
{
  
  const unsigned int        n_components = dof.get_fe().n_components();
  const bool                fe_is_system = (n_components != 1);
  
  difference.reinit (dof.get_triangulation().n_active_cells());
  
  UpdateFlags update_flags = UpdateFlags (update_quadrature_points  |
					  update_JxW_values);
  
  if (norm_string == "L2")
    update_flags |= UpdateFlags (update_values);
  else if (norm_string == "H1_semi")
    update_flags |= UpdateFlags (update_gradients);
  else if (norm_string == "H1")
    {
      update_flags |= UpdateFlags (update_values);
      update_flags |= UpdateFlags (update_gradients);
    }
  
  // loop over all cells
  typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active(),
    endc = dof.end();
  
  unsigned int dofs_per_cell = 9;
  
  std::vector<double> LocalDiscChi;
  std::vector<int> NodesAtInterface; 
  double ChiValue=0.;
  FullMatrix<double> M(dim, dofs_per_cell);
  
  Quadrature<dim> quadrature_form = compute_quadrature(1);
  const unsigned int max_n_q_points = quadrature_form.size();   
  
  std::vector< dealii::Vector<double> >
    function_values (max_n_q_points, dealii::Vector<double>(n_components));
  std::vector<std::vector<Tensor<1,dim> > >
    function_grads (max_n_q_points, std::vector<Tensor<1,dim> >(n_components));
  
  std::vector<double>
    weight_values (max_n_q_points);
  std::vector<dealii::Vector<double> >
    weight_vectors (max_n_q_points, dealii::Vector<double>(n_components));
  
  std::vector<dealii::Vector<double> >
    psi_values (max_n_q_points, dealii::Vector<double>(n_components));
  std::vector<std::vector<Tensor<1,dim> > >
    psi_grads (max_n_q_points, std::vector<Tensor<1,dim> >(n_components));
  
  std::vector<std::vector<Tensor<1,dim> > >
    psi_grads_tmp (max_n_q_points, std::vector<Tensor<1,dim> >(n_components));
  
  
  std::vector<double>
    psi_scalar (max_n_q_points);
  
  // tmp vector when we use the Function<dim> functions for
  // scalar functions
  std::vector<double>         tmp_values (max_n_q_points);
  std::vector<Tensor<1,dim> > tmp_gradients (max_n_q_points);
  
  Point<dim> IntPoint;
  
  std::vector<double> J (max_n_q_points); 
  unsigned cell_counter = 0;
  
  Quadrature<dim> quadrature_formula0 = compute_quadrature(0);
  Quadrature<dim> quadrature_formula1 = compute_quadrature(1);
  
  LocModFEValues<dim>* fe_values;
  LocModFEValues<dim> fe_values0 (fe, quadrature_formula0,
				  _hierarchical, update_values | update_quadrature_points  |
				  update_JxW_values | update_gradients);
  
  LocModFEValues<dim> fe_values1 (fe, quadrature_formula1,
				  _hierarchical, update_values | update_quadrature_points  |
				  update_JxW_values | update_gradients);
  
  for (; cell != endc; ++cell, ++cell_counter) {
    
    double diff=0;
    
    unsigned int femtype = 0;
    init_FEM (cell,cell_counter,M,dofs_per_cell,femtype, LocalDiscChi, NodesAtInterface);
    
    Quadrature<dim> quadrature_formula = compute_quadrature(femtype);
    //const unsigned int max_n_q_points = quadrature_formula.size();   
    
    if (femtype==0) fe_values = &fe_values0;
    else fe_values = &fe_values1;
    
    fe_values->SetFemtypeAndQuadrature(quadrature_formula, femtype, M);
    
    fe_values->reinit(J);
    
    const unsigned int   n_q_points = fe_values->n_quadrature_points;
    
    // resize all out scratch
    // arrays to the number of
    // quadrature points we use
    // for the present cell
    function_values.resize (n_q_points,
			    dealii::Vector<double>(n_components));
    function_grads.resize (n_q_points,
			   std::vector<Tensor<1,dim> >(n_components));
    
    weight_values.resize (n_q_points);
    weight_vectors.resize (n_q_points,
			   dealii::Vector<double>(n_components));
    
    psi_values.resize (n_q_points,
		       dealii::Vector<double>(n_components));
    psi_grads.resize (n_q_points,
		      std::vector<Tensor<1,dim> >(n_components));
    
    psi_grads_tmp.resize (n_q_points,
			  std::vector<Tensor<1,dim> >(n_components));
    psi_scalar.resize (n_q_points);
    
    tmp_values.resize (n_q_points);
    tmp_gradients.resize (n_q_points);
    
  if (weight!=0)
  {
	if (weight->n_components>1)
	  weight->vector_value_list (fe_values->get_quadrature_points(),
				     weight_vectors);
	else
	  {
	    weight->value_list (fe_values->get_quadrature_points(),
				weight_values);
	    for (unsigned int k=0; k<n_q_points; ++k)
	      weight_vectors[k] = weight_values[k];
	  }
  }
  else
  {
	for (unsigned int k=0; k<n_q_points; ++k)
	  weight_vectors[k] = 1.;
  }
    
    
  if (update_flags & update_values)
  {
	// Get the exact solution
	if (fe_is_system)
	  exact_solution.vector_value_list (fe_values->get_quadrature_points(),
					    psi_values);
	else
	  {
	    exact_solution.value_list (fe_values->get_quadrature_points(),
				       tmp_values);
	    for (unsigned int q=0; q<n_q_points; ++q) {
	      //For ModFE, modify the integration point
	      ComputePhysicalIntegrationPoint(IntPoint, *fe_values, M, dofs_per_cell,q);
	      ComputeLocalDiscChi(ChiValue, q, *fe_values, dofs_per_cell, LocalDiscChi);
	      exact_solution.SetDiscChi(ChiValue);
	      
	      psi_values[q](0) = exact_solution.value(IntPoint,0);
	    }
	  }
	
	// Subtract finite element
	// fe_function
	fe_values->get_function_values (cell,fe_function, function_values);
	
	for (unsigned int q=0; q<n_q_points; ++q)
	  psi_values[q] -= function_values[q];
	
	// Take square of integrand
	std::fill_n (psi_scalar.begin(), n_q_points, 0.0);
	for (unsigned int k=0; k<n_components; ++k)
	  for (unsigned int q=0; q<n_q_points; ++q)
	    psi_scalar[q] += (psi_values[q][k] * psi_values[q][k])
	      * weight_vectors[q](k);
	
	
	for (unsigned int q_int=0; q_int<n_q_points; ++q_int)
	  {
	    diff += psi_scalar[q_int] * J[q_int] * quadrature_formula.weight(q_int);
	  }
             
	}
    
    
    
    // Do the same for gradients for the H1 norm
  if (update_flags & update_gradients)
	{
	// Get the exact solution gradients
	if (fe_is_system)
	  {
	    exact_solution.vector_gradient_list (fe_values->get_quadrature_points(),
						 psi_grads_tmp);
	    
	    for (unsigned int k=0; k<n_components; ++k)
	      for (unsigned int q=0; q<n_q_points; ++q)
			 {
				psi_grads[q][k] = psi_grads_tmp[q][k];
		  
			 }
		}
	else
	  {
	    exact_solution.gradient_list (fe_values->get_quadrature_points(),
					  tmp_gradients);
	    for (unsigned int q=0; q<n_q_points; ++q) {
	      //For ModFE, modify the integration point
	      ComputePhysicalIntegrationPoint(IntPoint, *fe_values, M, dofs_per_cell,q);
	      
	      ComputeLocalDiscChi(ChiValue, q, *fe_values, dofs_per_cell, LocalDiscChi);
	      exact_solution.SetDiscChi(ChiValue);

	      
	      psi_grads[q][0] = exact_solution.gradient(IntPoint,0);
	    }
	    
	  }
	
	
	fe_values->get_function_gradients (cell,fe_function, function_grads);
	
	for (unsigned int k=0; k<n_components; ++k)
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {
	      psi_grads[q][k] -= function_grads[q][k];
	    }
	
	
	// take square of integrand
	std::fill_n (psi_scalar.begin(), n_q_points, 0.0);
	for (unsigned int k=0; k<n_components; ++k)
	  for (unsigned int q=0; q<n_q_points; ++q)
	    psi_scalar[q] += (psi_grads[q][k] * psi_grads[q][k])
	      * weight_vectors[q](k);
	
	
	for (unsigned int q_int=0; q_int<n_q_points; ++q_int)
	  {
	    diff += psi_scalar[q_int] * J[q_int] * quadrature_formula.weight(q_int);
	  }
	
  } // end (update_flags & gradients)
    
  diff = std::sqrt(diff);
  difference(cell_counter) = diff;
 }
  
}


//Function interpolate_boundary_values has to be modified due to 
//(possible) moved vertices at the boundary and due to the hierarchical basis
//(which is non-primitive)
template <int dim>
void
LocModFE<dim>::interpolate_boundary_values(const DoFHandler<dim> &dof_handler,
   const types::boundary_id                               boundary_component,
   const Function<dim,double>  				  &BP,
   std::map<types::global_dof_index,double>               &boundary_values)
{

  std::vector<types::global_dof_index> face_dofs;
  typename FunctionMap<dim>::type function_map;
  function_map[0] = &BP;
  
  std::vector<double>          dof_values_scalar;
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  const FiniteElement<dim> &fe0 = cell->get_fe();
  
  std::vector<double> LocalDiscChi;
  std::vector<int> NodesAtInterface;
  
  int cell_counter=0;
  FullMatrix<double> M(dim, fe0.dofs_per_cell);
  unsigned int femtype = 0;
  
  //Define arrays of nodes on the four edges
  int e2n[4][3] = { { 0, 2, 4 },  { 1, 3, 5 }, { 0, 1, 6 }, { 2, 3, 7 } };
  
  for (; cell!=endc; ++cell, ++cell_counter) {
    
    //Check, which vertices are moved
    init_FEM (cell,cell_counter,M,fe0.dofs_per_cell,femtype, LocalDiscChi, NodesAtInterface);
    
    const FiniteElement<dim> &fe = cell->get_fe();
    
    
    for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell;
       ++face_no)
      {
      const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);
      const types::boundary_id boundary_comp = face->boundary_id();
        
      if (boundary_component==boundary_comp) {
        face_dofs.resize (fe.dofs_per_face);
        face->get_dof_indices (face_dofs, cell->active_fe_index());
        std::vector<Point<2> > dof_locations;
        dof_locations.resize(fe.dofs_per_face);
          
        //Set dof_locations to the (possibly) moved vertices
        for (unsigned int i=0; i< fe.dofs_per_face; i++) {
          dof_locations[i][0] = M[0][e2n[face_no][i]];
          dof_locations[i][1] = M[1][e2n[face_no][i]];
        }              
          
          
        dof_values_scalar.resize (fe.dofs_per_face);
        function_map.find(boundary_component)->second
          ->value_list (dof_locations, dof_values_scalar, 0);
        
          
        //Calculate relative position of midpoint of edge
        double outer1x = M[0][e2n[face_no][0]];
        double outer2x = M[0][e2n[face_no][1]];
        double midx = M[0][e2n[face_no][2]];
        double outer1y = M[1][e2n[face_no][0]];
        double outer2y = M[1][e2n[face_no][1]];
        double midy = M[1][e2n[face_no][2]];
        double relative_pos;
              
        //As the line is boundary line is straight, the relative position
        //can be calculated based on any coordinate, unless one of them is constant
        if (fabs(outer2x-outer1x)>fabs(outer2y-outer1y)) {
          relative_pos = (midx-outer1x)/(outer2x-outer1x);
        } else {
          relative_pos = (midy-outer1y)/(outer2y-outer1y);
        }
          
        //Substitute linear interpolation in V_2h      
        if (_hierarchical) 
          dof_values_scalar[2] -= (1.-relative_pos)*dof_values_scalar[0]
            + relative_pos*dof_values_scalar[1];
        
        // enter into list
        for (unsigned int i=0; i<face_dofs.size(); ++i) {
          boundary_values[face_dofs[i]] = dof_values_scalar[i];
        }
      }
    }
  }
  
  
}

template class LocModFEValues<2,2>;
template class LocModFE<2>;
