#ifndef _TOOLS
#define _TOOLS

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

/** This file contains the problem-specific definition of geometry, boundary conditions and 
 *  the analytical solution, implemented in the following classes:
 *
 *    a) class LevelSet : Implicit definition of the interface and the sub-domains 
 *    b) classDirichletBoundaryConditions : Definition of the Dirichlet data
 *    c) class ManufacturedSolution : Analytical solution for error estimation 
*/


// Include files
//--------------
#include <deal.II/base/function.h>

// Important global namespaces from deal.II and C++				
using namespace dealii;
using namespace std;


//In the class LevelSet, we define a function, whose sign defines the 
//sub-domains Omega_1 and Omega_2. The interface between them is defined by
//the zero level-set. In this special case, the level-set function depends
//on a parameter _yoffset, defining the vertical position of the circular
//interface
template <int dim> class LevelSet
{
 private:
  double _yoffset;
  
 public:
  
  // Compute value of the LevelSet function in a point p
  double dist(const Point<dim> p) const
  {
    return p(0)*p(0) + (p(1)-_yoffset)*(p(1)-_yoffset) -0.5*0.5;
  }

  // Derivatives for Newton's method to find cut position
  double dist_x(const Point<dim> p) const
  {
    return 2.0*p(0);
  }
  double dist_y(const Point<dim> p) const
  { 
    return 2.0*(p(1)-_yoffset);
  }
  
  //Determine domain affiliation of a point p
  int domain(const Point<dim> p) const
  {
    double di = dist(p);
    if (di>=0) return 1;
    else return -1;
  }

  double get_y_offset () const
  {
    return _yoffset;
  }

  void set_y_offset (const double yoffset)
  {
    _yoffset = yoffset;
  }
 
};


// In this class, we define a function
// that deals with the boundary values.			
template <int dim>
class DirichletBoundaryConditions : public Function<dim> 
{
 private:
  double _visc_1;
  double _visc_2;
  double _yoffset;
  
 public:
  DirichletBoundaryConditions (const double visc_1, const double visc_2, const double yoffset)    
    : Function<dim>(1) 
    {
      _visc_1 = visc_1;
      _visc_2 = visc_2;
      _yoffset = yoffset;
    }
    
  // In this example, we specify boundary values for component 0 only
  double value (const Point<dim>  &p,
		const unsigned int component) const
  {
    if (component == 0)
      {
        return -1.0 * _visc_1 * (p(0) * p(0) + (p(1)-_yoffset) * (p(1)-_yoffset)) + 0.25 * _visc_1 - 0.125 * _visc_2;
      }
    
    return 0.0;
  }
  
};

/**************************************************************************/


//This class sets the analytical solution for error estimation
template <int dim>
class ManufacturedSolution : public Function<dim>
  {
  
  private:
    double _visc_1;
    double _visc_2;
    double _ChiValue;
    double _yoffset;
    
  public:
    //Constructor
    ManufacturedSolution (const double visc_1, const double visc_2, const double yoffset)
      : Function<dim>(1)
      {
        _visc_1 = visc_1;
        _visc_2 = visc_2;
        _ChiValue=0.;
        _yoffset = yoffset;
      }
    
    //Set discrete LevelSet function chi_h
    void SetDiscChi(double DiscChi) 
    {
      _ChiValue=DiscChi;
    }
    
    
    virtual double value (const Point<dim> &p, const unsigned int component = 0) const
    {
      double return_value = 0.0;
      if (component == 0)
      {
        if (_ChiValue < 0.) // inner domain
           return_value = -2.0 * _visc_2 * (p(0) * p(0) + (p(1)-_yoffset) * (p(1)-_yoffset)) 
                        * (p(0) * p(0) + (p(1)-_yoffset) * (p(1)-_yoffset));
        else // outer domain
           return_value = -1.0 * _visc_1 * (p(0) * p(0) + (p(1)-_yoffset) * (p(1)-_yoffset)) 
                          +0.25 * _visc_1 - 0.125 * _visc_2;
  
        return return_value;
      }
      
      return 0.0;
    }
    
    //Derivatives for H1-norm error
    virtual Tensor<1,dim> 
      gradient (const Point<dim>   &p, const unsigned int  component = 0) const
      {
         Tensor<1,dim> return_value;
         return_value.clear();

         if (component == 0)
         {
    
            if (_ChiValue < 0.) 
            {
               // inner domain
               double r2 = p(0) * p(0) + (p(1)-_yoffset) * (p(1)-_yoffset);
               return_value[0] = -8.0 * _visc_2 * r2 * p(0);
               return_value[1] = -8.0 * _visc_2 * r2 * (p(1)-_yoffset);
            }
            else 
            {
               // outer domain
               return_value[0] = -2.0 * _visc_1 * p(0);
               return_value[1] = -2.0 * _visc_1 * (p(1)-_yoffset);
            }   
            return return_value;	  
         }      

         return return_value;
      } 
    
  };


#endif
