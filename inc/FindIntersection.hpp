// This files contains the functions that find the intersection between the
// segment linking s (vector of the solution when the density is already 
// projected on rho = \epsilon) and the average solution on the cel \bar{w}

#ifndef FIND_INTERSECTION_HPP
#define FIND_INTERSECTION_HPP

template<int dim> dealii::Vector<dim> find_intersection_1d(
  dealii::Vector<dim> q,
  dealii::Vector<dim> w,
  double epsilon
  double a,
  double b);

template<int dim> dealii::Vector<dim> find_intersection(
  dealii::Vector<dim> q,
  dealii::Vector<dim> w,
  double epsilon
  double s);
  
#include "FindIntersection_IMP.hpp"
#endif