// This files contains the functions that find the intersection between the
// segment linking s (vector of the solution when the density is already 
// projected on rho = \epsilon) and the average solution on the cel \bar{w}

#ifndef FIND_INTERSECTION_HPP
#define FIND_INTERSECTION_HPP

#include"TypesDefinition.hpp"


template<int dim> dealii::Tensor<1, dim, myReal> find_intersection_1d(
  const dealii::Tensor<1, dim, myReal> & q,
  const dealii::Tensor<1, dim, myReal> & w,
  const myReal & epsilon,
  const myReal & a,
  const myReal & b);

template<int dim> double find_intersection(
  dealii::Tensor<1, dim, myReal>  q,
  dealii::Tensor<1, dim, myReal>  w,
  double epsilon,
  double s);
  
#include "FindIntersection_IMP.hpp"
#endif