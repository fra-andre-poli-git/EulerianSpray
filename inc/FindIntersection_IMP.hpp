// This file contains the implementation of template functions in
// VectorProjection.hpp






// This function finds s, the intersection between the segment joining q and
// w and the line rho * (b-epsilon) or the line rho * (a+epsilon). It is the 1d
// version.
// NOTE: here dim is the phase space dimension, not the physical dimension 
// (therefore is physical dimension plus one)
template<int dim> dealii::Vector<dim> find_intersection_1d(
  dealii::Vector<dim> q,
  dealii::Vector<dim> w,
  double epsilon
  double a,
  double b)
{
  dealii::Vector<dim+1> s;
  if(q[1] > q[0]*(b-epsilon)) // I am above the line rho * (b - \epsilon)
  {

  }
  else if(q[1] < q[0] * (a + epsilon)) // I am below the line rho * (a + \epsilon)
  {
    
  }
  else
    s = q;
  return s;
}

