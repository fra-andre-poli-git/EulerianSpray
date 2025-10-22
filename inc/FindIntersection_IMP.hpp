// This file contains the implementation of template functions in
// VectorProjection.hpp

#include<deal.II/base/tensor.h>






// This function finds s, the intersection between the segment joining q and
// w and the line rho * (b + epsilon) or the line rho * (a - epsilon). It is the 1d
// version.
// NOTE: here dim is the phase space dimension, not the physical dimension 
// (therefore is physical dimension plus one)
template<int dim> double find_intersection_1d(
  const dealii::Tensor<1, dim, Number> & q,
  const dealii::Tensor<1, dim, Number> & w,
  const Number & epsilon,
  const Number & a,
  const Number & b)
{

  // True if I am above the line rho * (b - \epsilon)
  bool above = (q[1] > q[0] * (b + epsilon));
  // True if I am below the line rho * (a + \epsilon)
  bool below = (q[1] < q[0] * (a - epsilon));

  // If the solution is already in G_\epsilon theta is = 1
  if (!above && !below)
    return 1.0;

  double y_edge = above ? (b+epsilon) : (a-epsilon);

  auto f = [&](double t)->double {
    const Number rho = w[0] + t*(q[0] - w[0]);
    const Number m   = w[1] + t*(q[1] - w[1]);
    return static_cast<double>(m - rho * y_edge);
  };

  double t0 = 0.0, t1 = 1.0;
  double f0 = f(t0), f1 = f(t1); 

  const int max_it = 50;
  const double tol = 1e-12;
  double t = 0.5;
  // if(f0 * f1 >0)
  //   return 0.0;
  Assert(f0 * f1 <= 0.0,
          ExcMessage("Problem inside FindIntersection: cell average results outside the realizability boundary"));
  // else // therefore (f0 * f1 <= 0.0) 
  {
    for (int it=0; it<max_it; ++it)
    {
      double tm = 0.5*(t0 + t1);
      double fm = f(tm);
      if (!std::isfinite(fm))
      {
        t = 0.5;
        break;
      }
      if ((t1 - t0)*0.5 < tol || std::abs(fm) < 1e-14)
      {
        t = tm;
        break;
      }
      if(f0 * fm <= 0.0)
      {
        t1 = tm;
        f1 = fm;
      }
      else
      {
        t0 = tm;
        f0 = fm;
      }
      t = tm;
    }
  }

  return t;
}

template<int dim> double find_intersection(
  dealii::Tensor<1, dim, Number> q,
  dealii::Tensor<1, dim, Number> w,
  double epsilon,
  double s)
{
  double theta = 1.0;

  return theta;
}