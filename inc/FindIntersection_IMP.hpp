// This file contains the implementation of template functions in
// VectorProjection.hpp

#include<deal.II/base/tensor.h>






// This function finds s, the intersection between the segment joining q and
// w and the line rho * (b-epsilon) or the line rho * (a+epsilon). It is the 1d
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
  double theta = 1.0;

  // True if I am above the line rho * (b - \epsilon)
  bool above = (q[1] > q[0] * (b - epsilon));
  // True if I am below the line rho * (a + \epsilon)
  bool below = (q[1] < q[0] * (a + epsilon));

  // If the solution is already in G_\epsilon theta is = 1
  if (!above && !below)
    return 1.0;

  Tensor<2, 2, Number> A;
  Tensor<1, 2, Number> s;


  A[0][0] = w[0] - q[0];
  A[1][0] = w[1] - q[1];
  A[0][1] = 1.0;
  A[1][1] = above ? (b - epsilon) : (a + epsilon);


  const Number detA = determinant(A);


  if (std::abs(detA) < 1e-13)
  {
    std::cout<< "Warning: det(A) = " << detA << std::endl;
    theta = 1;
  }
  else
  { 
    Tensor<1, 2, Number> q_truncated;
    Tensor<1, 2, Number> w_truncated;
    w_truncated[0] = w[0];  
    w_truncated[1] = w[1];
    q_truncated[0] = q[0];  
    q_truncated[1] = q[1];
    s = invert(A) * w_truncated;
    Tensor<1, 2, Number> diff_ws = w_truncated - s;
    Tensor<1, 2, Number> diff_wq = w_truncated - q_truncated;
    if (diff_wq.norm() < 1e-14)
      std::cout << "Warning: w ≈ q" << std::endl;
    theta = diff_ws.norm() / diff_wq.norm();
  }

  return theta;
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