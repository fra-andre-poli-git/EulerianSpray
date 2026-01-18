// This file contains the implementation of template functions in
// VectorProjection.hpp

#include<deal.II/base/tensor.h>





// This function finds s, the intersection between the segment joining q and
// w and the line rho * (b + epsilon) or the line rho * (a - epsilon). It is the 1d
// version.
// NOTE: here dim is the phase space dimension, not the physical dimension 
// (therefore is physical dimension plus one)
template<int dim> dealii::Tensor<1, dim, myReal> find_intersection_1d(const dealii::Tensor<1, dim, myReal> & q, const dealii::Tensor<1, dim, myReal> & w,
  const myReal & epsilon,
  const myReal & a,
  const myReal & b)
{

  // True if I am above the line rho * b - \epsilon
  bool above = (q[1] > q[0] * (b + epsilon));
  // True if I am below the line rho * a + \epsilon
  bool below = (q[1] < q[0] * (a - epsilon));

  // If the solution is already in G_\epsilon theta is = 1
  if (!above && !below)
    return q;

  double edge = above ? b + epsilon : a - epsilon;

  auto f = [&](double t)->double
  {
    // const myReal rho = w[0] + t*(q[0] - w[0]);
    // const myReal m   = w[1] + t*(q[1] - w[1]);
    const myReal rho = (1.-t) * w[0] + t * q[0];
    const myReal m = (1.-t) * w[1] + t * q[1];
    return static_cast<double>(m - rho * edge);
  };

  double t0 = 0.0, t1 = 1.0;
  double f0 = f(t0), f1 = f(t1); 

  const int max_it = 100;
  const double tol = 1e-14;
  // double t = 0.5;
  double tm,fm;
  if(f0 * f1 > 0.) // if the velocity are both outside the boundary
  {
    // // If the average is outside G_epsilon (but I know that it is inside G) the 
    // if((w[1] >= w[0] * (b + epsilon)) || (w[1] <= w[0] * (a - epsilon)))
    //   return w;
    // else
    //   Assert(false, ExcMessage("I don't know why it appears both q and w are on the same side of realizability boundary")) 
    return w;
  }
  // Assert(f0 * f1 <= 0.0,
  //         ExcMessage("Problem inside FindIntersection: cell average results outside the realizability boundary"));
  else // therefore (f0 * f1 <= 0.0) 
  {
    for (int it=0; it<max_it; ++it)
    {
      tm = 0.5*(t0 + t1);
      fm = f(tm);
      if ( (t1 - t0)*0.5 < tol )
      {
        // t = tm;
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
      // t = tm;
      if(it+1 == max_it)
        std::cout<<"Warning in FindIntersection: bisection method has not converged"<<std::endl;
    }
  }

  Assert( tm >= 0.0 && tm <= 1.0,
          ExcMessage("Problem in bisection for the research of s: t is "+ std::to_string(tm)));

  return (1-tm) * w + tm * q;
}

template<int dim> dealii::Tensor<1, dim, myReal>
find_intersection_1d_LLF_Variant(const dealii::Tensor<1, dim, myReal> & q, const dealii::Tensor<1, dim, myReal> & w, const myReal & epsilon, const myReal & a, const myReal & b)
{

  // True if I am above the line rho * b - \epsilon
  bool above = (q[1] > q[0] * b - epsilon);
  // True if I am below the line rho * a + \epsilon
  bool below = (q[1] < q[0] * a + epsilon);

  
  if (!above && !below)// If the solution is already in G_\epsilon theta is = 1
    return q;
  if (above && below) //If the solution is in the small triangle is not in G_\epsilon but still in G
    return q;

  double edge = above ? b : a ;

  auto f = [&](double t)->double
  {
    // const myReal rho = w[0] + t*(q[0] - w[0]);
    // const myReal m   = w[1] + t*(q[1] - w[1]);
    const myReal rho = (1.-t) * w[0] + t * q[0];
    const myReal m = (1.-t) * w[1] + t * q[1];
    return static_cast<double>(m - rho * edge + (1-2*static_cast<double>(above))*epsilon);
  };

  double t0 = 0.0, t1 = 1.0;
  double f0 = f(t0), f1 = f(t1); 

  const int max_it = 100;
  const double tol = 1e-14;
  // double t = 0.5;
  double tm,fm;
  if(f0 * f1 > 0.) // if the velocity are both outside the boundary
  {
    // // If the average is outside G_epsilon (but I know that it is inside G) the 
    // if((w[1] >= w[0] * (b + epsilon)) || (w[1] <= w[0] * (a - epsilon)))
    //   return w;
    // else
    //   Assert(false, ExcMessage("I don't know why it appears both q and w are on the same side of realizability boundary")) 
    return w;
  }
  // Assert(f0 * f1 <= 0.0,
  //         ExcMessage("Problem inside FindIntersection: cell average results outside the realizability boundary"));
  else // therefore (f0 * f1 <= 0.0) 
  {
    for (int it=0; it<max_it; ++it)
    {
      tm = 0.5*(t0 + t1);
      fm = f(tm);
      if ( (t1 - t0)*0.5 < tol )
      {
        // t = tm;
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
      // t = tm;
      if(it+1 == max_it)
        std::cout<<"Warning in FindIntersection: bisection method has not converged"<<std::endl;
    }
  }

  Assert( tm >= 0.0 && tm <= 1.0,
          ExcMessage("Problem in bisection for the research of s: t is "+ std::to_string(tm)));

  return (1-tm) * w + tm * q;
}





template<int dim> dealii::Tensor<1, dim, myReal> find_intersection(
  dealii::Tensor<1, dim, myReal> q,
  dealii::Tensor<1, dim, myReal> w,
  double epsilon,
  double S /* max velocity norm*/)
{ 
  // I put this piece in a scope to hide the name momentum_norm and use it in
  // the lambda function
  {
    double frontier = (S + epsilon)*(S + epsilon)*q[0]*q[0];
    double momentum_norm = 0;
    for(unsigned d = 1; d < dim; ++d)
      momentum_norm += (q[d]*q[d]);
    if( momentum_norm <= frontier)
      return q;
  }
  auto f = [&](double t)->double
  {
    const auto s = (1.-t) * w + t * q;
    double momentum_norm = 0.;
    for(unsigned d = 1; d < dim; ++d)
      momentum_norm += (s[d]*s[d]);

    return momentum_norm - (S + epsilon)*(S + epsilon)*s[0]*s[0];
  };

  double t0 = 0., t1 = 1.;
  double f0 = f(t0), f1 = f(t1);

  const int max_it = 100;
  const double tol = 1e-14;
  double tm, fm;
  // Assert(f0 * f1 > 0., ExcMessage("Problem in find_intersection: solution average appears to be outside G region"
  //   " indeed w = [" + std::to_string(w[0]) + ", " + std::to_string(w[1]) + ", " + std::to_string(w[2]) + "], S (max velocity norm) = " +
  //   std::to_string(S) + " and epsilon = " + std::to_string(epsilon) + " and q = ["+ std::to_string(q[0]) + ", " + std::to_string(q[1]) + ", " + std::to_string(q[2]) + "]"));
  if(f0 * f1 > 0.) //if both actual velocity and mean velocity are outside the admissibility region
    return w; // i choose s as the average solution
  else // otherwise I go on with bisection
  {
    for(int it = 0; it < max_it; ++it)
    {
      tm = 0.5 * (t0 + t1);
      fm = f(tm);
      if( (t1 - t0)*0.5 < tol)
        break;
      if( f0 * fm <= 0.)
      {
        t1 = tm;
        f1 = fm;
      }
      else
      {
        t0 = tm;
        f0 = fm;
      }
      if(it+1 == max_it)
        std::cout<<"Warning in FindIntersection: bisection method has not converged"<<std::endl;
    }
  }
  
  Assert( tm >= 0.0 && tm <= 1.0,
          ExcMessage("Problem in bisection for the research of s: t is "+ std::to_string(tm)));
  
  return (1-tm) * w + tm * q;
}