#ifndef FUNCTIONS_IMP_HPP
#define FUNCTIONS_IMP_HPP

#include"Functions.hpp"
#include"TypesDefinition.hpp"

#include<deal.II/base/point.h>
#include<cmath>

template<int dim>
double InitialSolution<dim>::value(const Point<dim> & p,
    const unsigned int component) const
{
  switch(parameters.testcase)
  {
    case 1:
    {
      double sigma = 0.1;
      if(component == 0)
        return 1./2 * exp(-pow((p[0]-1./2),2)/pow(sigma,2));
      if(component == 1)
        return -1./2 * exp(-pow((p[0]-1./2),2)/pow(sigma,2));
    }
    case 2:
    {
      if(component==0)
        return 0.5;
      if(component==1)
        return 0.5*(-0.5*(p[0]<-0.5) + 0.4*(-0.5<=p[0] && p[0]<0) + (0.4-p[0])*
          (0<=p[0] && p[0] <0.8) + -0.4*(p[0]>=0.8));
      // Testcase 1 the dimension is supposed to be 1. However, since output
      // functions are expected to work in dimension bigger than 1, I put this
      // return to make y momentum null.
      return 0.; 
    }
    case 3:
    {
      if(component==0)
        return (p[0]<0.)*1 + (p[0]>=0.)*0.25;
      if(component==1)
        return (p[0]<0.)*1*0.5 + (p[0]>=0.)*0.25*(-0.4);
      return 0.;
    }
    case 5:
    {
      if(component==0)
        return 0.9*(((-0.3<p[0] && p[0] <-0.2) && (-0.15<p[1] && p[1] <0.05)) ||
          ((0.2<p[0] && p[0]<0.3) && (-0.05<p[1]) && (p[1]<0.15))) + 0.1;
      if(component==1)
        return 0.5 * ((-0.3<p[0] && p[0] <-0.2) && (-0.15<p[1] && p[1] <0.05)) +
          -0.5 * ((0.2<p[0] && p[0]<0.3) && (-0.05<p[1]) && (p[1]<0.15));
      return 0.;
    }
    default:
    {
      Assert(false, ExcNotImplemented());
      return 0.;
    }
  }
}

template<int dim>
double FinalSolution<dim>::value(const Point<dim> & p,
  const unsigned int component) const
{
  switch(parameters.testcase)
  {
    case 2:
    {
      if(component == 0)
        return 0.5*(p[0]< -0.75) + 0*(p[0]>= -0.75)*(p[0]< -0.3) 
          + 0.5 * (p[0]>= -0.3 )*(p[0]<0.2) + 1 * (p[0]>=0.2) * (p[0]<0.6)
          + 0.5 * (p[0]>= 0.6 );
      if(component == 1)
        return 0.5*(-0.5)*(p[0]< -0.75) + 0 * (p[0]>= -0.75)*(p[0]< -0.3) 
          + 0.5 * 0.4 *(p[0]>= -0.3 )*(p[0]<0.2) + 1 * (0.4 - p[0]) / 0.5 
          * (p[0]>=0.2) * (p[0]<0.6) + 0.5 * (-0.4) * (p[0]>= 0.6 );
      return 0.;
    }
    case 3:
    {
      return 0.;
    }
    case 4:
    {
      return 0.;
    }
    default:
      Assert(false, ExcNotImplemented());
    return 0.;
  }
}

template<int dim>
double FinalSolutionVelocity<dim>::value(const Point<dim> & p,
  const unsigned int component) const
{
  switch(parameters.testcase)
  {
    case 1:
    {
      return 0.;
    }
    case 2:
    {
      if(component == 0)
        return 0.5*(p[0]< -0.75) + 0*(p[0]>= -0.75)*(p[0]< -0.3) 
          + 0.5 * (p[0]>= -0.3 )*(p[0]<0.2) + 1 * (p[0]>=0.2) * (p[0]<0.6)
          + 0.5 * (p[0]>= 0.6 );
      if(component == 1)
        return -0.5 * (p[0]<-0.5) + 0.4*(p[0]>=-0.5)*(p[0]<0.2) + (0.8-2*p[0]) * 
          (p[0]>=0.2)*(p[0]<0.6) - 0.4 * (p[0]>= 0.6);
      return 0.;
    }
    case 3:
    {
      return 0.;
    }
    case 4:
    {
      return 0.;
    }
    case 5:
    {
      return 0.;
    }
    case 6:
    {
      return 0.;
    }
    case 7:
    {
      return 0.;
    }
    case 8:
    {
      return 0.;
    }
    default:
      Assert(false, ExcNotImplemented());
    return 0.;
  }
}

template<int dim>
double DirichletFunction<dim>::value(const Point<dim> & p,
  const unsigned int component) const
{
  const double t = this->get_time();
  switch(parameters.testcase)
  {
    case 2:
    {
      (void) t;
      if(component == 0)
        return 0.5;
      else if(component == 1)
        return (0.5*(-0.4)*(p[0]>0) + 0.5*(-0.5)*(p[0]<0));
      else
        return 0.;
    }
    default:
      Assert(false, ExcNotImplemented());
    return 0.;
  }
}

// template<int dim>
// double ExactSolution<dim>::value(const Point<dim> & p,
//   const unsigned int component) const
// {
//   const double t = this->get_time();

//   switch(parameters.testcase)
//   {
//     case 2:
//     {
//       if(component==0)
//         return 0.;
//       if(component==1)
//       if(component==2)
//       return 0.;
//     }
//     default:
//       Assert(false, ExcNotImplemented());
//   }
// }


// template<int dim>
// double ExternalFlux<dim>::value(const Point<dim> & p,
//   const unsigned int component) const
// {
//   const double t = this->get_time();

//   switch(parameters.parameters.testcase)
//   {
//     case 4:
//     {
//       (void) t;
//       if(component==0)
//         return std::sin(2*M_PI*p[0])*std::cos(2*M_PI*p[1]);
//       if(component==1)
//         return -std::cos(2*M_PI*p[0])*std::sin(2*M_PI*p[1]);
//       return 0.;
//     }
//     default:
//       Assert(false, ExcNotImplemented());
//   }
// }


#endif