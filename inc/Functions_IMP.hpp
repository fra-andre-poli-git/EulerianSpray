// I define this to use M_PI in order to get pi value with a pre-C++20 standard
#define _USE_MATH_DEFINES

#include"TypesDefinition.hpp"

#include<deal.II/base/point.h>
#include<cmath>
// #include<numbers> // This version works with C++20

template<int dim>
double InitialSolution<dim>::value(const Point<dim> & p, const unsigned int component) const
{
  switch(parameters.testcase)
  {
    case 1:
    {
      //double sigma = 0.1;
      if(component == 0)
        //return 1./2 * exp(-pow((p[0]-1./2),2)/pow(sigma,2));
        return sin(p[0]) + 2;
      if(component == 1)
        //return -1./2 * exp(-pow((p[0]-1./2),2)/pow(sigma,2));
        return sin(p[0]) + 2;
      return 0.;
    }
    case 2:
    {
      if(component==0)
        return 0.5;
      if(component==1)
        return 0.5 
            * ( -0.5 * (p[0]<-0.5)
            +  0.4 * (-0.5<=p[0] && p[0]<0)
            + (0.4-p[0]) * (0<=p[0] && p[0] <0.8)
            + -0.4 * (p[0]>=0.8));
      return 0.; 
    }
    case 3:
    {
      if(component==0)
        return 1 * (p[0]<0.)
          + 0.25 * (p[0]>=0.);
      if(component==1)
        return 1 * (p[0]<0.)
          +  0.25 * 0  * (p[0]>=0.);
      return 0.;
    }
    case 4:
    {
      if(component == 0)
        return 0.5;
      if(component == 1)
        return 0.5 * (-0.5) * (p[0]<0.)
          + 0.5 * 0.4 * (p[0]>0.);
      return 0.; 
    }
    case 5:
    {
      if(component == 0)
        return 0.9*(((-0.3<p[0] && p[0] <-0.2) && (-0.15<p[1] && p[1] <0.05)) ||
          ((0.2<p[0] && p[0]<0.3) && (-0.05<p[1]) && (p[1]<0.15)))
          + 0.1;
      if(component==1)
        return 0.5 * ((-0.3<p[0] && p[0] <-0.2) && (-0.15<p[1] && p[1] <0.05)) 
          - 0.5 * ((0.2<p[0] && p[0]<0.3) && (-0.05<p[1]) && (p[1]<0.15));
      return 0.; 
    }
    case 7:
    {
      if(component == 0) // \rho
        return 1./100.;
      else
      {
        // Define theta
        double theta = std::acos(p[0]/std::sqrt(p[0]*p[0] + p[1]*p[1]));
        if (p[1] < 0.)
          theta = 2. * M_PI - theta;
          // theta = 2. * std::numbers::pi - theta; // this version works with C++20
        if(component == 1) // \rho u
          return 1./100. * (-1./10.) * std::cos(theta);
        if (component == 2) // \rho v
          return 1./100. * (-1./10.) * std::sin(theta);
        return 0.; // to avoid implicit fallthrough warning
      }
    }
    case 8:
    {
      if(component == 0)
        return 1./10.;
      if(component == 1)
        return 1./10. * 0.25 * (1. - 2. * (p[0]>0));
      if(component == 2)
        return 1./10. * 0.25 * (1. - 2. * (p[1]>0));
      return 0.; // to avoid implicit fallthrough warning
    }
    case 9:
    {
      if(component == 0)
        return 1./100.;
      else
      {
        double r = std::sqrt(p[0]*p[0] + p[1]*p[1]);
        double theta = std::acos(p[0]/r);
        if (p[1] < 0.)
          theta = 2. * M_PI - theta;
        if(r < 0.3)
        {
          if(component == 1)
            return 1./100. * std::cos(theta);
          if(component == 2)
            return 1./100. * std::sin(theta);
        }
        else
        {
          if(component == 1)
            return - 1./100. * 1./2. * std::cos(theta);
          if(component == 2)
            return - 1./100. * 1./2. * std::sin(theta);
        }
      }
      return 0.; // to avoid implicit fallthrough warning
    }
    case 10:
    {
   
    }
    default:
      Assert(false, ExcNotImplemented());
  }
}

template<int dim>
double FinalSolution<dim>::value(const Point<dim> & p, const unsigned int component) const
{
  switch(parameters.testcase)
  {
    case 2:
    {
      if(component == 0)
        return 0.5 * (p[0]< -0.75) 
          + 0 * (p[0]>= -0.75) * (p[0]< -0.3) 
          + 0.5 * (p[0]>= -0.3 ) * (p[0]<0.2)
          + 1 * (p[0]>=0.2) * (p[0]<0.6)
          + 0.5 * (p[0]>= 0.6 );
      if(component == 1)
        return 0.5*(-0.5)*(p[0]< -0.75)
          + 0 * (p[0]>= -0.75)*(p[0]< -0.3) 
          + 0.5 * 0.4 *(p[0]>= -0.3 )*(p[0]<0.2)
          + 1 * (0.4 - p[0]) / 0.5 * (p[0]>=0.2) * (p[0]<0.6)
          + 0.5 * (-0.4) * (p[0]>= 0.6 );
      return 0.;
    }
    default:
      Assert(false, ExcNotImplemented());
  }
}

template<int dim>
double FinalSolutionVelocity<dim>::value(const Point<dim> & p, const unsigned int component) const
{
  switch(parameters.testcase)
  {
    case 2:
    {
      if(component == 0)
        return 0.5 * (p[0]< -0.75) 
          + 0 * (p[0]>= -0.75) * (p[0]< -0.3) 
          + 0.5 * (p[0]>= -0.3) * (p[0]<0.2) 
          + 1 * (p[0]>=0.2) * (p[0]<0.6)
          + 0.5 * (p[0]>= 0.6 );
      if(component == 1)
        return -0.5 * (p[0]<-0.5)
          + 0.4*(p[0]>=-0.5)*(p[0]<0.2)
          + (0.8-2*p[0]) * (p[0]>=0.2)*(p[0]<0.6)
          - 0.4 * (p[0]>= 0.6);
      return 0.;
    }
    default:
      Assert(false, ExcNotImplemented());
  }
}

template<int dim>
double DirichletFunction<dim>::value(const Point<dim> & p, const unsigned int component) const
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
  }
}

template<int dim>
double ExactSolution<dim>::value(const Point<dim> & p, const unsigned int component) const
{
  const double t = this->get_time();

  switch(parameters.testcase)
  {
    case 1:
    {
      // Solve the implicit equation x_0 + t u_0(x_0) = x
      // where u_0(x) = sin(x) + 2
      double x_0;

      if(component == 0)
        return (sin(x_0) + 2)/(1 + cos(x_0));
      if(component == 1)
        return (sin(x_0) + 2) * (sin(x_0) + 2)/(1 + cos(x_0));
      if(component == 2)
        return 0.;
    }
    case 2:
    {
      if (component == 0) // rho(x,t)
        return 0.5 * (p[0] < -0.5 - 0.5*t)
          + 0.0 * (p[0] >= -0.5 - 0.5*t) * (p[0] < -0.5 + 0.4*t)
          + 0.5 * (p[0] >= -0.5 + 0.4*t) * (p[0] < 0.4*t)
          + (0.5 / (1.0 - t)) * (p[0] >= 0.4*t) * (p[0] < 0.8 - 0.4*t)
          + 0.5 * (p[0] >= 0.8 - 0.4*t);
      if (component == 1) // rho(x,t) * u(x,t)
        return 0.5 *( -0.5) * (p[0] < -0.5 - 0.5*t)
          + 0.0 * (p[0] >= -0.5 - 0.5*t) * (p[0] < -0.5 + 0.4*t)
          + 0.5 * 0.4 * (p[0] >= -0.5 + 0.4*t) * (p[0] < 0.4*t)
          + (0.5 / (1.0 - t)) * ((0.4 - p[0]) / (1.0 - t)) * (p[0] >= 0.4*t) * (p[0] < 0.8 - 0.4*t)
          - 0.5 * 0.4 * (p[0] >= 0.8 - 0.4*t);
      return 0.;

    }
    default:
      Assert(false, ExcNotImplemented());
  }
}


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

