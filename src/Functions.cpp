#include"Functions.h"
#include"TypesDefinition.h"
#include<deal.II/base/point.h>
#include<cmath>

template<int dim>
double InitialSolution<dim>::value(const Point<dim> & p,
    const unsigned int component) const
{
  switch(testcase)
  {
    case 1:
    {
      if(component==0)
        return 0.5;
      if(component==1)
        return 0.5*(-0.5*(p[0]<-0.5) + 0.4*(-0.5<=p[0] && p[0]<0) + (0.4-p[0])*
          (0<=p[0] && p[0] <0.8) + -0.4*(p[0]>=0.8));
      // In testcase 1 the dimension is supposed to be 1. However, since output
      // functions are expected to work in dimension bigger than 1, I put this
      // return to make y momentum null.
      return 0.; 
    }
    case 2:
    {
      if(component==0)
        return (p[0]<0.)*1 + (p[0]>=0.)*0.5;
      if(component==1)
        return (p[0]<0.)*1*0.25 + (p[0]>=0.)*0.5*(-0.4);
      return 0.;
    }
    case 3:
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
  switch(testcase)
  {
    case 1:
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
    case 2:
    {
      return 0.;
    }
    case 3:
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
  switch(testcase)
  {
    case 1:
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

template<int dim>
double ExternalFlux<dim>::value(const Point<dim> & p,
  const unsigned int component) const
{
  const double t = this->get_time();

  switch(testcase)
  {
    case 2:
    {
      if(component==0)
        return
      if(component==1)
      if(component==2)
    }
    default:
      Assert(false, ExcNotImplemented());
  }
}


template<int dim>
double ExternalFlux<dim>::value(const Point<dim> & p,
  const unsigned int component) const
{
  const double t = this->get_time();

  switch(testcase)
  {
    case 4:
    {
      (void) t;
      if(component==0)
        return std::sin(2*M_PI*p[0])*std::cos(2*M_PI*p[1]);
      if(component==1)
        return -std::cos(2*M_PI*p[0])*std::sin(2*M_PI*p[1]);
      return 0.;
    }
    default:
      Assert(false, ExcNotImplemented());
  }
}

template class InitialSolution<2>;

template class FinalSolution<2>;

template class DirichletFunction<2>;

template class ExternalFlux<2>;