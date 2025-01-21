#include"Functions.h"
#include"TypesDefinition.h"
#include<deal.II/base/point.h>

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
      return 0.;
    }
    default:
  {
    Assert(false, ExcNotImplemented());
    return 0.;
  }
  }
}

 template class InitialSolution<1>;
 template class InitialSolution<2>;
 template class InitialSolution<3>;