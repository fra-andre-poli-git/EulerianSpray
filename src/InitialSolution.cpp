#include"InitialSolution.h"
#include"TypesDefinition.h"
#include<deal.II/base/point.h>

template<int dim>
double InitialSolution<dim>::value(const Point<dim> & p, const unsigned int component) const {
    switch(testcase){
        case 1:{
            if(component==0)
                return 0.5;
            if(component==1)
                return 0.5*(-0.5*(p[0]<-0.5) + 0.4*(-0.5<=p[0] && p[0]<0) + (0.4-p[0])*(0<=p[0] && p[0] <0.8) + -0.4*(p[0]>=0.8));
            // I put this return otherwise I get a warning. Maybe is the case to put an assertion,
            // since case 1 is meant to work in dimension 1 (therfore 2 components, one is the density and one the x momentum)
            return 0; 
        }
        default:{
            Assert(false, ExcNotImplemented());
            return 0.;
        }
    }

}

 template class InitialSolution<1>;
 template class InitialSolution<2>;
 template class InitialSolution<3>;