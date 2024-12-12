#ifndef EULERIAN_SPRAY_OPERATOR_HH
#define EULERIAN_SPRAY_OPERATOR_HH

#include"TypesDefinition.h"

#include<deal.II/matrix_free/matrix_free.h>
#include<deal.II/base/function.h>

using namespace dealii;

template<int dim, int degree, int n_points_1d>
class EulerianSprayOperator{
    public:
        void project(const Function<dim> & function,
                     SolutionType &solution) const;
    private:
        MatrixFree<dim, Number> data;
};


#endif