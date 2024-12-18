#ifndef EULERIAN_SPRAY_OPERATOR_HH
#define EULERIAN_SPRAY_OPERATOR_HH

#include"TypesDefinition.h"

#include<deal.II/matrix_free/matrix_free.h>
#include<deal.II/base/function.h>
#include<deal.II/base/vectorization.h>

using namespace dealii;


// This class implements the evaluators for Eulerian Spray problem
// in analogy to the 'EulerOperator' class in step-67 and 'LaplaceOperator'
// in step-37 or step-59. Since this operator is non-linear and does not require
//
template<int dim, int degree, int n_points_1d>
class EulerianSprayOperator{
    public:
        void reinit(const Mapping<dim> &   mapping,
                const DoFHandler<dim> &dof_handler);
        void project(const Function<dim> & function,
                     SolutionType &solution) const;
        void initialize_vector(SolutionType &vector) const;
    private:
        MatrixFree<dim, Number> data;
};


// Here I declare the helper function that I use for evaluation in EulerianSprayOperator methods
// This is a template function that is used for the evaluation
// Completely copied from tutorial 67
template <int dim, typename Number>
VectorizedArray<Number>
evaluate_function(const Function<dim> &, const Point<dim, VectorizedArray<Number>> &, const unsigned int){};

template <int dim, typename Number, int n_components = dim + 1>
Tensor<1, n_components, VectorizedArray<Number>>
evaluate_function(const Function<dim> &, const Point<dim, VectorizedArray<Number>> &){};


#endif