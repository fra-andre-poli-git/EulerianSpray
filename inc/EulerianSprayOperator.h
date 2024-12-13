#ifndef EULERIAN_SPRAY_OPERATOR_HH
#define EULERIAN_SPRAY_OPERATOR_HH

#include"TypesDefinition.h"

#include<deal.II/matrix_free/matrix_free.h>
#include<deal.II/base/function.h>
#include<deal.II/base/vectorization.h>

using namespace dealii;

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
evaluate_function(const Function<dim> &                      function,
                const Point<dim, VectorizedArray<Number>> &p_vectorized,
                const unsigned int                         component){
    VectorizedArray<Number> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v){
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vectorized[d][v];
        result[v] = function.value(p, component);
    }
    return result;
}


template <int dim, typename Number, int n_components = dim + 1>
Tensor<1, n_components, VectorizedArray<Number>>
evaluate_function(const Function<dim> &                      function,
                const Point<dim, VectorizedArray<Number>> &p_vectorized){
    AssertDimension(function.n_components, n_components);
    Tensor<1, n_components, VectorizedArray<Number>> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v){
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vectorized[d][v];
        for (unsigned int d = 0; d < n_components; ++d)
            result[d][v] = function.value(p, d);
    }
    return result;
}
#endif