#include"EulerianSprayOperator.h"
#include"InlinedFunctions.h"
#include<deal.II/matrix_free/fe_evaluation.h>
#include<deal.II/matrix_free/operators.h>
#include<deal.II/base/vectorization.h>


// This function is entirely copied from tutorial 67.
template <int dim, int degree, int n_points_1d>
  void EulerianSprayOperator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const DoFHandler<dim> &dof_handler){
    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
    const AffineConstraints<double>            dummy;
    const std::vector<const AffineConstraints<double> *> constraints = {&dummy};
    const std::vector<Quadrature<1>> quadratures = {QGauss<1>(n_q_points_1d),
                                                    QGauss<1>(fe_degree + 1)};

    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::none;

    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);
}



// This function is entirely copied from tutorial 67. The only thing changed is
// dim +2 which becomes dim+1 since here I don't have an energy equation
template <int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::project(
                                                const Function<dim> & function,
                                                SolutionType &solution) const{
    FEEvaluation<dim, degree, degree + 1, dim + 1, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 1, Number>
        inverse(phi);
    solution.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell){
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_dof_value(evaluate_function(function,
                                                    phi.quadrature_point(q)),
                                 q);
        inverse.transform_from_q_points_to_basis(dim + 1,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(solution);
    }                                                
}



template <int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::initialize_vector(
  SolutionType &vector) const
{
  data.initialize_dof_vector(vector);
}

template<int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::apply(
                                                      const double current_time,
                                                      const SolutionType & src,
                                                      SolutionType & dst) const{
  // In this block I apply the nonlinear operator proper, consisting in (...)
  {
      // This is for the output, I may use it later
      // TimerOutput::Scope t(timer, "apply - integrals");

      // This is for the boundary values, I will have to change it (TODO)
      // for (auto &i : inflow_boundaries)
      //   i.second->set_time(current_time);
      // for (auto &i : subsonic_outflow_boundaries)
      //   i.second->set_time(current_time);
      data.loop();


  }
  // In this block I apply the inverse matrix
  {

  }


}


template<int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::local_apply_cell(
        const MatrixFree<dim, Number> & data,
        SolutionType & dst,
        const SolutionType &src,
        const std::pair<unsigned int, unsigned int> & cell_range) const{
  // This is a class that provides all functions necessary to evaluate functions
  // at quadrature points and cell integrations. 
  FEEvaluation<dim, degree, n_points_1d, dim + 1, Number> phi(data);

  // I comment this passage since for the moment I do not have body force, but I
  // report it here below

    //   Tensor<1, dim, VectorizedArray<Number>> constant_body_force;
    // const Functions::ConstantFunction<dim> *constant_function =
    //   dynamic_cast<Functions::ConstantFunction<dim> *>(body_force.get());

    // if (constant_function)
    //   constant_body_force = evaluate_function<dim, Number, dim>(
    //     *constant_function, Point<dim, VectorizedArray<Number>>());


  // This is a loop over the cells
  for( unsigned int cell = cell_range.first; cell < cell_range.second; ++cell){
    phi.reinit(cell);
    phi.gather_evaluate(src, EvaluationFlags::values);

    // Loop over quadrature points
    for( unsigned int q = 0; q < phi.n_q_points; ++q){
      const auto w_q = phi.get_value(q);
      
    }

  }



}



// This is the implementation of the two helper function (defined here since
// they are only used by EulerianSprayOperator methods)
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


//Instantiation
template class EulerianSprayOperator<1, 2, 4>;
template class EulerianSprayOperator<2, 2, 4>;
template class EulerianSprayOperator<3, 2, 4>;