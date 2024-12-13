#include"EulerianSprayOperator.h"
#include<deal.II/matrix_free/fe_evaluation.h>
#include<deal.II/matrix_free/operators.h>
#include<deal.II/base/vectorization.h>



// This function is entirely copied from tutorial 67. The only thing changed is dim +2
// which becomes dim+1 since here I don't have an energy equation
template <int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::project(const Function<dim> & function,
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

  template <int dim, int degree, int n_points_1d>
  void EulerianSprayOperator<dim, degree, n_points_1d>::initialize_vector(
    SolutionType &vector) const
  {
    data.initialize_dof_vector(vector);
  }




//Instantiation
template class EulerianSprayOperator<1, 2, 4>;
template class EulerianSprayOperator<2, 2, 4>;
template class EulerianSprayOperator<3, 2, 4>;