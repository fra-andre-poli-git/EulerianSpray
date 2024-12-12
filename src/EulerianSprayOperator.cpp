#include"EulerianSprayOperator.h"
#include<deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>


// This function is entirely copied from tutorial 67. The only thing changed is dim +2
// which becomes dim+1 since here I don't have an energy equation
template <int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::project(const Function<dim> & function,
                                                    SolutionType &solution) const{
FEEvaluation<dim, degree, degree + 1, dim + 1, Number> phi(data, 0, 1);
MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 1, Number>
    inverse(phi);
solution.zero_out_ghost_values();
for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
    phi.reinit(cell);
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_dof_value(evaluate_function(function,
                                                phi.quadrature_point(q)), q);
    inverse.transform_from_q_points_to_basis(dim + 1,
                                                phi.begin_dof_values(),
                                                phi.begin_dof_values());
    phi.set_dof_values(solution);
    }                                                
}

//template class EulerOperator<1,1,1>;