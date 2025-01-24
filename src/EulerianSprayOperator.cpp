#include"EulerianSprayOperator.h"
#include"InlinedFunctions.h"
#include<deal.II/matrix_free/fe_evaluation.h>
#include<deal.II/matrix_free/operators.h>
#include<deal.II/base/vectorization.h>

template <int dim, int degree, int n_points_1d>
EulerianSprayOperator<dim, degree, n_points_1d>::EulerianSprayOperator(
  TimerOutput & timer): timer(timer){}

// For the initialization of the Euler operator, we set up the MatrixFree
// variable contained in the class. This can be done given a mapping to
// describe possible curved boundaries as well as a DoFHandler object
// describing the degrees of freedom. Since we use a discontinuous Galerkin
// discretization in this tutorial program where no constraints are imposed
// strongly on the solution field, we do not need to pass in an
// AffineConstraints object and rather use a dummy for the
// construction. With respect to quadrature, we want to select two different
// ways of computing the underlying integrals: The first is a flexible one,
// based on a template parameter `n_points_1d` (that will be assigned the
// `n_q_points_1d` value specified at the top of this file). More accurate
// integration is necessary to avoid the aliasing problem due to the
// variable coefficients in the Euler operator. The second less accurate
// quadrature formula is a tight one based on `fe_degree+1` and needed for
// the inverse mass matrix. While that formula provides an exact inverse
// only on affine element shapes and not on deformed elements, it enables
// the fast inversion of the mass matrix by tensor product techniques,
// necessary to ensure optimal computational efficiency overall.
template <int dim, int degree, int n_points_1d>
  void EulerianSprayOperator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const DoFHandler<dim> &dof_handler)
{
  const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
  const AffineConstraints<double> dummy;
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

template<int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::apply(
  const Number current_time,
  const SolutionType & src,
  SolutionType & dst) const
{
  (void) current_time;
  // In this block I apply the nonlinear operator proper
  {
      // This is for the output, I may use it later
      // TimerOutput::Scope t(timer, "apply - integrals");

      // This is for the boundary values, I will have to change it (TODO)
      // for (auto &i : inflow_boundaries)
      //   i.second->set_time(current_time);
      // for (auto &i : subsonic_outflow_boundaries)
      //   i.second->set_time(current_time);
      data.loop(& EulerianSprayOperator::local_apply_cell,
        & EulerianSprayOperator::local_apply_face,
        & EulerianSprayOperator::local_apply_boundary_face,
        this,
        dst,
        src,
        true,
        MatrixFree<dim, Number>::DataAccessOnFaces::values,
        MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }
  // In this block I apply the inverse matrix
  {
    // TimerOutput::Scope t(timer, "apply - inverse mass");
    
    data.cell_loop(& EulerianSprayOperator::local_apply_inverse_mass_matrix,
                    this,
                    dst,
                    dst);
  }
}

// This function performs one stage of Runge-Kutta integration. It is very
// similar EulerianSprayOperator::apply() with an update of 
template<int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::perform_stage(
  const Number current_time,
  const Number factor_solution,
  const Number factor_ai,
  const SolutionType & current_ri,
  SolutionType & vec_ki,
  SolutionType & solution,
  SolutionType & next_ri) const
{
  (void) current_time;
  {
    TimerOutput::Scope t(timer, "rk_stage - integrals L_h");

    // for (auto &i : inflow_boundaries)
    //   i.second->set_time(current_time);
    // for (auto &i : subsonic_outflow_boundaries)
    //   i.second->set_time(current_time);

    data.loop(&EulerianSprayOperator::local_apply_cell,
              &EulerianSprayOperator::local_apply_face,
              &EulerianSprayOperator::local_apply_boundary_face,
              this,
              vec_ki,
              current_ri,
              true,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  {
    TimerOutput::Scope t(timer, "rk_stage - inv mass + vec upd");
    // This is the sixth version of data.cell_loop. I highlight that the first
    // of the two functions in the argument (which  corresponds to 
    // "operation_before_loop" is a void function.
    // As a complete newbie I must note that [&] captures ALL the variables in
    // in the scope by reference, therefore I can use "solution" in the body
    // of the lamba function (look up on the slides)
    data.cell_loop(
      &EulerianSprayOperator::local_apply_inverse_mass_matrix,
      this,
      next_ri,
      vec_ki,
      std::function<void(const unsigned int, const unsigned int)>(),
      [&](const unsigned int start_range, const unsigned int end_range) {
        const Number ai = factor_ai;
        const Number bi = factor_solution;
        if (ai == Number())
          {
            /* DEAL_II_OPENMP_SIMD_PRAGMA */
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = next_ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
              }
          }
        else
          {
            /* DEAL_II_OPENMP_SIMD_PRAGMA */
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = next_ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
                next_ri.local_element(i)  = sol_i + ai * k_i;
              }
          }
      });
  }
}

// This function projects a function to the solution vector.
template <int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::project(
  const Function<dim> & function,
  SolutionType &solution) const
{
  FEEvaluation<dim, degree, degree + 1, dim + 1, Number> phi(data, 0, 1);
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 1, Number>
    inverse(phi);
  solution.zero_out_ghost_values();
  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
  {
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

template<int dim, int degree, int n_points_1d>
std::array<double, 3>
EulerianSprayOperator<dim, degree, n_points_1d>::compute_errors(
  const Function<dim> & function,
  const SolutionType & solution) const{
  TimerOutput::Scope t(timer, "compute errors");
  double errors_squared[3] = {};
  FEEvaluation<dim, degree, n_points_1d, dim + 1, Number> phi(data, 0, 0);

  for (unsigned int cell = 0; cell<data.n_cell_batches(); ++cell){
    phi.reinit(cell);
    phi.gather_evaluate(solution, EvaluationFlags::values);
    VectorizedArray<Number> local_errors_squared[3] = {};
    for (unsigned int q = 0; q < phi.n_q_points; ++q){
      const auto error = evaluate_function(function, phi.quadrature_point(q)) -
        phi.get_value(q);
      const auto JxW = phi.JxW(q);
      local_errors_squared[0] += error[0] * error[0] * JxW;
      for (unsigned int d = 0; d < dim; ++d)
        local_errors_squared[1] += (error[d + 1] * error[d + 1]) * JxW;

      local_errors_squared[2] += (error[dim+1] * error[dim + 1]) * JxW;
    }
  for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
    for (unsigned int d = 0; d < 3; ++d)
      errors_squared[d] += local_errors_squared[d][v];
  }

  Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);

  std::array<double, 3> errors;
  for ( unsigned int d = 0; d < 3; ++d)
    errors[d] = std::sqrt(errors_squared[d]);

  return errors;
}


// I will implement the following function, otherwise TODO erase it

// template<int dim, int degree, int n_points_1d>
// double
// EulerianSprayOperator<dim, degree, n_points_1d>::compute_cell_transport_speed(
//   const SolutionType & solution) const;{
// }


template <int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::initialize_vector(
  SolutionType &vector) const
{
  data.initialize_dof_vector(vector);
}

template<int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> & data,
    SolutionType & dst,
    const SolutionType & src,
    const std::pair<unsigned int, unsigned int> & cell_range) const
{
  FEEvaluation<dim, degree, n_points_1d, dim + 1, Number>
    phi(data, 0, 1);
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 1, Number>
    inverse(phi);
  
  for( unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    phi.reinit(cell);
    phi.read_dof_values(src);

    inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

    phi.set_dof_values(dst);
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
  // report it here since I may implement it.

    //   Tensor<1, dim, VectorizedArray<Number>> constant_body_force;
    // const Functions::ConstantFunction<dim> *constant_function =
    //   dynamic_cast<Functions::ConstantFunction<dim> *>(body_force.get());

    // if (constant_function)
    //   constant_body_force = evaluate_function<dim, Number, dim>(
    //     *constant_function, Point<dim, VectorizedArray<Number>>());

  // This is a loop over the cells
  for( unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(src, EvaluationFlags::values);

    // Loop over quadrature points
    for( unsigned int q = 0; q < phi.n_q_points; ++q){
      const auto w_q = phi.get_value(q);
      phi.submit_gradient(eulerian_spray_flux<dim>(w_q), q);
      // as before, I comment this if and I don't write its body
      // if (body_force.get() != nullptr)
    }

    phi.integrate_scatter((/*(body_force.get() != nullptr) ?
                                 EvaluationFlags::values :*/
                                 EvaluationFlags::nothing) |
                                 EvaluationFlags::gradients, dst);
  }
}

// This function performs the integration over the element's faces. The only
// modification w.r.t. the tutorial 67 is the fact that here I am using a
// different numerical flux, defined in InlinedFunctions.h
template<int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::local_apply_face(
  const MatrixFree<dim, Number> & data,
  SolutionType & dst,
  const SolutionType & src,
  const std::pair<unsigned int, unsigned int> & face_range) const
{
  FEFaceEvaluation<dim, degree, n_points_1d, dim + 1, Number> phi_m(data, true);
  FEFaceEvaluation<dim, degree, n_points_1d, dim + 1, Number> phi_p(data,
    false);
  
  for(unsigned int face = face_range.first; face < face_range.second; ++face){
    phi_p.reinit(face);
    phi_p.gather_evaluate(src, EvaluationFlags::values);

    phi_m.reinit(face);
    phi_m.gather_evaluate(src, EvaluationFlags::values);

    for(unsigned int q = 0; q < phi_m.n_q_points; ++q){
      const auto numerical_flux =
        eulerian_spray_numerical_flux<dim>(phi_m.get_value(q),
          phi_p.get_value(q),
          phi_m.normal_vector(q));
      phi_m.submit_value(-numerical_flux, q);
      phi_p.submit_value(numerical_flux, q);
    }
    phi_p.integrate_scatter(EvaluationFlags::values, dst);
    phi_m.integrate_scatter(EvaluationFlags::values, dst);
  }
}

template<int dim, int degree, int n_points_1d>
void EulerianSprayOperator<dim, degree, n_points_1d>::local_apply_boundary_face(
  const MatrixFree<dim, Number> &                   data,
  SolutionType &      dst,
  const SolutionType & src,
  const std::pair<unsigned int, unsigned int> &     face_range) const{
  
  FEFaceEvaluation<dim, degree, n_points_1d, dim + 1, Number> phi(data, true);

  for( unsigned int face = face_range.first; face < face_range.second; ++face){
    phi.reinit(face);
    phi.gather_evaluate(src, EvaluationFlags::values);
    
    for( unsigned int q = 0; q < phi.n_q_points; ++q){
      const auto w_m = phi.get_value(q);
      const auto normal = phi.normal_vector(q);
      // TODO: understand my boundary conditions
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


//Instantiations of the template
template class EulerianSprayOperator<1, 2, 4>;
template class EulerianSprayOperator<2, 2, 4>;
template class EulerianSprayOperator<3, 2, 4>;