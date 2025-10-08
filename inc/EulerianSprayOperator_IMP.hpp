#include"EulerianSprayOperator.hpp"
#include"InlinedFunctions.hpp"
#include"FindIntersection.hpp"

#include<deal.II/fe/fe_system.h> 
#include<deal.II/fe/mapping_q.h>
#include<deal.II/fe/fe_values_extractors.h> 
#include<deal.II/matrix_free/fe_evaluation.h>
#include<deal.II/matrix_free/operators.h>
#include<deal.II/base/vectorization.h>

#include<algorithm>
#include<cmath>

template <int dim, int degree, int n_q_points_1d>
EulerianSprayOperator<dim, degree, n_q_points_1d>::EulerianSprayOperator(
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
// based on a template parameter `n_q_points_1d` (that will be assigned by the
// `n_q_points_1d` value specified in the declaration of EulerianSprayProblem).
// More accurate integration is necessary to avoid the aliasing problem due to
// the variable coefficients in the Euler operator. The second less accurate
// quadrature formula is a tight one based on `fe_degree+1` and needed for
// the inverse mass matrix. While that formula provides an exact inverse
// only on affine element shapes and not on deformed elements, it enables
// the fast inversion of the mass matrix by tensor product techniques,
// necessary to ensure optimal computational efficiency overall.
template <int dim, int degree, int n_q_points_1d>
  void EulerianSprayOperator<dim, degree, n_q_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const DoFHandler<dim> &dof_handler)
{
  const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
  const AffineConstraints<double> dummy;
  const std::vector<const AffineConstraints<double> *> constraints = {&dummy};
  const std::vector<Quadrature<1>> quadratures = {QGauss<1>(n_q_points_1d),
                                                  QGauss<1>(degree + 1)};

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

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::set_neumann_boundary(
  const types::boundary_id boundary_id)
{
  // TODO: it would be nice to set an assert to ensure I didn't set other
  // boundary condition, like the one in step 67
  neumann_boundaries.insert(boundary_id);
}

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::set_dirichlet_boundary(
  const types::boundary_id boundary_id,
  std::unique_ptr<Function<dim>> dirichlet_function)
{
  // TODO: it would be nice to set an assert to ensure I didn't set other
  // boundary condition, like the one in step 67
  dirichlet_boundaries[boundary_id] = std::move(dirichlet_function);
}


template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::apply(
  const Number current_time,
  const SolutionType & src,
  SolutionType & dst) const
{
  (void) current_time;
  // In this block I apply the nonlinear operator proper
  {
    // This is for the output, I may use it later
    // TimerOutput::Scope t(timer, "apply - integrals");
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

// This function performs one stage of low storage Runge-Kutta integration. It 
// is very similar EulerianSprayOperator::apply() with an update of the vectors
// ki and ri used in RK
template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::perform_lsrk_stage(
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
      [&](const unsigned int start_range, const unsigned int end_range)
      {
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
      }
    );
  }
}

// This function projects a function to the solution vector.
template <int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::project(
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



template<int dim, int degree, int n_q_points_1d>
std::array<double, 2>
EulerianSprayOperator<dim, degree, n_q_points_1d>::compute_errors(
  const Function<dim> & function,
  const SolutionType & solution) const{
  TimerOutput::Scope t(timer, "compute errors");
  double errors_squared[2] = {};
  FEEvaluation<dim, degree, n_q_points_1d, dim + 1, Number> phi(data, 0, 0);

  for (unsigned int cell = 0; cell<data.n_cell_batches(); ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(solution, EvaluationFlags::values);
    VectorizedArray<Number> local_errors_squared[2] = {};
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto error = evaluate_function(function, phi.quadrature_point(q)) -
        phi.get_value(q);
      const auto JxW = phi.JxW(q);
      local_errors_squared[0] += error[0] * error[0] * JxW;
      for (unsigned int d = 0; d < dim; ++d)
        local_errors_squared[1] += (error[d + 1] * error[d + 1]) * JxW;
    }
  for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
    for (unsigned int d = 0; d < 2; ++d)
      errors_squared[d] += local_errors_squared[d][v];
  }

  Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);

  std::array<double, 2> errors;
  for ( unsigned int d = 0; d < 2; ++d)
    errors[d] = std::sqrt(errors_squared[d]);

  return errors;
}


// This function is used to compute the cell transport speed, needed to set a 
// time step in according to CFL condition.
// Conceptually, we want to assure that the transportation of iformation is not
// so big that that information enters and leaves the cell within one time step.
// For information transported along with
// the medium, $\mathbf u$ is scaled by the mesh size,
// so an estimate of the maximal velocity can be obtained by computing
// $\|J^{-\mathrm T} \mathbf{u}\|_\infty$, where $J$ is the Jacobian of the
// transformation from real to the reference domain. Note that
// FEEvaluationBase::inverse_jacobian() returns the inverse and transpose
// Jacobian, representing the metric term from real to reference
// coordinates, so we do not need to transpose it again.
template<int dim, int degree, int n_q_points_1d>
double
EulerianSprayOperator<dim, degree, n_q_points_1d>::compute_cell_transport_speed(
  const SolutionType & solution) const
{
  TimerOutput::Scope t(timer, "compute transport speed");
    Number             max_transport = 0;
    FEEvaluation<dim, degree, degree + 1, dim + 1, Number> phi(data, 0, 1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<Number> local_max = 0.;
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto solution = phi.get_value(q);
            const auto velocity = eulerian_spray_velocity<dim>(solution);
            const auto inverse_jacobian = phi.inverse_jacobian(q);
            const auto convective_speed = inverse_jacobian * velocity;
            VectorizedArray<Number> convective_limit = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              convective_limit =
                std::max(convective_limit, std::abs(convective_speed[d]));
            local_max =
              std::max(local_max, convective_limit);
          }

        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
          for (unsigned int d = 0; d < 3; ++d)
            max_transport = std::max(max_transport, local_max[v]);
      }
    max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);
    return max_transport;
}


template <int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::initialize_vector(
  SolutionType &vector) const
{
  data.initialize_dof_vector(vector);
}

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::set_numerical_flux(
  const NumericalFlux & flux)
{
  numerical_flux_type = flux;
}

// This function applies the limiter given by Zhang, Shu,
// "On positivity-preserving high order discontinuous Galerkin schemes for 
// compressible Euler equations on rectangular meshes", adapted to the
// pressureless gas dynamics system of equations.
// This implementation is tied to Lagrange basis functions on quadrilaterals
template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::apply_positivity_limiter(
  SolutionType & solution,
  const DoFHandler<dim> & dof_handler,
  const MappingQ1<dim> & mapping,
  const FESystem<dim> & fe) const
{

  //-------------------------Compute cell averages------------------------------
  std::vector< dealii::Vector<Number>> cell_averages;
  QGauss<dim>   quadrature_formula(
    static_cast<unsigned int>std::ceil(fe.degree + 1)/.2);
  unsigned int n_q_points = quadrature_formula.size();
  FEValues<dim> fe_values (mapping,
    fe,
    quadrature_formula,
    update_values | update_JxW_values);
  std::vector<dealii::Vector<Number>> local_solution_values(n_q_points,
    dealii::Vector<Number>(dim+1));
  // Loop over active cells
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for(; cell!=endc; ++cell)
  {
    // Compute cell average
    unsigned int cell_no = cell->active_cell_index();
    fe_values.reinit(cell);
    fe_values.get_function_values(solution, local_solution_values);
    cell_averages.push_back(dealii::Vector<Number>(dim+1));
    for(unsigned int q=0; q<n_q_points; ++q)
      for(unsigned int d=0; d<dim+1; ++d)
        cell_averages[cell_no][d] += local_solution_values[q][d]*
          fe_values.JxW(q);
    for(unsigned int d=0; d<dim+1; ++d)
      cell_averages[cell_no][d] /= cell->measure(); 
  }

  //-------------------------Modify the solution--------------------------------
  // TODO: this implementation is only for 1D cases, extend it to general dim
  // Set up a small number
  epsilon = 1e-13;
  // Set the quadrature points
  // For the moment I use a quadrature formula only for 1D in disguise
  // since I first test the limiter in 1D cases. TODO: extend it to 2D and
  // if possible to 3D eventually.
  unsigned int N = (degree + 3) % 2 == 0 ? (degree + 3)/2 : (degree + 4)/2;
  QGauss<dim> quadrature_x (QGaussLobatto<1>(N), QGauss<1>(degree +1));
  FEValues<dim> fe_values_x (mapping, fe, quadrature_x, update_values);
  n_q_points = quadrature_x.size();
  std::vector<Number> density_values(n_q_points);
  // Loop over cells
  cell = dof_handler.begin_active();
  std::vector<unsigned int> local_dof_indices (fe.dofs_per_cell);
  for (; cell!=endc; ++cell)
  {
    unsigned int cell_no = cell->active_cell_index();
    cell->get_dof_indices(local_dof_indices);
    Number cell_average_density = cell_averages[cell_no][0];
    if(cell_average_density < epsilon)
    {
      // Set the mean value as solution
      // Loop over DoFs
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
      {
        // Each DoF is associated to a different component of the system
        unsigned int comp_i = fe.system_to_component_index(i).first;
        solution(local_dof_indices[i]) = cell_averages[cell_no][comp_i];
      }
    }
    else
    {
      // I use the projection inroduced by Yang, Wei, SHu, in [49]. 
      // In [42] are showed different chices for the projection
      fe_values_x.reinit(cell);
      // Modify the density
      Number rho_min = std::numeric_limits<Number>::max();
      fe_values_x[density].get_function_values(solution, density_values);
      for(unsigned int q=0; q<n_q_points; ++q)
        rho_min = std::min(rho_min, density_values[q]);
      Number theta = (cell_average_density - rho_min)/
        (cell_average_density - rho_min);
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
      {
        // Each DoF is associated to a different component of the system
        unsigned int comp_i = fe.system_to_component_index(i).first;
        solution(local_dof_indices[i]) = cell_averages[cell_no][comp_i] +
          theta * 
          (solution(local_dof_indices[i]) - cell_averages[cell_no][comp_i]);
      }
      double theta_j = 1.0;
      // Modify the velocity
      for(/*every quadreture point in the cell*/){
        // TODO: here I should modify this structure to work in 2D and 3D
        // Find s, the intersection between q - \line{w} and \partial G_\epsilon
        dealii::Tensor<1, dim> s;
        // Compute theta^i_j = ||s - \line{w}||/||q - \line{w}||
        // Compare theta^i_j with theta_j and set theta_j = min(theta_j, theta^i_j)
      }
      
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
      {
        // Each DoF is associated to a different component of the system
        unsigned int comp_i = fe.system_to_component_index(i).first;
        if(comp_i > 0) // velocity components
          solution(local_dof_indices[i]) = cell_averages[cell_no][comp_i] +
            theta_j * 
            (solution(local_dof_indices[i]) - cell_averages[cell_no][comp_i]);
      }      
    }
  }
}

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::
  local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> & data,
    SolutionType & dst,
    const SolutionType & src,
    const std::pair<unsigned int, unsigned int> & cell_range) const
{
  FEEvaluation<dim, degree, degree + 1, dim + 1, Number>
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

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::local_apply_cell(
        const MatrixFree<dim, Number> & data,
        SolutionType & dst,
        const SolutionType &src,
        const std::pair<unsigned int, unsigned int> & cell_range) const{
  (void) dst;
  // This is a class that provides all functions necessary to evaluate functions
  // at quadrature points and cell integrations. 
  FEEvaluation<dim, degree, n_q_points_1d, dim + 1, Number> phi(data);

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
template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::local_apply_face(
  const MatrixFree<dim, Number> & data,
  SolutionType & dst,
  const SolutionType & src,
  const std::pair<unsigned int, unsigned int> & face_range) const
{
  FEFaceEvaluation<dim, degree, n_q_points_1d, dim + 1, Number> phi_m(data,
    true);
  FEFaceEvaluation<dim, degree, n_q_points_1d, dim + 1, Number> phi_p(data,
    false);
  
  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    phi_p.reinit(face);
    phi_p.gather_evaluate(src, EvaluationFlags::values);

    phi_m.reinit(face);
    phi_m.gather_evaluate(src, EvaluationFlags::values);

    for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
    {
      const auto numerical_flux =
        eulerian_spray_numerical_flux<dim>(phi_m.get_value(q),
          phi_p.get_value(q),
          phi_m.get_normal_vector(q),
          numerical_flux_type);
      phi_m.submit_value(-numerical_flux, q);
      phi_p.submit_value(numerical_flux, q);
    }
    phi_p.integrate_scatter(EvaluationFlags::values, dst);
    phi_m.integrate_scatter(EvaluationFlags::values, dst);
  }
}

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::local_apply_boundary_face(
  const MatrixFree<dim, Number> &,
  SolutionType & dst,
  const SolutionType & src,
  const std::pair<unsigned int, unsigned int> & face_range) const
{
  FEFaceEvaluation<dim, degree, n_q_points_1d, dim + 1, Number> phi(data, true);

  for( unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    phi.reinit(face);
    phi.gather_evaluate(src, EvaluationFlags::values);
    
    for( unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto w_m = phi.get_value(q);
      const auto normal =  phi.normal_vector(q);

      Tensor<1, dim + 1, VectorizedArray<Number>> w_p;
      const auto boundary_id = data.get_boundary_id(face);
      if(neumann_boundaries.find(boundary_id) != neumann_boundaries.end())
      {
        for(unsigned int d = 0; d < dim+1; ++d)
          w_p[d] = w_m[d];
      }
      else if(dirichlet_boundaries.find(boundary_id) != 
        dirichlet_boundaries.end())
      {
        w_p = evaluate_function(*dirichlet_boundaries.find(boundary_id)->second,
          phi.quadrature_point(q));
      }
      else
        AssertThrow(false,
          ExcMessage("Unknown boundary id, did you set a boundary condition"
            " for this part of the domain boundary?"));

      auto flux = eulerian_spray_numerical_flux<dim>(w_m,
        w_p,
        normal,
        numerical_flux_type);

      phi.submit_value(-flux, q);
    }
     phi.integrate_scatter(EvaluationFlags::values, dst);
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
