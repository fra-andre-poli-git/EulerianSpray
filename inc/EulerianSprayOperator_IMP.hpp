#include"EulerianSprayOperator.hpp"
#include"InlinedOperations.hpp"
#include"FindIntersection.hpp"

#include<deal.II/fe/fe_system.h> 
#include<deal.II/fe/mapping_q.h>
#include<deal.II/fe/fe_values_extractors.h> 
#include<deal.II/matrix_free/fe_evaluation.h>
#include<deal.II/matrix_free/operators.h>
#include<deal.II/base/vectorization.h>
#include <deal.II/base/quadrature_lib.h>

#include<algorithm>
#include<cmath>
#include <iomanip>
#include <limits>

template <int dim, int degree, int n_q_points_1d>
EulerianSprayOperator<dim, degree, n_q_points_1d>::EulerianSprayOperator(
  TimerOutput & timer, const Parameters & params):
  timer(timer), parameters(params){}

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
  void EulerianSprayOperator<dim, degree, n_q_points_1d>::reinit(const Mapping<dim> &   mapping, const DoFHandler<dim> &dof_handler)
{
  const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
  const AffineConstraints<double> dummy;
  const std::vector<const AffineConstraints<double> *> constraints = {&dummy};
  const std::vector<Quadrature<1>> quadratures = {QGauss<1>(n_q_points_1d),
                                                  QGauss<1>(degree + 1)};

  typename MatrixFree<dim, myReal>::AdditionalData additional_data;
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
    MatrixFree<dim, myReal>::AdditionalData::none;

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
  const myReal current_time,
  const SolutionType & src,
  SolutionType & dst) const
{
  (void) current_time;
  // In this block I apply the nonlinear operator proper
  {
    // This is for the output, I may use it later
    TimerOutput::Scope t(timer, "apply - integrals");
    data.loop(& EulerianSprayOperator::local_apply_cell,
      & EulerianSprayOperator::local_apply_face,
      & EulerianSprayOperator::local_apply_boundary_face,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, myReal>::DataAccessOnFaces::values,
      MatrixFree<dim, myReal>::DataAccessOnFaces::values);
  }
  // In this block I apply the inverse matrix
  {
    TimerOutput::Scope t(timer, "apply - inverse mass");
    
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
  const myReal current_time,
  const myReal factor_solution,
  const myReal factor_ai,
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
              MatrixFree<dim, myReal>::DataAccessOnFaces::values,
              MatrixFree<dim, myReal>::DataAccessOnFaces::values);
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
        const myReal ai = factor_ai;
        const myReal bi = factor_solution;
        if (ai == myReal())
        {
          /* DEAL_II_OPENMP_SIMD_PRAGMA */
          for (unsigned int i = start_range; i < end_range; ++i)
          {
            const myReal k_i          = next_ri.local_element(i);
            const myReal sol_i        = solution.local_element(i);
            solution.local_element(i) = sol_i + bi * k_i;
          }
        }
        else
        {
          /* DEAL_II_OPENMP_SIMD_PRAGMA */
          for (unsigned int i = start_range; i < end_range; ++i)
          {
            const myReal k_i          = next_ri.local_element(i);
            const myReal sol_i        = solution.local_element(i);
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
  FEEvaluation<dim, degree, 2*(degree + 1), dim + 1, myReal> phi(data, 0, 1);
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 1, myReal>
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
  FEEvaluation<dim, degree, n_q_points_1d, dim + 1, myReal> phi(data, 0, 0);

  for (unsigned int cell = 0; cell<data.n_cell_batches(); ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(solution, EvaluationFlags::values);
    VectorizedArray<myReal> local_errors_squared[2] = {};
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
    myReal             max_transport = 0;
    FEEvaluation<dim, degree, degree + 1, dim + 1, myReal> phi(data, 0, 1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<myReal> local_max = 0.;
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto solution = phi.get_value(q);
          const auto velocity = eulerian_spray_velocity<dim>(solution);
          const auto inverse_jacobian = phi.inverse_jacobian(q);
          const auto convective_speed = inverse_jacobian * velocity;
          VectorizedArray<myReal> convective_limit = 0.;
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

// This function applies the limiter given by Zhang, Shu,
// "On positivity-preserving high order discontinuous Galerkin schemes for 
// compressible Euler equations on rectangular meshes", adapted to the
// pressureless gas dynamics system of equations.
// This implementation is tied to Lagrange basis functions on quadrilaterals
template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::
bound_preserving_projection_1d(SolutionType & solution, const DoFHandler<dim> & dof_handler, const MappingQ1<dim> & mapping, const FESystem<dim> & fe) const
{
  // TODO: Refactor the cell average in a single dealii::Vector, since I merged
  // the two loops over the cells
  // I create the vector to store cell averages and I initialize it
  // std::vector< dealii::Vector<myReal>> cell_averages;
  // cell_averages.assign(dof_handler.get_triangulation().n_active_cells(),
  //   dealii::Vector<myReal>(dim+1));
  dealii::Vector<myReal> cell_average(dim+1);


  // This vector is to store the averages of the velocities
  // std::vector<myReal> cell_average_x_velocities(
  //   dof_handler.get_triangulation().n_active_cells());
  myReal cell_average_x_velocity;
  // I define the quadrature formula for the computation of the average
  // QGauss<dim>   quadrature_formula(
  //   static_cast<unsigned int>(std::ceil((fe.degree + 1)/2.)));
  QGauss<dim>   quadrature_formula(fe.degree+1);
  unsigned int n_q_points_average = quadrature_formula.size();
  FEValues<dim> fe_values (mapping,
    fe,
    quadrature_formula,
    update_values | update_JxW_values);
  std::vector<dealii::Vector<myReal>> local_solution_values(n_q_points_average,
    dealii::Vector<myReal>(dim+1));

  // Set the quadrature points for the projection
  // For the moment I use a quadrature formula only for 1D in disguise
  // since I first test the limiter in 1D cases. TODO: extend it to 2D and
  // if possible to 3D eventually.
  unsigned int M = (degree + 3) % 2 == 0 ? (degree + 3)/2 : (degree + 4)/2;
  // I use the brace initialization since the compiler complaints, using a
  // function definition
  Quadrature<dim> quadrature_x {QGaussLobatto<1>(M), QGauss<1>(1)};
  FEValues<dim> fe_values_x (mapping, fe, quadrature_x, update_values);
  unsigned int n_q_points = quadrature_x.size();

  std::vector<unsigned int> local_dof_indices (fe.dofs_per_cell);

  // Define the iterators for the loop
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  // TODO: this loop may not be very efficient since I do not access cell_averages
  // sequentially
  for (; cell!=endc; ++cell) // Loop over active cells
  {
//-------------------------Compute cell averages--------------------------------
    //unsigned int cell_no = cell->active_cell_index();
    fe_values.reinit(cell);
    fe_values.get_function_values(solution, local_solution_values);
    for(unsigned int d=0; d < dim + 1; ++d)
      cell_average[d] = 0.;
    cell_average_x_velocity = 0.;
    for(unsigned int q=0; q<n_q_points_average; ++q)
    {
      cell_average_x_velocity += local_solution_values[q][1] /
        local_solution_values[q][0];
      for(unsigned int d=0; d<dim+1; ++d)
        cell_average[d] += local_solution_values[q][d]*
          fe_values.JxW(q);
    }
    cell_average_x_velocity /= cell->measure();
    // Assert((cell_average_x_velocities[cell_no] <= max_velocity ) &&
    //       (cell_average_x_velocities[cell_no] >= min_velocity ),
    //       ExcMessage("Error: average velocity exceeds realizability bounds"
    //        + std::to_string(cell_average_x_velocities[cell_no])
    //        + " is indeed greater than " + std::to_string(max_velocity)
    //        + "or smaller than " + std::to_string(min_velocity)));
    for(unsigned int d=0; d<dim+1; ++d)
    {
      cell_average[d] /= cell->measure();
      if(d == 0)
        Assert(cell_average[d] >= 0.0,
          ExcMessage("Error: average density is negative: \bar{rho} = " + std::to_string(cell_average[d])));
      // if(d == 1)
      //   Assert((cell_averages[cell_no][d] <= cell_averages[cell_no][0] * max_velocity ) &&
      //     (cell_averages[cell_no][d] >= cell_averages[cell_no][0] * min_velocity ),
      //     ExcMessage("Error: average momentum exceeds realizability bounds"));
    }

//-------------------------Modify the solution----------------------------------
  //  unsigned int cell_no = cell->active_cell_index();
    cell->get_dof_indices(local_dof_indices);
    myReal cell_average_density = cell_average[0];
    // myReal cell_average_density = 1e-13;
    if(cell_average_density <= parameters.epsilon) // If the density is less than epsilon, set the solution as the average
    {
      // Set the mean value as solution
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)// Loop over DoFs
      {
        // Each DoF is associated to a different component of the system
        unsigned int comp_i = fe.system_to_component_index(i).first;
        // std::cout << "comp_i is equal to "<< comp_i<< " and i is " << i << std::endl;
        solution(local_dof_indices[i]) = cell_average[comp_i];
      }
      // std::exit(0);
    }
    else // If the average density is more than epsilon proceed to modify the solution
    {
      // I use the projection inroduced by Yang, Wei, Shu, in [49]. 
      // In [42] are showed different chices for the projection


      // Modify the density
      fe_values_x.reinit(cell);
      std::vector<myReal> density_values(n_q_points);
      const FEValuesExtractors::Scalar density(0);
      myReal rho_min = std::numeric_limits<myReal>::max();

      fe_values_x[density].get_function_values(solution, density_values);
      for(unsigned int q=0; q<n_q_points; ++q)
        rho_min = std::min(rho_min, density_values[q]);
      
      // I could define rho_min = std::min(rho_min, epsilon) as the min between
      // epsilon and itself, sparing the if (rho_min < epsilon), but I want to
      // avoid some situation where diff_den and diff_num should be exactly 
      // the same but in fact they are numerically different. Also, I could
      // check (TODO) which version runs faster: the one where I use the if or
      // the one where I don't use the if and I run this chunk for all the cells
      if (rho_min < parameters.epsilon)
      {
        myReal diff_den = cell_average_density - rho_min;
        // std::cout << "Diff den is "<< std::setprecision(std::numeric_limits<double>::max_digits10)<<diff_den<<std::endl;
        // myReal theta = 1.;
        // if(diff_den > 1e-12)
        // {
        //   myReal diff_num = cell_average_density - epsilon;
        //   //std::cout << "Diff num is " << std::setprecision(std::numeric_limits<double>::max_digits10)
        //       // <<diff_num<<std::endl;
        //   if(std::abs(diff_num - diff_den) < 1e-12)
        //     theta = diff_num / (diff_den );
        //   else
        //     theta = diff_num / diff_den;
        // }  

        myReal diff_num = cell_average_density - parameters.epsilon;
        myReal theta = diff_num / (diff_den + 1e-14);

        // std::cout << "Theta is "<< std::setprecision(std::numeric_limits<double>::max_digits10) << theta << std::endl;
        Assert(theta >= 0.0 && theta <= 1.0,
          ExcMessage("theta = "+ std::to_string(theta) +
          " must be between 0 and 1"));
        for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
        {
          // Each DoF is associated to a different component of the system
          unsigned int comp_i = fe.system_to_component_index(i).first;
          if(comp_i == 0)
          {
            // std::cout<< "Old density is "<< std::setprecision(std::numeric_limits<double>::max_digits10) << solution(local_dof_indices[i])<<std::endl;
            // std::cout<< "Average density is " << std::setprecision(std::numeric_limits<double>::max_digits10)<< cell_averages[cell_no][comp_i] << std::endl;
            solution(local_dof_indices[i]) = 
              (1.-theta) * cell_average[comp_i] +
              theta * (solution(local_dof_indices[i]));
            // std::cout<< "New density is "<< std::setprecision(std::numeric_limits<double>::max_digits10) << solution(local_dof_indices[i])<<std::endl;
            if(solution(local_dof_indices[i]) <= parameters.epsilon)
            {
              std::cerr<<"The new density is less than epsilon"<<std::endl;
              //std::cerr<<"Theta is "<<theta<<", average is "<<cell_averages[cell_no][comp_i] << ", new solution is " << solution(local_dof_indices[i])<<std::endl;
              //std::exit(1);
            }
          }  
        }
      }
      
      // Modify the velocity
      myReal theta_j = 2.0;
      myReal theta_i_j = 2.0;
      dealii::Tensor<1, dim  + 1, myReal> state_q, mean_w;
      dealii::Tensor<1, dim, myReal> state_velocity, mean_velocity;
      std::vector<dealii::Vector<myReal>> solution_values(n_q_points,
        Vector<myReal>(dim + 1));

      
      fe_values_x.get_function_values(solution, solution_values);
      for(unsigned int x_i=0; x_i < n_q_points; ++x_i)
      {
 
        for(size_t n = 0; n < 2; ++n)
        {
          state_q[n] = solution_values[x_i][n];
          mean_w[n] = cell_average[n]; 
        }
        double diff_den=(mean_w - state_q).norm();
        // if(diff_den< parameters.epsilon/2)
        //   // std::cout << "Min velocity: " << min_velocity << " Max velocity: " << max_velocity << " velocity "
        //   theta_i_j = 1.;
        // else
        // {
          // TODO: a lot of times it is called and then theta is = 1----> how to spare it?
          
          // state_velocity = eulerian_spray_velocity<dim>(state_q);
          // mean_velocity = eulerian_spray_velocity<dim>(mean_w); /// ATTENZIONE la velocità media non è la divisione tra quantità di moto media e densità media 
          
          // Here I find s, the intersection between q - \line{w} and
          // \partial G_\epsilon and then
          dealii::Tensor<1, dim  + 1, myReal> s = find_intersection_1d(state_q, mean_w,
            parameters.epsilon, min_velocity, max_velocity);
          // Compute theta^j = ||\line{w} - s||/||\line{w} - q||
          // std::cout << "The value in the dof is q =[ "<<state_q[0]<< " "<<state_q[1]<<" "<< state_q[2] << "]"<<std::endl;
          // std::cout << "The value in of the mean is w =[ "<<mean_w[0]<< " "<<mean_w[1]<<" "<< mean_w[2] << "]"<<std::endl;
          // std::cout << "The value in of the projection is s =[ "<<s[0]<< " "<<s[1]<<" "<< s[2] << "]"<<std::endl;

          // std::cout << "The numerator is " << (mean_w - s).norm();
          // std::cout << "The denominator is " <<  (mean_w - state_q).norm();
          theta_i_j = (mean_w - s).norm() / (diff_den);
          // std::cout << "Theta_i_j is " << theta_i_j << std::endl;
        // }
        

        // dealii::Tensor<1, dim  + 1, myReal> s = find_intersection_1d(state_q, mean_w,
        //     0.0, min_velocity, max_velocity);
        // theta_i_j = (mean_w - s).norm() / (diff_den + 1e-14);
        // Compare theta^i_j with theta_j and set theta_j = min(theta_j, theta^i_j)
        theta_j = std::min( theta_j, theta_i_j);
        Assert(theta_j >= 0.0 && theta_j <= 1.0,
          ExcMessage("theta_j = "+ std::to_string(theta_j) +
          " must be between 0 and 1"));
      }
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
      {
        // Each DoF is associated to a different component of the system
        unsigned int comp_i = fe.system_to_component_index(i).first;
        solution(local_dof_indices[i]) =
          (1.-theta_j) * cell_average[comp_i] +
          theta_j * (solution(local_dof_indices[i]));
      }
      // for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      // {
      //   unsigned int comp_i = fe.system_to_component_index(i).first;

      //   if (comp_i == 1)// If the dof is associated to x momentum (ρu)
      //   {
      //     // Define the local index of the node (0..dofs_per_node-1)
      //     const unsigned int base_index =
      //         fe.system_to_component_index(i).second;
      //     // Find the index of density associated to the same node
      //     const unsigned int system_index_rho =
      //         fe.component_to_system_index(0, base_index);

      //     const unsigned int dof_rho  = local_dof_indices[system_index_rho];
      //     const unsigned int dof_rhou = local_dof_indices[i];

      //     const double rho  = solution(dof_rho);
      //     const double rhou = solution(dof_rhou);
      //     double u = (rho != 0.0) ? rhou / rho : 0.0;

      //     if (u > max_velocity + 1e-13 || u < min_velocity - 1e-13)
      //     {
      //       std::cerr << "Problem in projected velocity: u = " <<std::setprecision(std::numeric_limits<double>::max_digits10)<< u;
      //       std::cerr << " theta_j is " << theta_j;

      //       //std::cerr << "The average velocity is " << cell_average_x_velocities[cell_no];
      //       std::cerr << " Max velocity is " << max_velocity
      //         << " and min velocity is " << min_velocity<<std::endl;
      //     }
      //   }
      // }

    }
  }
}

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::
straight_bound_preserving_projection_1d(SolutionType & solution, const DoFHandler<dim> & dof_handler, const MappingQ1<dim> & mapping, const FESystem<dim> & fe) const
{
  // TODO: Refactor the cell average in a single dealii::Vector, since I merged
  // the two loops over the cells
  // I create the vector to store cell averages and I initialize it
  // std::vector< dealii::Vector<myReal>> cell_averages;
  // cell_averages.assign(dof_handler.get_triangulation().n_active_cells(),
  //   dealii::Vector<myReal>(dim+1));
  dealii::Vector<myReal> cell_average(dim+1);


  // This vector is to store the averages of the velocities
  // std::vector<myReal> cell_average_x_velocities(
  //   dof_handler.get_triangulation().n_active_cells());
  myReal cell_average_x_velocity;
  // I define the quadrature formula for the computation of the average
  // QGauss<dim>   quadrature_formula(
  //   static_cast<unsigned int>(std::ceil((fe.degree + 1)/2.)));
  QGauss<dim>   quadrature_formula(fe.degree+1);
  unsigned int n_q_points_average = quadrature_formula.size();
  FEValues<dim> fe_values (mapping,
    fe,
    quadrature_formula,
    update_values | update_JxW_values);
  std::vector<dealii::Vector<myReal>> local_solution_values(n_q_points_average,
    dealii::Vector<myReal>(dim+1));

  // Set the quadrature points for the projection
  // For the moment I use a quadrature formula only for 1D in disguise
  // since I first test the limiter in 1D cases. TODO: extend it to 2D and
  // if possible to 3D eventually.
  unsigned int M = (degree + 3) % 2 == 0 ? (degree + 3)/2 : (degree + 4)/2;
  // I use the brace initialization since the compiler complaints, using a
  // function definition
  Quadrature<dim> quadrature_x {QGaussLobatto<1>(M), QGauss<1>(1)};
  FEValues<dim> fe_values_x (mapping, fe, quadrature_x, update_values);
  unsigned int n_q_points = quadrature_x.size();

  std::vector<unsigned int> local_dof_indices (fe.dofs_per_cell);

  // Define the iterators for the loop
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  // TODO: this loop may not be very efficient since I do not access cell_averages
  // sequentially
  for (; cell!=endc; ++cell) // Loop over active cells
  {
//-------------------------Compute cell averages--------------------------------
    //unsigned int cell_no = cell->active_cell_index();
    fe_values.reinit(cell);
    fe_values.get_function_values(solution, local_solution_values);
    for(unsigned int d=0; d < dim + 1; ++d)
      cell_average[d] = 0.;
    cell_average_x_velocity = 0.;
    for(unsigned int q=0; q<n_q_points_average; ++q)
    {
      cell_average_x_velocity += local_solution_values[q][1] /
        local_solution_values[q][0];
      for(unsigned int d=0; d<dim+1; ++d)
        cell_average[d] += local_solution_values[q][d]*
          fe_values.JxW(q);
    }
    cell_average_x_velocity /= cell->measure();
    // Assert((cell_average_x_velocities[cell_no] <= max_velocity ) &&
    //       (cell_average_x_velocities[cell_no] >= min_velocity ),
    //       ExcMessage("Error: average velocity exceeds realizability bounds"
    //        + std::to_string(cell_average_x_velocities[cell_no])
    //        + " is indeed greater than " + std::to_string(max_velocity)
    //        + "or smaller than " + std::to_string(min_velocity)));
    for(unsigned int d=0; d<dim+1; ++d)
    {
      cell_average[d] /= cell->measure();
      if(d == 0)
        Assert(cell_average[d] >= 0.0,
          ExcMessage("Error: average density is negative: \bar{rho} = " + std::to_string(cell_average[d])));
      // if(d == 1)
      //   Assert((cell_averages[cell_no][d] <= cell_averages[cell_no][0] * max_velocity ) &&
      //     (cell_averages[cell_no][d] >= cell_averages[cell_no][0] * min_velocity ),
      //     ExcMessage("Error: average momentum exceeds realizability bounds"));
    }

//-------------------------Modify the solution----------------------------------
  //  unsigned int cell_no = cell->active_cell_index();
    cell->get_dof_indices(local_dof_indices);
    myReal cell_average_density = cell_average[0];
    // myReal cell_average_density = 1e-13;
    if(cell_average_density <= parameters.epsilon) // If the density is less than epsilon, set the solution as the average
    {
      // Set the mean value as solution
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)// Loop over DoFs
      {
        // Each DoF is associated to a different component of the system
        unsigned int comp_i = fe.system_to_component_index(i).first;
        // std::cout << "comp_i is equal to "<< comp_i<< " and i is " << i << std::endl;
        solution(local_dof_indices[i]) = cell_average[comp_i];
      }
      // std::exit(0);
    }
    else // If the average density is more than epsilon proceed to modify the solution
    {
      // I use the projection inroduced by Yang, Wei, Shu, in [49]. 
      // In [42] are showed different chices for the projection
      myReal theta = 1.0;

      // Modify the density
      fe_values_x.reinit(cell);
      std::vector<myReal> density_values(n_q_points);
      const FEValuesExtractors::Scalar density(0);
      myReal rho_min = std::numeric_limits<myReal>::max();

      fe_values_x[density].get_function_values(solution, density_values);
      for(unsigned int q=0; q<n_q_points; ++q)
        rho_min = std::min(rho_min, density_values[q]);
      
      // I could define rho_min = std::min(rho_min, epsilon) as the min between
      // epsilon and itself, sparing the if (rho_min < epsilon), but I want to
      // avoid some situation where diff_den and diff_num should be exactly 
      // the same but in fact they are numerically different. Also, I could
      // check (TODO) which version runs faster: the one where I use the if or
      // the one where I don't use the if and I run this chunk for all the cells
      if (rho_min < parameters.epsilon)
      {
        myReal diff_den = cell_average_density - rho_min;
        // std::cout << "Diff den is "<< std::setprecision(std::numeric_limits<double>::max_digits10)<<diff_den<<std::endl;
        // myReal theta = 1.;
        // if(diff_den > 1e-12)
        // {
        //   myReal diff_num = cell_average_density - epsilon;
        //   //std::cout << "Diff num is " << std::setprecision(std::numeric_limits<double>::max_digits10)
        //       // <<diff_num<<std::endl;
        //   if(std::abs(diff_num - diff_den) < 1e-12)
        //     theta = diff_num / (diff_den );
        //   else
        //     theta = diff_num / diff_den;
        // }  

        myReal diff_num = cell_average_density - parameters.epsilon;
        theta = diff_num / (diff_den + 1e-14);

        // std::cout << "Theta is "<< std::setprecision(std::numeric_limits<double>::max_digits10) << theta << std::endl;
        Assert(theta >= 0.0 && theta <= 1.0,
          ExcMessage("theta = "+ std::to_string(theta) +
          " must be between 0 and 1"));
      }
      
      // Modify the velocity
      myReal theta_i_j = 2.0;
      dealii::Tensor<1, dim  + 1, myReal> state_q, mean_w;
      dealii::Tensor<1, dim, myReal> state_velocity, mean_velocity;
      std::vector<dealii::Vector<myReal>> solution_values(n_q_points,
        Vector<myReal>(dim + 1));

      
      fe_values_x.get_function_values(solution, solution_values);
      for(unsigned int x_i=0; x_i < n_q_points; ++x_i)
      {
 
        for(size_t n = 0; n < 2; ++n)
        {
          state_q[n] = solution_values[x_i][n];
          mean_w[n] = cell_average[n]; 
        }
        double diff_den=(mean_w - state_q).norm();
        // if(diff_den< parameters.epsilon/2)
        //   // std::cout << "Min velocity: " << min_velocity << " Max velocity: " << max_velocity << " velocity "
        //   theta_i_j = 1.;
        // else
        // {
          // TODO: a lot of times it is called and then theta is = 1----> how to spare it?
          
          // state_velocity = eulerian_spray_velocity<dim>(state_q);
          // mean_velocity = eulerian_spray_velocity<dim>(mean_w); /// ATTENZIONE la velocità media non è la divisione tra quantità di moto media e densità media 
          
          // Here I find s, the intersection between q - \line{w} and
          // \partial G_\epsilon and then
          dealii::Tensor<1, dim  + 1, myReal> s = find_intersection_1d_LLF_Variant(state_q, mean_w,
            parameters.epsilon, min_velocity, max_velocity);
          // Compute theta^j = ||\line{w} - s||/||\line{w} - q||
          // std::cout << "The value in the dof is q =[ "<<state_q[0]<< " "<<state_q[1]<<" "<< state_q[2] << "]"<<std::endl;
          // std::cout << "The value in of the mean is w =[ "<<mean_w[0]<< " "<<mean_w[1]<<" "<< mean_w[2] << "]"<<std::endl;
          // std::cout << "The value in of the projection is s =[ "<<s[0]<< " "<<s[1]<<" "<< s[2] << "]"<<std::endl;

          // std::cout << "The numerator is " << (mean_w - s).norm();
          // std::cout << "The denominator is " <<  (mean_w - state_q).norm();
          theta_i_j = (mean_w - s).norm() / (diff_den+1e-14);
          // std::cout << "Theta_i_j is " << theta_i_j << std::endl;
        // }
        

        // dealii::Tensor<1, dim  + 1, myReal> s = find_intersection_1d(state_q, mean_w,
        //     0.0, min_velocity, max_velocity);
        // theta_i_j = (mean_w - s).norm() / (diff_den + 1e-14);
        // Compare theta^i_j with theta_j and set theta_j = min(theta_j, theta^i_j)
        theta = std::min( theta, theta_i_j);
        Assert(theta >= 0.0 && theta <= 1.0,
          ExcMessage("theta = "+ std::to_string(theta) +
          " must be between 0 and 1"));
      }
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
      {
        // Each DoF is associated to a different component of the system
        unsigned int comp_i = fe.system_to_component_index(i).first;
        solution(local_dof_indices[i]) =
          (1.-theta) * cell_average[comp_i] +
          theta * (solution(local_dof_indices[i]));
      }
      // for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      // {
      //   unsigned int comp_i = fe.system_to_component_index(i).first;

      //   if (comp_i == 1)// If the dof is associated to x momentum (ρu)
      //   {
      //     // Define the local index of the node (0..dofs_per_node-1)
      //     const unsigned int base_index =
      //         fe.system_to_component_index(i).second;
      //     // Find the index of density associated to the same node
      //     const unsigned int system_index_rho =
      //         fe.component_to_system_index(0, base_index);

      //     const unsigned int dof_rho  = local_dof_indices[system_index_rho];
      //     const unsigned int dof_rhou = local_dof_indices[i];

      //     const double rho  = solution(dof_rho);
      //     const double rhou = solution(dof_rhou);
      //     double u = (rho != 0.0) ? rhou / rho : 0.0;

      //     if (u > max_velocity + 1e-13 || u < min_velocity - 1e-13)
      //     {
      //       std::cerr << "Problem in projected velocity: u = " <<std::setprecision(std::numeric_limits<double>::max_digits10)<< u;
      //       std::cerr << " theta_j is " << theta_j;

      //       //std::cerr << "The average velocity is " << cell_average_x_velocities[cell_no];
      //       std::cerr << " Max velocity is " << max_velocity
      //         << " and min velocity is " << min_velocity<<std::endl;
      //     }
      //   }
      // }

    }
  }
}


// This is the two dimension version of the previous one. For the moment I am
// supposing the physical dimension is 2, I may extend it to 3d later 
template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::
bound_preserving_projection(SolutionType & solution, const DoFHandler<dim> & dof_handler, const MappingQ1<dim> & mapping,const FESystem<dim> & fe) const
{

  // I comment the next lines since, having merged the two loops, I don't need
  // the vector of cell averages anymore
  // std::vector< dealii::Vector<myReal>> cell_averages;
  // cell_averages.assign(dof_handler.get_triangulation().n_active_cells(),
  //   dealii::Vector<myReal>(dim+1));
  
  // Instead, I initialize a single deali::Vector
  dealii::Vector<myReal> cell_average(dim+1);


  // QGauss<dim>   quadrature_formula(
  //   static_cast<unsigned int>(std::ceil((fe.degree + 1)/2.)));
  QGauss<dim>   quadrature_formula(fe.degree+1);
  unsigned int n_q_points_average = quadrature_formula.size();
  FEValues<dim> fe_values (mapping,
    fe,
    quadrature_formula,
    update_values | update_JxW_values);
  std::vector<dealii::Vector<myReal>> local_solution_values(n_q_points_average,
    dealii::Vector<myReal>(dim+1));
  // Define the iterators for the loop over active cells
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  // Set the quadrature points
  // Number of points needed for Gauss-Lobatto
  unsigned int M = /*degree + 3;*/(degree + 3) % 2 == 0 ? (degree + 3)/2 : (degree + 4)/2;
  // Number of points needed for Gauss
  unsigned int L = degree + 1;
  // I use the brace initialization since the compiler complaints, using a
  // function definition
  Quadrature<dim> quadrature_x {QGaussLobatto<1>(M), QGauss<1>(L)};
  Quadrature<dim> quadrature_y { QGauss<1>(L), QGaussLobatto<1>(M)};
  FEValues<dim> fe_values_x (mapping, fe, quadrature_x, update_values);
  FEValues<dim> fe_values_y (mapping, fe, quadrature_y, update_values);
  unsigned n_q_points = quadrature_x.size();
  // I define here some vectors that I need in the loop (TODO: check that are always assigned)
  std::vector<dealii::Vector<myReal>> solution_values(n_q_points,
    Vector<myReal>(dim + 1));
  std::vector<myReal> density_values(n_q_points);
  std::vector<unsigned int> local_dof_indices (fe.dofs_per_cell);

  for (; cell!=endc; ++cell)// Loop over active cells
  {
  //-------------------------Compute cell averages------------------------------
    //unsigned int cell_no = cell->active_cell_index();
    fe_values.reinit(cell);
    fe_values.get_function_values(solution, local_solution_values);

    for (unsigned d=0; d<dim+1; ++d) // Initialize cell_average
      cell_average[d] = 0.0;

    for(unsigned int q=0; q<n_q_points; ++q)
      for(unsigned int d=0; d<dim+1; ++d)
        cell_average[d] += local_solution_values[q][d]*
          fe_values.JxW(q);
    for(unsigned int d=0; d<dim+1; ++d)
    {
      cell_average[d] /= cell->measure(); 
      if(d == 0)
        Assert(cell_average[d] >= 0.0,
          ExcMessage("Error: average density is negative"));
      // TODO: modify this control for physical dimension 2 and 3
      // if(d == 1)
      //   Assert((cell_averages[cell_no][d] <= cell_averages[cell_no][0] * max_velocity ) &&
      //     (cell_averages[cell_no][d] >= cell_averages[cell_no][0] * min_velocity ),
      //     ExcMessage("Error: average velocity exceeds realizability bounds"));
    }

  //-------------------------Modify the solution--------------------------------
    // unsigned int cell_no = cell->active_cell_index();
    cell->get_dof_indices(local_dof_indices);
    myReal cell_average_density = cell_average[0];
    if(cell_average_density <= parameters.epsilon)
    {
      // Set the mean value as solution
      // Loop over DoFs
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
      {
        // Each DoF is associated to a different component of the system
        unsigned int comp_i = fe.system_to_component_index(i).first;
        solution(local_dof_indices[i]) = cell_average[comp_i];
      }
    }
    else
    {
      // I use the projection inroduced by Yang, Wei, Shu, in [49]. 
      // In [42] are showed different chices for the projection

      
      // Modify the density
      fe_values_x.reinit(cell);
      fe_values_y.reinit(cell);
      // I define it outside the loop
      // std::vector<myReal> density_values(n_q_points);
      const FEValuesExtractors::Scalar density(0);
      myReal rho_min = std::numeric_limits<myReal>::max();

      fe_values_x[density].get_function_values(solution, density_values);
      for(unsigned int q=0; q<n_q_points; ++q)
        rho_min = std::min(rho_min, density_values[q]);
      fe_values_y[density].get_function_values(solution, density_values);
      for(unsigned int q=0; q<n_q_points; ++q)
        rho_min = std::min(rho_min, density_values[q]);
      if(rho_min < parameters.epsilon)
      {
        myReal diff_num = cell_average_density - parameters.epsilon;/*std::abs(cell_average_density - parameters.epsilon);*/
        myReal diff_den = cell_average_density - rho_min; /*std::abs(cell_average_density - rho_min);*/
        myReal theta = 1.0;
        // if(diff_den < 1e-12)
        //   std::cout<<"Warning in density modification: you are dividing for a small myReal"<<std::endl;
        theta = diff_num / (diff_den+1e-14);
        Assert(theta >= 0.0 && theta <= 1.0,
          ExcMessage("theta = "+ std::to_string(theta) +
          " must be between 0 and 1"));
        for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
        {
          // Each DoF is associated to a different component of the system
          unsigned int comp_i = fe.system_to_component_index(i).first;
          if(comp_i == 0)
            solution(local_dof_indices[i]) = (1.-theta) * cell_average[comp_i]
              + theta * solution(local_dof_indices[i]);
              // cell_average[comp_i] +
              // theta * 
              // (solution(local_dof_indices[i]) - cell_average[comp_i]);
        }
      }
      // Modify the velocity
      myReal theta_j = 1.0;
      myReal theta_i_j = 2.0;   
      dealii::Tensor<1, dim  + 1, myReal> state_q, mean_w;
      dealii::Tensor<1, dim, myReal> state_velocity, mean_velocity;
      // I define it outside the loop
      // std::vector<dealii::Vector<myReal>> solution_values(n_q_points,
      //   Vector<myReal>(dim + 1));

      
      fe_values_x.get_function_values(solution, solution_values);
      for(unsigned int x_i=0; x_i < n_q_points; ++x_i)
      {

        for(size_t n = 0; n < dim + 1; ++n)
        {
          state_q[n] = solution_values[x_i][n];
          mean_w[n] = cell_average[n];
        }
        // state_velocity = eulerian_spray_velocity<dim>(state_q);
        // mean_velocity = eulerian_spray_velocity<dim>(mean_w);
        double diff_den=(mean_w - state_q).norm();
        // if(diff_den < parameters.epsilon/2)
        //   theta_i_j = 1.;
        // else
        // {
          // Here I find s, the intersection between q - \line{w} and
          // \partial G_\epsilon and then
          auto s = find_intersection( state_q, mean_w,
            parameters.epsilon, max_velocity);
        
          // Compute theta^j = ||\line{w} - s||/||\line{w} - q||
          theta_i_j = (mean_w - s).norm() / (diff_den+ 1e-14);

        // }

        // Compare theta^i_j with theta_j and set 
        // theta_j = min(theta_j, theta^i_j)  
        theta_j = std::min( theta_j, theta_i_j);

        Assert(theta_j >= 0.0 && theta_j <= 1.0,
          ExcMessage("theta_j = "+ std::to_string(theta_j) +
          " must be between 0 and 1"));
      }

      fe_values_y.get_function_values(solution, solution_values);
      for(unsigned int x_i=0; x_i < n_q_points; ++x_i)
      {

        for(size_t n = 0; n < dim + 1; ++n)
        {
          state_q[n] = solution_values[x_i][n];
          mean_w[n] = cell_average[n];
        }

        double diff_den=(mean_w - state_q).norm();
        // if(diff_den < parameters.epsilon/2)
        //   theta_i_j = 1.;
        // else
        // {

          // Here I find s, the intersection between q - \line{w} and
          // \partial G_\epsilon and then 
          auto s = find_intersection( state_q, mean_w,
            parameters.epsilon, max_velocity);
          // Compute theta^j = ||\line{w} - s||/||\line{w} - q||
          theta_i_j = (mean_w - s).norm() / (diff_den+1e-14);
        // }
        // Compare theta^i_j with theta_j and set 
        // theta_j = min(theta_j, theta^i_j)
        theta_j = std::min( theta_j, theta_i_j);
        Assert(theta_j >= 0.0 && theta_j <= 1.0,
          ExcMessage("theta_j = "+ std::to_string(theta_j) +
          " must be between 0 and 1"));
      }
      if(theta_j < 1.0)
      {
        for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
        {
          // Each DoF is associated to a different component of the system
          unsigned int comp_i = fe.system_to_component_index(i).first;
          solution(local_dof_indices[i]) = 
            (1.-theta_j) * cell_average[comp_i] +
            theta_j * (solution(local_dof_indices[i]));
            // cell_average[comp_i] +
            // theta_j * 
            // (solution(local_dof_indices[i]) - cell_average[comp_i]);
        }
      }
    }
  }  
}

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::
  compute_velocity_extrema_1d(const SolutionType & solution)
{
  // FEEvaluation<dim, degree, degree + 1, dim + 1, myReal> phi(data, 0, 1);


  // myReal max_velocity_u = std::numeric_limits<myReal>::lowest();
  // myReal min_velocity_u = std::numeric_limits<myReal>::max();

  // for(unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
  // {
  //   phi.reinit(cell);
  //   phi.gather_evaluate(solution, EvaluationFlags::values);
  //   dealii::VectorizedArray<myReal> local_max =
  //     - std::numeric_limits<myReal>::max();
  //   dealii::VectorizedArray<myReal> local_min =
  //     std::numeric_limits<myReal>::max();
  //   for(unsigned int q = 0; q <  phi.n_q_points; ++q)
  //   {
  //     const auto solution = phi.get_value(q);
  //     const auto velocity = eulerian_spray_velocity<dim>(solution);
  //     const auto inverse_jacobian = phi.inverse_jacobian(q);
  //     const auto convective_speed = inverse_jacobian * velocity;
  //     const auto u = convective_speed[0];

  //     local_max = std::max(local_max, u);
  //     local_min = std::min(local_min, u);
  //   }

  //   for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
  //   {
  //     max_velocity_u = std::max(max_velocity_u, local_max[v]);
  //     min_velocity_u = std::min(min_velocity_u, local_min[v]);
  //   }
  // }

  // max_velocity_u = Utilities::MPI::max(max_velocity_u, MPI_COMM_WORLD);
  // min_velocity_u = Utilities::MPI::min(min_velocity_u, MPI_COMM_WORLD);

  // max_velocity = max_velocity_u;
  // min_velocity = min_velocity_u;

  TimerOutput::Scope t(timer, "compute velocity x extrema");
  myReal max_velocity_x = std::numeric_limits<myReal>::lowest();
  myReal min_velocity_x = std::numeric_limits<myReal>::max();
  FEEvaluation<dim, degree, degree + 1, dim + 1, myReal> phi(data, 0, 1);

  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(solution, EvaluationFlags::values);
    VectorizedArray<myReal> local_max = std::numeric_limits<myReal>::lowest();
    VectorizedArray<myReal> local_min = std::numeric_limits<myReal>::max();

    for (unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto solution_q = phi.get_value(q);
      const auto velocity = eulerian_spray_velocity<dim>(solution_q);
                  
      // Extract x component of velocity
      const auto velocity_x = velocity[0];
                  
      local_max = std::max(local_max, velocity_x);
      local_min = std::min(local_min, velocity_x);
    }
            
    // Extract values from VectorizedArray
    for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
    {
      max_velocity_x = std::max(max_velocity_x, local_max[v]);
      min_velocity_x = std::min(min_velocity_x, local_min[v]);
    }
  }
      
  // MPI reduction
  max_velocity_x = Utilities::MPI::max(max_velocity_x, MPI_COMM_WORLD);
  min_velocity_x = Utilities::MPI::min(min_velocity_x, MPI_COMM_WORLD);

  max_velocity = max_velocity_x;
  min_velocity = min_velocity_x;
}

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::
  compute_velocity_max_norm(const SolutionType & solution)
{
  TimerOutput::Scope t(timer, "compute velocity norm extrema");
  myReal max_velocity_norm = std::numeric_limits<myReal>::lowest();
  FEEvaluation<dim, degree, degree + 1, dim + 1, myReal> phi(data, 0, 1);

  for(unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(solution, EvaluationFlags::values);
    VectorizedArray<myReal> local_max = std::numeric_limits<myReal>::lowest();

    for(unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto solution_q = phi.get_value(q);
      const auto velocity = eulerian_spray_velocity<dim>(solution_q);

      const auto velocity_norm = velocity * velocity;

      local_max = std::max(local_max, velocity_norm);
    }

    for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
      max_velocity_norm = std::max(max_velocity_norm, local_max[v]);
  }

  // MPI reduction
  max_velocity_norm = Utilities::MPI::max(max_velocity_norm, MPI_COMM_WORLD);

  max_velocity = std::sqrt(max_velocity_norm);
  min_velocity = 0.;  
}

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::local_apply_inverse_mass_matrix(const MatrixFree<dim, myReal> & data, SolutionType & dst, const SolutionType & src, const std::pair<unsigned int, unsigned int> & cell_range) const
{
  FEEvaluation<dim, degree, degree + 1, dim + 1, myReal>
    phi(data, 0, 1);
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 1, myReal>
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
        const MatrixFree<dim, myReal> & data,
        SolutionType & dst,
        const SolutionType &src,
        const std::pair<unsigned int, unsigned int> & cell_range) const{
  (void) dst;
  // This is a class that provides all functions necessary to evaluate functions
  // at quadrature points and cell integrations. 
  FEEvaluation<dim, degree, n_q_points_1d, dim + 1, myReal> phi(data);

  // I comment this passage since for the moment I do not have body force, but I
  // report it here since I may implement it.

    //   Tensor<1, dim, VectorizedArray<myReal>> constant_body_force;
    // const Functions::ConstantFunction<dim> *constant_function =
    //   dynamic_cast<Functions::ConstantFunction<dim> *>(body_force.get());

    // if (constant_function)
    //   constant_body_force = evaluate_function<dim, myReal, dim>(
    //     *constant_function, Point<dim, VectorizedArray<myReal>>());

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
// different numerical flux, defined in InlinedOperations.h
template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::local_apply_face(
  const MatrixFree<dim, myReal> & data,
  SolutionType & dst,
  const SolutionType & src,
  const std::pair<unsigned int, unsigned int> & face_range) const
{
  FEFaceEvaluation<dim, degree, n_q_points_1d, dim + 1, myReal> phi_m(data,
    true);
  FEFaceEvaluation<dim, degree, n_q_points_1d, dim + 1, myReal> phi_p(data,
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
          parameters.numerical_flux_type);
      phi_m.submit_value(-numerical_flux, q);
      phi_p.submit_value(numerical_flux, q);
    }
    phi_p.integrate_scatter(EvaluationFlags::values, dst);
    phi_m.integrate_scatter(EvaluationFlags::values, dst);
  }
}

template<int dim, int degree, int n_q_points_1d>
void EulerianSprayOperator<dim, degree, n_q_points_1d>::local_apply_boundary_face(
  const MatrixFree<dim, myReal> &,
  SolutionType & dst,
  const SolutionType & src,
  const std::pair<unsigned int, unsigned int> & face_range) const
{
  FEFaceEvaluation<dim, degree, n_q_points_1d, dim + 1, myReal> phi(data, true);

  for( unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    phi.reinit(face);
    phi.gather_evaluate(src, EvaluationFlags::values);
    
    for( unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto w_m = phi.get_value(q);
      const auto normal =  phi.normal_vector(q);

      Tensor<1, dim + 1, VectorizedArray<myReal>> w_p;
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
        parameters.numerical_flux_type);

      phi.submit_value(-flux, q);
    }
     phi.integrate_scatter(EvaluationFlags::values, dst);
  }
}


// This is the implementation of the two helper function (defined here since
// they are only used by EulerianSprayOperator methods)
template <int dim, typename myReal>
VectorizedArray<myReal>
evaluate_function(const Function<dim> &                      function,
                const Point<dim, VectorizedArray<myReal>> &p_vectorized,
                const unsigned int                         component){
    VectorizedArray<myReal> result;
    for (unsigned int v = 0; v < VectorizedArray<myReal>::size(); ++v){
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vectorized[d][v];
        result[v] = function.value(p, component);
    }
    return result;
}

template <int dim, typename myReal, int n_components = dim + 1>
Tensor<1, n_components, VectorizedArray<myReal>>
evaluate_function(const Function<dim> &                      function,
                const Point<dim, VectorizedArray<myReal>> &p_vectorized){
  AssertDimension(function.n_components, n_components);
  Tensor<1, n_components, VectorizedArray<myReal>> result;
  for (unsigned int v = 0; v < VectorizedArray<myReal>::size(); ++v){
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
      for (unsigned int d = 0; d < n_components; ++d)
          result[d][v] = function.value(p, d);
  }
  return result;
}
