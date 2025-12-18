#include"TypesDefinition.hpp"
#include"RungeKuttaIntegrator.hpp"
#include"Functions.hpp"
#include"InlinedOperations.hpp"
#include"Parameters.hpp"


#include<deal.II/grid/grid_generator.h>
#include<deal.II/grid/tria_description.h>
#include<deal.II/grid/grid_tools.h>
#include<deal.II/fe/fe_dgq.h>
#include<deal.II/base/utilities.h>
#include<deal.II/numerics/vector_tools.h>
#include<deal.II/numerics/data_out.h>


#include<memory>
#include<iostream>
#include<chrono>


template<int dim, int degree>
EulerianSprayProblem<dim, degree>::EulerianSprayProblem(
  const Parameters & params):
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
#ifdef DEAL_II_WITH_P4EST
    triangulation(MPI_COMM_WORLD),
#endif
    parameters(params),    
    fe(FE_DGQ<dim>(degree),dim+1), // Lagrange basis functions
    mapping(),
    dof_handler(triangulation),
    time(0),
    time_step(0),
    timer(pcout, TimerOutput::never, TimerOutput::wall_times),
    eulerian_spray_operator(timer, params){}

template<int dim, int degree>
void EulerianSprayProblem<dim, degree>::make_grid_and_dofs()
{
  switch(parameters.testcase)
  {
    
    case 1:// Accuracy 1d (Example 3 from [49])
    {
      Assert(false, ExcNotImplemented());
      GridGenerator::subdivided_hyper_rectangle(triangulation,
        {parameters.n_el_x_direction, parameters.n_el_x_direction /10},
        Point<dim>(0,0),
        Point<dim>(1,0.5),
        true);

      // TODO: put an if if I am using parallel::distributed::triangulation    
      
      // These three lines make a periodicity constraint on top and bottom
      // boundaries. Periodicity constraint works marking the periodic faces
      // as they where interior faces neighboring to the ones at the opposite
      // boundary, and as so will be treated in the apply functions of 
      // EulerianSprayOperator
      std::vector<GridTools::PeriodicFacePair<
        typename Triangulation<dim>::cell_iterator>> periodicity_vector;
      GridTools::collect_periodic_faces(triangulation,
        2,
        3,
        1,
        periodicity_vector);
      triangulation.add_periodicity(periodicity_vector);
      
      periodicity_vector.clear();

      GridTools::collect_periodic_faces(triangulation,
        0,
        1,
        0,
        periodicity_vector);
      triangulation.add_periodicity(periodicity_vector);


      eulerian_spray_operator.set_1d_in_disguise();

      final_time = 0.1;
      break;
    }
    case 2:// Delta shock 1d (Example 4 from [49])
    {
      GridGenerator::subdivided_hyper_rectangle(triangulation,
        {parameters.n_el_x_direction,parameters.n_el_x_direction/10},
        Point<dim>(-0.5,0),
        Point<dim>(0.5,0.1),
        true);
#ifdef DEAL_II_WITH_P4EST
      std::vector<GridTools::PeriodicFacePair<
        typename parallel::distributed::Triangulation<dim>::cell_iterator>> periodicity_vector;
#else
      std::vector<GridTools::PeriodicFacePair<
      typename Triangulation<dim>::cell_iterator>> periodicity_vector;
#endif
      GridTools::collect_periodic_faces(triangulation,
        2,
        3,
        1,
        periodicity_vector);
      triangulation.add_periodicity(periodicity_vector);

      eulerian_spray_operator.set_neumann_boundary(0);
      eulerian_spray_operator.set_neumann_boundary(1);

      eulerian_spray_operator.set_1d_in_disguise();

      
      // final_time = parameters.final_time;
      final_time = 0.5;
      break;
    }  
    case 3:// Vacuum 1D (Example 5 from [49])
    {
      // Note the fact that last argument is true, therefore we get different
      // boundary_id for the four boundaries (process called "colorization")
      //
      // I link the number of elements in y direction to the ones in x, assuring
      // that the cells are squared, since I may have problems in determining
      // the time step (even though I am rewiewing the function that computes
      // the speed)
      GridGenerator::subdivided_hyper_rectangle(triangulation,
        {parameters.n_el_x_direction,parameters.n_el_x_direction/20},
        Point<dim>(-1,0),
        Point<dim>(1,0.1),
        true);
      
      // These three lines make a periodicity constraint on top and bottom
      // boundaries. Periodicity constraint works marking the periodic faces
      // as they where interior faces neighboring to the ones at the opposite
      // boundary, and as so will be treated in the apply functions of 
      // EulerianSprayOperator
#ifdef DEAL_II_WITH_P4EST
      std::vector<GridTools::PeriodicFacePair<
        typename parallel::distributed::Triangulation<dim>::cell_iterator>> periodicity_vector;
#else
      std::vector<GridTools::PeriodicFacePair<
      typename Triangulation<dim>::cell_iterator>> periodicity_vector;
#endif

      GridTools::collect_periodic_faces(triangulation,
        2,
        3,
        1,
        periodicity_vector);
      triangulation.add_periodicity(periodicity_vector);

      eulerian_spray_operator.set_neumann_boundary(0);
      eulerian_spray_operator.set_neumann_boundary(1);
      eulerian_spray_operator.set_1d_in_disguise();

      // final_time = parameters.final_time;
      final_time = 0.5;
      break;
    }
    case 4:// Vacuum close up 1d (Example 6 of [49])
    {
      Assert(false, ExcNotImplemented());
    }
    case 5:// Accuracy 2d (Example 7 of [49])
    {
      Assert(false, ExcNotImplemented());
    }
    case 6: // Radial delta in the origin 2d (Example 8 of [49])
    {
      GridGenerator::subdivided_hyper_cube(triangulation,
        parameters.n_el_x_direction,
        -0.5,
        0.5,
        true);

      eulerian_spray_operator.set_neumann_boundary(0);
      eulerian_spray_operator.set_neumann_boundary(1);
      eulerian_spray_operator.set_neumann_boundary(2);
      eulerian_spray_operator.set_neumann_boundary(3);
      final_time = parameters.final_time;
      break;
    }
    case 7:// Delta crossing in the origin 2d (Example 9 of [49])
    {
      GridGenerator::subdivided_hyper_cube(triangulation,
        parameters.n_el_x_direction,
        -0.5,
        0.5,
        true);
      eulerian_spray_operator.set_neumann_boundary(0);
      eulerian_spray_operator.set_neumann_boundary(1);
      eulerian_spray_operator.set_neumann_boundary(2);
      eulerian_spray_operator.set_neumann_boundary(3);
      final_time = parameters.final_time;
      break;
    }
    case 8: // Anular delta 2d (Example 10 of [49])
    {
      GridGenerator::subdivided_hyper_cube(triangulation,
        parameters.n_el_x_direction,
        -0.5,
        0.5,
        true);
      eulerian_spray_operator.set_neumann_boundary(0);
      eulerian_spray_operator.set_neumann_boundary(1);
      eulerian_spray_operator.set_neumann_boundary(2);
      eulerian_spray_operator.set_neumann_boundary(3);
      final_time = parameters.final_time;
      break;
    } 
    case 9:// Vacuum 2d (Example 11 of [49])
    {
      Assert(false, ExcNotImplemented());
    }  
    case 10:// Taylor-Green vortices
    {
      Assert(false, ExcNotImplemented());
      GridGenerator::hyper_cube(triangulation, 0, 1, true);
    }
    /*
    case ???:// Test diffusivity 2d
    {
      GridGenerator::subdivided_hyper_cube(triangulation,
        parameters.n_el_x_direction,
        -0.5,
        0.5,
        true);
      eulerian_spray_operator.set_neumann_boundary(0);
      eulerian_spray_operator.set_neumann_boundary(1);
      eulerian_spray_operator.set_neumann_boundary(2);
      eulerian_spray_operator.set_neumann_boundary(3);
      final_time = parameters.final_time;
      break;
    }
    */
  }

  dof_handler.distribute_dofs(fe);

  eulerian_spray_operator.reinit(mapping, dof_handler);
  eulerian_spray_operator.initialize_vector(solution);

  std::cout<< "Number of degrees of freedom "<<dof_handler.n_dofs()
            << " ( = " << (dim + 1) << " [vars] x "
            << triangulation.n_global_active_cells() << " [cells] x "
            << Utilities::pow(degree + 1, dim) << " [dofs/cell/var] )"
            << std::endl;
}

// This is the function that writes the solution in a .vtk file
template<int dim, int degree>
void EulerianSprayProblem<dim, degree>::output_results(const unsigned int result_myReal, bool final_time)
{
  // In testcase 3 I have the exact solution at final time
  if(parameters.testcase==3 && final_time)
  {
    const std::array<double, 2> errors =
      eulerian_spray_operator.compute_errors(FinalSolution<dim>(parameters),
        solution);
    const std::string quantity_name = "error";

    pcout << "Time:" << std::setw(8) << std::setprecision(3) << time
      << ", " << quantity_name << " rho: " << std::setprecision(4)
      << std::setw(10) << errors[0] << ", rho * u: " << std::setprecision(4)
      << std::setw(10) << errors[1] << std::endl;
  }

  TimerOutput::Scope t(timer, "output");

  Postprocessor postprocessor;
  DataOut<dim> data_out;

  DataOutBase::VtkFlags flags;

  // Flag determining whether to write patches as linear cells or as a
  // high-order Lagrange cell.
  if(degree>1)
    flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  data_out.attach_dof_handler(dof_handler);
  std::vector<std::string> names;
  names.emplace_back("density");
  for(unsigned int d = 0; d<dim; ++d)
    names.emplace_back("momentum");
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation;
  interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);
  for(unsigned int d = 0; d < dim; ++d)
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

  data_out.add_data_vector(dof_handler, solution, names, interpretation);


  // Here I insert the exact solution
  if(parameters.testcase==3 && final_time)
  {
    SolutionType ExactFinalSolution;
    ExactFinalSolution.reinit(solution);
    eulerian_spray_operator.project(FinalSolutionVelocity<dim>(parameters),
     ExactFinalSolution);
    // VectorTools::interpolate(dof_handler,
    //   FinalSolutionVelocity<dim>(parameters),
    //   ExactFinalSolution);

    std::vector<std::string> names;
    names.emplace_back("Exact_density");
    for(unsigned int d = 0; d<dim; ++d)
      names.emplace_back("Exact_velocity");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation;
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    for(unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(dof_handler, ExactFinalSolution, names,
      interpretation);
  }
  data_out.add_data_vector(solution, postprocessor);

  Vector<double> mpi_owner(triangulation.n_active_cells());
  mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  data_out.add_data_vector(mpi_owner, "owner");

  data_out.build_patches(mapping,
    fe.degree,
    DataOut<dim>::curved_inner_cells);
  
  if(final_time)
  {
    const std::string filename = 
      "./results/final_solution.vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
  }
  else
  {
    const std::string filename = 
      "./results/solution_" +
        Utilities::int_to_string(result_myReal, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
  }
}

template<int dim, int degree>
void EulerianSprayProblem<dim, degree>::run()
{
  {
    const unsigned int n_vect_myReal = VectorizedArray<myReal>::size();
    const unsigned int n_vect_bits   = 8 * sizeof(myReal) * n_vect_myReal;

    pcout << "Running with "
          << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
          << " MPI processes" << std::endl;
    pcout << "Vectorization over " << n_vect_myReal << ' '
          << (std::is_same<myReal, double>::value ? "doubles" : "floats")
          << " = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    }

  make_grid_and_dofs();

  std::unique_ptr<RungeKuttaIntegrator<SolutionType, 
    EulerianSprayOperator<dim,degree,n_q_points_1d>, dim>>
      integrator;
  if(parameters.scheme == forward_euler || parameters.scheme==ssp_stage_2_order_2 ||
    parameters.scheme==ssp_stage_3_order_3)
    integrator =
      std::make_unique<SSPRungeKuttaIntegrator
        <SolutionType,
          EulerianSprayOperator<dim,degree,n_q_points_1d>, dim>>(parameters.scheme);
  else
    integrator =
      std::make_unique<LSRungeKuttaIntegrator
        <SolutionType,
          EulerianSprayOperator<dim,degree,n_q_points_1d>, dim>>(parameters.scheme); 

  SolutionType rk_register1;
  SolutionType rk_register2;
  rk_register1.reinit(solution);
  rk_register2.reinit(solution);

  // Here I initialize the solution with the initial data and I compute its
  // extrema to be used in the limiting phase
  eulerian_spray_operator.project(InitialSolution<dim>(parameters), solution);
  

  if(eulerian_spray_operator.get_1d_in_disguise())
  {
    eulerian_spray_operator.compute_velocity_extrema_1d(solution);
    std::cout << "I computed intial velocity extrema, that happen to be "
      << eulerian_spray_operator.get_max_velocity() << " and "
      << eulerian_spray_operator.get_min_velocity()<<std::endl;
  }
  else
  {  
    eulerian_spray_operator.compute_velocity_max_norm(solution);
    std::cout << "I computed initial velocity max norm, that happen to be "
      << eulerian_spray_operator.get_max_velocity()<<std::endl;
  }

  // if(parameters.limiter_type == bound_preserving && degree > 0)
  // {
  //   if(eulerian_spray_operator.get_1d_in_disguise())
  //     eulerian_spray_operator.bound_preserving_projection_1d(solution, dof_handler,
  //       mapping, fe);
  //   else
  //     eulerian_spray_operator.bound_preserving_projection(solution, dof_handler,
  //       mapping, fe);
  // }
  // This small chunk aims at finding h, the smallest distance between two
  // vertices
  double min_cell_length = std::numeric_limits<double>::max();
  // If the test is a 1d test in disguise, I will take as h the size of the cell
  // in x direction even though for now I have made them squared
  if(eulerian_spray_operator.get_1d_in_disguise())
  {
    for(const auto & cell : triangulation.active_cell_iterators())
    {
      // const Point<dim> &v0 = cell->vertex(0);
      // const Point<dim> &v1 = cell->vertex(1);
      // const double hx = std::abs(v1(0) - v0(0)); 
      min_cell_length =
        std::min(min_cell_length, /*hx*/cell->extent_in_direction(0));
    }
  }
  else // Otherwise I will use the diagonal
  {
    for(const auto & cell : triangulation.active_cell_iterators())
      min_cell_length =
        std::min(min_cell_length, /*cell->diameter()*/ cell->extent_in_direction(0));
  }
  // with MPI here I have to make the minimum over all processors
  min_cell_length =
    Utilities::MPI::min(min_cell_length, MPI_COMM_WORLD);
  
  
  // In this block I set the time step to comply with CFL condition
  
  // time_step = CFL*min_cell_length /
  //   std::max(std::abs(eulerian_spray_operator.get_max_velocity()),
  //   std::abs(eulerian_spray_operator.get_min_velocity()));
  double CFL = parameters.CFL;
  if(eulerian_spray_operator.get_1d_in_disguise())
    time_step = CFL * min_cell_length * min_cell_length;
  else
    time_step = CFL * std::pow(min_cell_length, 3./2.);

  // If I use Godunov flux I can check if the time step satisfies the CFL condition
  // provided by [49]
  if(parameters.numerical_flux_type == godunov &&
    eulerian_spray_operator.get_1d_in_disguise())
  {
  unsigned int M = (degree + 3) % 2 == 0 ? (degree + 3)/2 : (degree + 4)/2;
  QGaussLobatto<1> qgl (M);
  const auto &weights = qgl.get_weights(); 
  double CFL = weights[0];
    Assert(time_step < CFL*min_cell_length /
      std::max(std::abs(eulerian_spray_operator.get_max_velocity()),
      std::abs(eulerian_spray_operator.get_min_velocity())),
      ExcMessage("This time step doesn't comply with its CFL condition") );
  }

  //double CFL = 1./2.;
  // // Now I set the time step to be exactly the biggest to satisfy CFL condition
  //  time_step = CFL/
  //   Utilities::truncate_to_n_digits(
  //     eulerian_spray_operator.compute_cell_transport_speed(solution), 3);

  pcout << "Time step size: " << time_step
    << ", minimal h: " << min_cell_length
    << std::endl
    << std::endl;

  output_results(0, false);
  // This is the time loop
  time = 0;
  unsigned int timestep_myReal = 0;
  while(time < final_time - 1e-12)
  {
    ++timestep_myReal;
    auto t0 = std::chrono::high_resolution_clock::now();
    // Here the integration in time is performed by the integrator
    integrator->perform_time_step(eulerian_spray_operator,
      time,
      time_step,
      solution,
      rk_register1,
      rk_register2,
      dof_handler,
      mapping,
      fe);
    auto t1 = std::chrono::high_resolution_clock::now();

    pcout<<"Performed time step at time: "<<time<<
      ", time step number: "<< timestep_myReal<<
      "\n"<< "Computation of time step took "<<
      std::chrono::duration<double>(t1-t0).count() << " s\n"<<std::endl;

    if(parameters.plot_everything)
      output_results(timestep_myReal, false);
    else
    {
      if(static_cast<int>(time / parameters.snapshot_instant) !=
        static_cast<int>((time - time_step) / parameters.snapshot_instant) ||
        time >= final_time - 1e-12)
        output_results(
          static_cast<unsigned int>(
            std::round(time / parameters.snapshot_instant)),
          false);
    }
    

    // Now if I want a different time step I compute the new one
    // time_step = CFL/
    //   Utilities::truncate_to_n_digits(
    //     eulerian_spray_operator.compute_cell_transport_speed(solution), 3);
    // time_step = CFL*min_cell_length/ eulerian_spray_operator.compute_cell_transport_speed(solution);
    // if the new time step exceeds the final time i cut it
    if((time + time_step) >= final_time)
        time_step = final_time - time;
    // pcout<<"New time step: "<<time_step<<std::endl;
    time += time_step; 
  }

  // A final output
  output_results(timestep_myReal, true);
  timer.print_wall_time_statistics(MPI_COMM_WORLD);
  pcout<<std::endl;
}

template<int dim, int degree>
EulerianSprayProblem<dim, degree>::Postprocessor::Postprocessor(){}

template<int dim, int degree>
void EulerianSprayProblem<dim, degree>::Postprocessor::evaluate_vector_field(
  const DataPostprocessorInputs::Vector<dim> &inputs,
  std::vector<Vector<double>> &computed_quantities) const
{
  const unsigned int n_evaluation_points = inputs.solution_values.size();

  Assert(computed_quantities.size() == n_evaluation_points, ExcInternalError());
  Assert(inputs.solution_values[0].size() == dim + 1, ExcInternalError());

  for(unsigned int p = 0; p < n_evaluation_points; ++p)
  {
    Tensor<1, dim + 1> solution;
    for(unsigned int d = 0; d < dim +1; ++d)
      solution[d] = inputs.solution_values[p](d);
    const Tensor<1, dim> velocity = eulerian_spray_velocity<dim>(solution);
    
    for(unsigned int d = 0; d<dim; ++d)
      computed_quantities[p](d) = velocity[d];
  }
}

template<int dim, int degree>
std::vector<std::string> EulerianSprayProblem<dim, degree>::Postprocessor::
  get_names() const
{
  std::vector<std::string> names;
  for(unsigned int d = 0; d < dim; ++d)
    names.emplace_back("velocity");

  return names;
}

template<int dim, int degree>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
  EulerianSprayProblem<dim, degree>::Postprocessor::get_data_component_interpretation() const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation;

  for(unsigned int d = 0; d < dim; ++d)
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

  return interpretation;
}

template<int dim, int degree>
UpdateFlags
EulerianSprayProblem<dim, degree>::Postprocessor::get_needed_update_flags() const
{
  return update_values;
}
