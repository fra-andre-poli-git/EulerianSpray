#include"EulerianSprayProblem.h"
#include"TypesDefinition.h"
#include"RungeKuttaIntegrator.h"
#include"Functions.h"
#include"InlinedFunctions.h"
#include<deal.II/grid/grid_generator.h>
#include<deal.II/grid/tria_description.h>
#include<deal.II/grid/grid_tools.h>
#include<deal.II/fe/fe_dgq.h>
#include<deal.II/base/utilities.h>
#include<deal.II/numerics/vector_tools.h>
#include<deal.II/numerics/data_out.h>


#include<iostream>


template <int dim>
EulerianSprayProblem<dim>::EulerianSprayProblem():
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),    
    fe(FE_DGQ<dim>(fe_degree),dim+1),
    // mapping only works with a degree>=2
    // mapping(fe_degree >= 2 ? fe_degree : 2),
    mapping(),
    dof_handler(triangulation),
    time(0),
    time_step(0),
    timer(pcout, TimerOutput::never, TimerOutput::wall_times),
    eulerian_spray_operator(timer)
    {}

template <int dim>
void EulerianSprayProblem<dim>::make_grid_and_dofs()
{
  // In step 67 this is a global variable. I may opt for a solution like 
  // Felotti's one, which uses a parameter memeber and make it
  // parameters.testcase
  switch(testcase)
  {
    case 1:{
      // Note the fact that last argument is true, therefore we get different
      // boundary_id for the four boundaries
      GridGenerator::subdivided_hyper_rectangle(triangulation,
        {n_el_x_direction,5},
        Point<dim>(-1,0),
        Point<dim>(1,0.1),
        true);

      // TODO: put an if if I am using parallel::distributed::triangulation
      // These three lines make a periodicity constraint on top and bottom
      // boundaries
      std::vector<GridTools::PeriodicFacePair<
        typename Triangulation<dim>::cell_iterator>> periodicity_vector;
      GridTools::collect_periodic_faces(triangulation,
        2,
        3,
        1,
        periodicity_vector);
      triangulation.add_periodicity(periodicity_vector);

      eulerian_spray_operator.set_neumann_boundary(0);
      eulerian_spray_operator.set_neumann_boundary(1);

      final_time = 0.5;
      break;
    }
    case 2:
    {
      GridGenerator::subdivided_hyper_rectangle(triangulation,
        {n_el_x_direction,5},
        Point<dim>(-1,0),
        Point<dim>(1,0.1),
        true);
      std::vector<GridTools::PeriodicFacePair<
        typename Triangulation<dim>::cell_iterator>> periodicity_vector;
      GridTools::collect_periodic_faces(triangulation,
        2,
        3,
        1,
        periodicity_vector);
      triangulation.add_periodicity(periodicity_vector);

      eulerian_spray_operator.set_neumann_boundary(0);
      eulerian_spray_operator.set_neumann_boundary(1);
      final_time = 0.5;
      break;
    }
    case 3:
    {
      GridGenerator::hyper_cube(triangulation, -0.5,0.5, true);
      eulerian_spray_operator.set_neumann_boundary(0);
      eulerian_spray_operator.set_neumann_boundary(1);
      eulerian_spray_operator.set_neumann_boundary(2);
      eulerian_spray_operator.set_neumann_boundary(3);
      final_time = 1;
      triangulation.refine_global(n_global_refinements);
      break;
    }
    case 4:
    {
      GridGenerator::hyper_cube(triangulation, 0, 1, true);
    }
  }

  dof_handler.distribute_dofs(fe);

  eulerian_spray_operator.reinit(mapping, dof_handler);
  eulerian_spray_operator.initialize_vector(solution);

  std::cout<< "Number of degrees of freedom "<<dof_handler.n_dofs()
            << " ( = " << (dim + 1) << " [vars] x "
            << triangulation.n_global_active_cells() << " [cells] x "
            << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
            << std::endl;
}

template<int dim>
void EulerianSprayProblem<dim>::output_results(const unsigned int result_number)
{
  {
    TimerOutput::Scope t(timer, "output");

    Postprocessor postprocessor;
    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true; // TODO: a che serve?
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    {
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
    }

    data_out.add_data_vector(solution, postprocessor);


    Vector<double> mpi_owner(triangulation.n_active_cells());
    mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    data_out.add_data_vector(mpi_owner, "owner");

    data_out.build_patches(mapping,
      fe.degree,
      DataOut<dim>::curved_inner_cells);

    const std::string filename = 
      "./results/solution_" +
        Utilities::int_to_string(result_number, 3) + ".vtu";

    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
  }
}

template <int dim>
void EulerianSprayProblem<dim>::run()
{
  {
    const unsigned int n_vect_number = VectorizedArray<Number>::size();
    const unsigned int n_vect_bits   = 8 * sizeof(Number) * n_vect_number;

    pcout << "Running with "
          << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
          << " MPI processes" << std::endl;
    pcout << "Vectorization over " << n_vect_number << ' '
          << (std::is_same<Number, double>::value ? "doubles" : "floats")
          << " = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    }

  make_grid_and_dofs();

  const SSPRungeKuttaIntegrator integrator(scheme);
  

  SolutionType rk_register1;
  SolutionType rk_register2;
  rk_register1.reinit(solution);
  rk_register2.reinit(solution);

  // Here I should initialize the solution
  // Step 67 does this projecting the exact solution onto the solution vector
  // but I don't have an exact solution for every time step, therefore I use the
  // initial solution
  eulerian_spray_operator.project(InitialSolution<dim>(), solution);

  //This small chunk aims at finding h, the smallest distance between two nodes
  double min_vertex_distance = std::numeric_limits<double>::max();
  // If the test is a 1d test in disguise, I will take as h the size of the cell
  // in x direction
  if(testcase==1 || testcase==2)
  {
    for(const auto & cell : triangulation.active_cell_iterators())
      min_vertex_distance = 
        std::min(min_vertex_distance, cell->extent_in_direction(0));
  }
  // Otherwise I will use the minimum distance between vertices
  else
  {
    for(const auto & cell : triangulation.active_cell_iterators())
      min_vertex_distance =
        std::min(min_vertex_distance, cell->minimum_vertex_distance());
  }
  
  
  // with MPI here I have to make the minimum over all processors
  min_vertex_distance =
    Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);
  double CFL = 1;
  // Now I set the time step to be exactly the biggest to satisfy CFL condition
  time_step = CFL/std::pow((fe_degree+1),2) * min_vertex_distance;
  pcout << "Time step size: " << time_step
    << ", minimal h: " << min_vertex_distance
    << std::endl
    << std::endl;

    output_results(0);
  // This is the time loop
  time = 0;
  unsigned int timestep_number = 0;
  while(time < final_time - 1e-12)
  {
    ++timestep_number;
    // Here the integration in time is performed by the integrator
    {
      // TODO: uncomment this for varying time step
      // if(timestep_number % 5 == 0)
      //   time_step = CFL /std::pow((2*fe_degree+1),2) * min_vertex_distance *
      //     integrator.n_stages() / Utilities::truncate_to_n_digits(
      //       eulerian_spray_operator.compute_cell_transport_speed(solution), 3);
      integrator.perform_time_step(eulerian_spray_operator,
        time,
        time_step,
        solution,
        rk_register1,
        rk_register2);
    }
    std::cout<<"Performed timestep at time: "<<time<<std::endl;
    if (static_cast<int>(time / snapshot) !=
              static_cast<int>((time - time_step) / snapshot) ||
            time >= final_time - 1e-12)
          output_results(
            static_cast<unsigned int>(std::round(time / snapshot)));
    time += time_step;
  }
  timer.print_wall_time_statistics(MPI_COMM_WORLD);
  pcout<<std::endl;
}

template <int dim>
EulerianSprayProblem<dim>::Postprocessor::Postprocessor(){}

template <int dim>
void EulerianSprayProblem<dim>::Postprocessor::evaluate_vector_field(
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

template<int dim>
std::vector<std::string> EulerianSprayProblem<dim>::Postprocessor::get_names()
  const
{
  std::vector<std::string> names;
  for(unsigned int d = 0; d < dim; ++d)
    names.emplace_back("velocity");

  return names;
}

template<int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
  EulerianSprayProblem<dim>::Postprocessor::get_data_component_interpretation()
  const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation;

  for(unsigned int d = 0; d < dim; ++d)
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

  return interpretation;
}

template<int dim>
UpdateFlags
EulerianSprayProblem<dim>::Postprocessor::get_needed_update_flags() const
{
  return update_values;
}

// Instantiations of the template
template class EulerianSprayProblem<2>;