#include"EulerianSprayProblem.h"
#include"TypesDefinition.h"
#include"RungeKuttaIntegrator.h"
#include"InitialSolution.h"
#include"InlinedFunctions.h"
#include<deal.II/grid/grid_generator.h>
#include<deal.II/fe/fe_dgq.h>
#include<deal.II/base/utilities.h>
#include<deal.II/numerics/vector_tools.h>


#include<iostream>


template <int dim>
EulerianSprayProblem<dim>::EulerianSprayProblem():
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    // Il +1 è perché ho momento nelle direzioni delle dimensioni + massa
    // (a differenza di Eulero non ho energia)    
    fe(FE_DGQ<dim>(fe_degree),dim+1),
    // mapping only works with a degree>=2
    mapping(fe_degree >= 2 ? fe_degree : 2),
    dof_handler(triangulation),
    time(0),
    time_step(0),
    timer(pcout, TimerOutput::never, TimerOutput::wall_times),
    eulerianspray_operator(timer)
    {}

template <int dim>
void EulerianSprayProblem<dim>::make_grid_and_dofs(){
    // In step 67 this is a global variable. I may opt for a solution like Felotti's one, which uses a parameter memeber and make it parameters.testcase
    switch(testcase){
        case 1:{
            GridGenerator::hyper_cube(triangulation, -1., 1.);
            // I don't know why, but in step 67 it refines the mesh two times previously than n_global_refinement
            //triangulation.refine_global(2);
            final_time = 0.5;
            break;
        }
    }

    triangulation.refine_global(n_global_refinements);

    dof_handler.distribute_dofs(fe);

    eulerianspray_operator.reinit(mapping, dof_handler);
    eulerianspray_operator.initialize_vector(solution);

    std::cout<< "Number of degrees of freedom "<<dof_handler.n_dofs()
             << " ( = " << (dim + 1) << " [vars] x "
             << triangulation.n_global_active_cells() << " [cells] x "
             << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
             << std::endl;
}



template <int dim>
void EulerianSprayProblem<dim>::run()
{
  //Qua nello step 67 c'è un pezzetto per quando si usa MPI

  make_grid_and_dofs();

  const RungeKuttaIntegrator integrator(scheme);

  SolutionType rk_register1;
  SolutionType rk_register2;
  rk_register1.reinit(solution);
  rk_register2.reinit(solution);

  // Here I should initialize the solution
  // Step 67 does this projecting the exact solution onto the solution vector
  // but I don't have an exact solution for every time step, therefore I use the initial solution
  eulerianspray_operator.project(InitialSolution<dim>(), solution);

  //This small chunk aims at finding h, the smallest distance between two nodes
  double min_vertex_distance = std::numeric_limits<double>::max();
  for(const auto & cell : triangulation.active_cell_iterators()){
      min_vertex_distance =
          std::min(min_vertex_distance, cell->minimum_vertex_distance());
  }
  // with MPI here I have to make the minimum over all processors


  // Now I set the time step to be exactly the biggest to satisfy CFL condition
  time_step = 1./std::pow((fe_degree+1),2) * min_vertex_distance;
  std::cout << "Time step: " << time_step << std::endl;

  // This is the time loop
  time = 0;
  unsigned int timestep_number = 0;
  while(time < final_time - 1e-12)
  {
    ++timestep_number;
    // Here the integration in time is performed by the integrator
    {
      integrator.perform_time_step(eulerianspray_operator,
        time,
        time_step,
        solution,
        rk_register1,
        rk_register2);
    }
    std::cout<<"Performed timestep at time: "<<time<<std::endl;
    time += time_step;
  }
}

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
    const double density = solution[0];
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
template class EulerianSprayProblem<1>;
template class EulerianSprayProblem<2>;
template class EulerianSprayProblem<3>;