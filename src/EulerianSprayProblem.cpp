#include"EulerianSprayProblem.h"
#include<deal.II/grid/grid_generator.h>
#include<deal.II/fe/fe_dgq.h>
#include<deal.II/base/utilities.h>
#include<iostream>

constexpr unsigned int testcase = 1;
constexpr int fe_degree = 2;
constexpr int n_global_refinements = 7;

template <int dim>
EulerianSprayProblem<dim>::EulerianSprayProblem():
    fe(FE_DGQ<dim>(fe_degree),dim+1), //il +1 è perché ho momento nelle direzioni delle dimensioni + massa (a differenza di Eulero non ho energia)
    mapping(fe_degree >= 2 ? fe_degree : 2), // mapping only works with a degree>=2
    dof_handler(triangulation),
    time(0),
    time_step(0)
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
        default:
            Assert(false, ExcNotImplemented());     
    }

    triangulation.refine_global(n_global_refinements);

    dof_handler.distribute_dofs(fe);

    std::cout<< "Number of degrees of freedom "<<dof_handler.n_dofs()
             << " ( = " << (dim + 1) << " [vars] x "
             << triangulation.n_global_active_cells() << " [cells] x "
             << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
             << std::endl;
}

template <int dim>
void EulerianSprayProblem<dim>::run(){
    //Qua nello step 67 c'è un pezzetto per quando si usa MPI

    make_grid_and_dofs();

    //This small chunk aims at finding h, the smallest distance between two nodes
    double min_vertex_distance = std::numeric_limits<double>::max();
    for(const auto & cell : triangulation.active_cell_iterators()){
        min_vertex_distance =
            std::min(min_vertex_distance, cell->minimum_vertex_distance());
    }
    // with MPI here I have to make the minimum over all processors

    // Here I should initialize the solution
    // Step 67 does this projectin the exact solution onto the solution vector


    // Now I set the time step to be exactly the biggest to satisfy CFL condition
    time_step = 1./std::pow((fe_degree+1),2) * min_vertex_distance;
    std::cout << "Time step: " << time_step << std::endl;

    // This is the time loop
    time = 0; // I don't know why time is defined in the class, maybe it will be useful in other function
    unsigned int timestep_number = 0;
    while(time < final_time - 1e-12){
        timestep_number++;

        // Here the integration in time is performed by class called integrator



        time += time_step;
    }


}

// Instantiation of the template
template class EulerianSprayProblem<1>;
template class EulerianSprayProblem<2>;
template class EulerianSprayProblem<3>;
