#include"EulerianSprayProblem.h"
#include<deal.II/grid/grid_generator.h>
#include<deal.II/fe/fe_dgq.h>
#include<deal.II/base/utilities.h>
#include<iostream>

constexpr unsigned int testcase = 1;
constexpr int fe_degree = 0;
constexpr int n_global_refinements = 4;

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
            triangulation.refine_global(2);
            break;
        }
        default:
            Assert(false, ExcNotImplemented());     
    }

    triangulation.refine_global(n_global_refinements);

    dof_handler.distribute_dofs(fe);

    std::cout<< "Number of degrees of freedom"<<dof_handler.n_dofs()
             << " ( = " << (dim + 2) << " [vars] x "
             << triangulation.n_global_active_cells() << " [cells] x "
             << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
             << std::endl;
}

template <int dim>
void EulerianSprayProblem<dim>::run(){
    //Qua nello step 67 c'è un pezzetto per quando si usa MPI

    make_grid_and_dofs();

    time_step = 1./std::pow((fe_degree+1),2)/* *h */;


}

// Instantiation of the template
template class EulerianSprayProblem<1>;
template class EulerianSprayProblem<2>;
