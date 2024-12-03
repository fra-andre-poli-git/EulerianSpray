#include"EulerianSprayProblem.h"
#include<deal.II/grid/grid_generator.h>
#include<deal.II/fe/fe_dgq.h>

constexpr unsigned int testcase = 1;
constexpr int fe_degree=0;

template <int dim>
EulerianSprayProblem<dim>::EulerianSprayProblem():
    fe(FE_DGQ<dim>(fe_degree),dim+1), //il +1 è perché ho momento nelle direzioni delle dimensioni + massa (a differenza di Eulero non ho energia)
    mapping(fe_degree >= 2 ? fe_degree : 2), // mapping funziona solo con un grado >=2
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
            triangulation.refine_global(2);
            break;
        }
        default:
            Assert(false, ExcNotImplemented());
            
    }
}

template <int dim>
void EulerianSprayProblem<dim>::run(){
    //Qua nello step 67 c'è un pezzetto per quando si usa MPI

    make_grid_and_dofs();


}

// Instantiation of the template
template class EulerianSprayProblem<1>;
template class EulerianSprayProblem<2>;
