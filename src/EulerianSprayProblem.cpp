#include"EulerianSprayProblem.h"

template <int dim>
void EulerianSprayProblem::make_grid_and_dofs(){
    // In step 67 this is a global variable. I may opt for a solution like Felotti's one, which uses a parameter memeber and make it parameters.testcase
    switch(testcase){
        case 1:
            
    }
}

template <int dim>
void EulerianSprayProblem::run(){
    //Qua nello step 67 c'è un pezzetto per quando si usa MPI

    make_grid_and_dofs();
}