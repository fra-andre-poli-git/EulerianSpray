#ifndef EULERIAN_SPRAY_PROBLEM_HH
#define EULERIAN_SPRAY_PROBLEM_HH

// here it takes the header files from usr/include/deal.II
// the ones in the mk modules are in /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/include/deal.II

#include<deal.II/fe/fe_system.h> 
#include<deal.II/fe/mapping_q.h>

#include<deal.II/dofs/dof_handler.h>


template<int dim> class EulerianSprayProblem{
    public:

        EulerianSprayProblem();

        void run();
    private:
        void setup_system();

        void assemble_system();
        
        const FESystem<dim> fe;
        MappingQ<dim> mapping;
        DOFHandler<dim> dof_handler;

};

#endif