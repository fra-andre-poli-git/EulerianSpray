#ifndef EULERIAN_SPRAY_PROBLEM_HH
#define EULERIAN_SPRAY_PROBLEM_HH

// here my machine takes the header files from usr/include/deal.II
// the ones in the mk modules are in /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/include/deal.II
#include"TypesDefinition.h"
#include"EulerianSprayOperator.h"
#include<deal.II/grid/tria.h>
#include<deal.II/fe/fe_system.h> 
#include<deal.II/fe/mapping_q.h>
#include<deal.II/dofs/dof_handler.h>
#include<deal.II/numerics/data_postprocessor.h>
#include<vector>



template<int dim> class EulerianSprayProblem{
    public:

        EulerianSprayProblem();

        void run();
    private:
        // This is the function that makes grid and dofs
        void make_grid_and_dofs();
        //Here the version with p4est would need a different type: parallel::distributed::Triangulation<dim> triangulation;

        //If I decide to use MPI I will take the opportunity to define a ConditionalOStream here

        SolutionType solution;
        Triangulation<dim> triangulation;
        const FESystem<dim> fe;
        MappingQ<dim> mapping;
        DoFHandler<dim> dof_handler;

        EulerianSprayOperator<dim,fe_degree,n_q_points_1d> eulerianspray_operator;

        double final_time, time, time_step;
        // Questo magari lo dichiaro meglio quando capisco cosa voglio fare di post processing
        class Postprocessor : public DataPostprocessor<dim>{
            public:
                Postprocessor();
        };

};

#endif