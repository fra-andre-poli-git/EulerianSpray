#ifndef EULERIAN_SPRAY_PROBLEM_HH
#define EULERIAN_SPRAY_PROBLEM_HH

// here my machine takes the header files from usr/include/deal.II
// the ones in the mk modules are in /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/include/deal.II
#include"TypesDefinition.h"
#include"EulerianSprayOperator.h"
#include"InlinedFunctions.h"
#include<deal.II/grid/tria.h>
#include<deal.II/fe/fe_system.h> 
#include<deal.II/fe/mapping_q.h>
#include<deal.II/dofs/dof_handler.h>
#include<deal.II/numerics/data_postprocessor.h>
#include<deal.II/base/timer.h>
#include<deal.II/base/conditional_ostream.h>
#include<vector>



template<int dim> class EulerianSprayProblem{
  public:
    EulerianSprayProblem();

    void run();

  private:
    // This is the function that makes grid and dofs
    void make_grid_and_dofs();

    void output_results(const unsigned int result_number);
    
    SolutionType solution;

    ConditionalOStream pcout;
    // TODO: put an if to get distributed triangulation
    Triangulation<dim> triangulation;
    const FESystem<dim> fe;
    MappingQ1<dim> mapping;
    DoFHandler<dim> dof_handler;

    double time;

    double time_step;

    TimerOutput timer;

    double final_time;

    EulerianSprayOperator<dim,fe_degree,n_q_points_1d> eulerian_spray_operator;
    
    class Postprocessor : public DataPostprocessor<dim>{
      public:
        Postprocessor();

        virtual void evaluate_vector_field(
          const DataPostprocessorInputs::Vector<dim> &inputs,
          std::vector<Vector<double>> &computed_quantities) const override;

        virtual std::vector<std::string> get_names() const override;

        virtual std::vector<
          DataComponentInterpretation::DataComponentInterpretation>
          get_data_component_interpretation() const override;

        virtual UpdateFlags get_needed_update_flags() const override;

    };
};

#endif