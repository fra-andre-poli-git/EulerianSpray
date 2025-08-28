#ifndef EULERIAN_SPRAY_PROBLEM_HH
#define EULERIAN_SPRAY_PROBLEM_HH

#include"TypesDefinition.h"
#include"EulerianSprayOperator.hpp"
#include"EulerianSprayOperator_IMP.h"
#include"InlinedFunctions.h"
#include"Parameters.h"

#include<deal.II/grid/tria.h>
#include<deal.II/fe/fe_system.h> 
#include<deal.II/fe/mapping_q1.h>// TODO or just q?
#include<deal.II/dofs/dof_handler.h>
#include<deal.II/numerics/data_postprocessor.h>
#include<deal.II/base/timer.h>
#include<deal.II/base/conditional_ostream.h>
#include<vector>



template<int dim, int degree>
class EulerianSprayProblem{
  public:
    EulerianSprayProblem(const Parameters &);

    void run();

  private:
    // This is the function that makes grid and dofs
    void make_grid_and_dofs();

    void output_results(const unsigned int result_number, bool final_time);

    SolutionType solution;

    ConditionalOStream pcout;
    // TODO: put an if to get distributed triangulation
    Triangulation<dim> triangulation;

    Parameters parameters;

    const FESystem<dim> fe;

    MappingQ1<dim> mapping;
    
    DoFHandler<dim> dof_handler;

    double time;

    double time_step;

    TimerOutput timer;

    double final_time;

    static constexpr int n_q_points_1d = degree + 2;
    EulerianSprayOperator<dim,degree,n_q_points_1d> eulerian_spray_operator;
    
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