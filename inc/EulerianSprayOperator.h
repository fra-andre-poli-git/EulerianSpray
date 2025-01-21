#ifndef EULERIAN_SPRAY_OPERATOR_HH
#define EULERIAN_SPRAY_OPERATOR_HH

#include"TypesDefinition.h"

#include<deal.II/matrix_free/matrix_free.h>
#include<deal.II/base/function.h>
#include<deal.II/base/vectorization.h>
#include<deal.II/base/timer.h>

using namespace dealii;


// This class implements the evaluators for Eulerian Spray problem
// in analogy to the 'EulerOperator' class in step-67 and 'LaplaceOperator'
// in step-37 or step-59. It implements the differential operator Since this 
// operator is non-linear and does not require
// a matrix interface to be used by a preconditioner, we implement an 'apply'
// function (...)
template<int dim, int degree, int n_points_1d>
class EulerianSprayOperator{
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    EulerianSprayOperator(TimerOutput & timer_output); // TODO: a che serve timer_output?

    void reinit(const Mapping<dim> & mapping,
      const DoFHandler<dim> & dof_handler);

    // Now I put some methods relating boundary conditions and body force
    // for which I have to understand better what I need. For the moment
    // they will be commented.

    // void set_inflow_boundary(const types::boundary_id       boundary_id,
    //                      std::unique_ptr<Function<dim>> inflow_function);

    // void set_subsonic_outflow_boundary(
    //             const types::boundary_id       boundary_id,
    //             std::unique_ptr<Function<dim>> outflow_energy);

    // void set_wall_boundary(const types::boundary_id boundary_id);

    // void set_body_force(std::unique_ptr<Function<dim>> body_force);

    void apply(const double current_time,
      const SolutionType & src,
      SolutionType & dst) const;

    void perform_stage(const Number current_time,
      const Number factor_solution,
      const Number factor_ai,
      const SolutionType & current_ri,
      SolutionType & vec_ki,
      SolutionType & solution,
      SolutionType & next_ri) const;

    void project(const Function<dim> & function,
      SolutionType &solution) const;
    
    std::array<double, 3> compute_errors(const Function<dim> & function,
      const SolutionType & solution) const;

    // I comment the following function because for the moment I am using a
    // fixed time step, I don't compute it
    // double compute_cell_transport_speed(const SolutionType & solution)
    //     const;   

    void initialize_vector(SolutionType &vector) const;

  private:
    MatrixFree<dim, Number> data;

    TimerOutput & timer;

    // TODO: understand this chunk
    // std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
    //     inflow_boundaries;
    // std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
    //     subsonic_outflow_boundaries;
    // std::set<types::boundary_id>   wall_boundaries;
    // std::unique_ptr<Function<dim>> body_force;

    void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> & data,
      SolutionType & dst,
      const SolutionType & src,
      const std::pair<unsigned int, unsigned int> & cell_range) const;

    void local_apply_cell(const MatrixFree<dim, Number> & data,
      SolutionType & dst,
      const SolutionType &src,
      const std::pair<unsigned int, unsigned int> & 
      cell_range) const;

    void local_apply_face(
      const MatrixFree<dim, Number> & data,
      SolutionType & dst,
      const SolutionType & src,
      const std::pair<unsigned int, unsigned int> & face_range) const;
        
    void local_apply_boundary_face(
      const MatrixFree<dim, Number> & data,
      SolutionType & dst,
      const SolutionType &src,
      const std::pair<unsigned int, unsigned int> & face_range) const;
};


// Here I declare the helper function that I use for evaluation in
// EulerianSprayOperator methods. This is a template function that is used for
// the evaluation.
// Completely copied from tutorial 67
template <int dim, typename Number>
VectorizedArray<Number>
evaluate_function(const Function<dim> &,
                  const Point<dim, VectorizedArray<Number>> &,
                  const unsigned int);

template <int dim, typename Number, int n_components = dim + 1>
Tensor<1, n_components, VectorizedArray<Number>>
evaluate_function(const Function<dim> &,
                  const Point<dim,
                  VectorizedArray<Number>> &);


#endif