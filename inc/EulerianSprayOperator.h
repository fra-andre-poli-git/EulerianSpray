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

    void set_neumann_boundary(const types::boundary_id boundary_id);

    void set_dirichlet_boundary(const types::boundary_id boundary_id,
      std::unique_ptr<Function<dim>> dirichlet_function);

    void apply(const Number current_time,
      const SolutionType & src,
      SolutionType & dst) const;

    void perform_lsrk_stage(const Number current_time,
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


    double compute_cell_transport_speed(const SolutionType & solution)
      const;   

    void initialize_vector(SolutionType &vector) const;

  private:
    MatrixFree<dim, Number> data;

    TimerOutput & timer;

    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
      dirichlet_boundaries;
    std::set<types::boundary_id> neumann_boundaries;

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