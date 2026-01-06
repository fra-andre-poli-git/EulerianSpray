#ifndef EULERIAN_SPRAY_OPERATOR_HH
#define EULERIAN_SPRAY_OPERATOR_HH

#include"TypesDefinition.hpp"
#include"Parameters.hpp"

#include<deal.II/matrix_free/matrix_free.h>
#include<deal.II/base/function.h>
#include<deal.II/base/vectorization.h>
#include<deal.II/base/timer.h>
#include<deal.II/fe/fe_system.h> 

using namespace dealii;


// This class implements the evaluators for Eulerian Spray problem
// in analogy to the 'EulerOperator' class in step-67 and 'LaplaceOperator'
// in step-37 or step-59. It implements the differential operator. Since this 
// operator is non-linear and does not require
// a matrix interface to be used by a preconditioner, we implement an 'apply'
// function (...)
template<int dim, int degree, int n_q_points_1d>
class EulerianSprayOperator{
  public:
    // static constexpr unsigned int n_quadrature_points_1d = n_q_points_1d;
    // TODO: capire meglio a che serve timer_output
    
    // Constructor
    EulerianSprayOperator(TimerOutput & timer_output,const Parameters & params);

    static constexpr int polynomial_degree = degree;

    void reinit(const Mapping<dim> & mapping,
      const DoFHandler<dim> & dof_handler);

    void set_neumann_boundary(const types::boundary_id boundary_id);

    void set_dirichlet_boundary(const types::boundary_id boundary_id,
      std::unique_ptr<Function<dim>> dirichlet_function);

    void apply(const myReal current_time,
      const SolutionType & src,
      SolutionType & dst) const;

    void perform_lsrk_stage(const myReal current_time,
      const myReal factor_solution,
      const myReal factor_ai,
      const SolutionType & current_ri,
      SolutionType & vec_ki,
      SolutionType & solution,
      SolutionType & next_ri) const;

    void project(const Function<dim> & function,
      SolutionType &solution) const;
    
    std::array<double, 2> compute_errors(const Function<dim> & function,
      const SolutionType & solution) const;

    double compute_cell_transport_speed(const SolutionType & solution) const;   

    void initialize_vector(SolutionType &vector) const;

    void bound_preserving_projection_1d(SolutionType & solution,
      const DoFHandler<dim> & dof_handler,
      const MappingQ1<dim> & mapping,
      const FESystem<dim> & fe) const;

    void bound_preserving_projection(SolutionType & solution,
      const DoFHandler<dim> & dof_handler,
      const MappingQ1<dim> & mapping,
      const FESystem<dim> & fe) const;

    void compute_velocity_extrema_1d(const SolutionType & solution);

    void compute_velocity_max_norm(const SolutionType & solution);

    myReal get_max_velocity() const {return max_velocity;};

    myReal get_min_velocity() const {return min_velocity;};
      
    void set_1d_in_disguise(){one_dimensional_in_disguise = true;};

    bool get_1d_in_disguise() const {return one_dimensional_in_disguise;};



  private:
    // MatrixFree<dim, myReal> class collects all the data that is stored for
    // the matrix free implementation.
    // The stored data can be subdivided into three main components:
    //  - DoFInfo: It stores how local degrees of freedom relate to global
    //      degrees of freedom. It includes a description of constraints that
    //      are evaluated as going through all local degrees of freedom on a
    //      cell.
    //  - MappingInfo: It stores the transformations from real to unit cells
    //      that are necessary in order to build derivatives of finite element
    //      functions and find location of quadrature weights in physical space.
    //  - ShapeInfo: It contains the shape functions of the finite element,
    //      evaluated on the unit cell.
    MatrixFree<dim, myReal> data;

    TimerOutput & timer;

  public:
    const Parameters & parameters;
  private:

    // I store the maximum and the minimum of the initial velocity, to be used
    // in the positivity limiter
    myReal max_velocity, min_velocity;

    bool one_dimensional_in_disguise = false;

    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
      dirichlet_boundaries;
    std::set<types::boundary_id> neumann_boundaries;

    void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, myReal> & data,
      SolutionType & dst,
      const SolutionType & src,
      const std::pair<unsigned int, unsigned int> & cell_range) const;

    void local_apply_cell(const MatrixFree<dim, myReal> & data,
      SolutionType & dst,
      const SolutionType &src,
      const std::pair<unsigned int, unsigned int> & 
      cell_range) const;

    void local_apply_face(
      const MatrixFree<dim, myReal> & data,
      SolutionType & dst,
      const SolutionType & src,
      const std::pair<unsigned int, unsigned int> & face_range) const;
        
    void local_apply_boundary_face(
      const MatrixFree<dim, myReal> & data,
      SolutionType & dst,
      const SolutionType &src,
      const std::pair<unsigned int, unsigned int> & face_range) const;

};


// Here I declare the helper function that I use for evaluation in
// EulerianSprayOperator methods. This is a template function that is used for
// the evaluation.
template <int dim, typename myReal>
VectorizedArray<myReal>
evaluate_function(const Function<dim> &,
                  const Point<dim, VectorizedArray<myReal>> &,
                  const unsigned int);

template <int dim, typename myReal, int n_components = dim + 1>
Tensor<1, n_components, VectorizedArray<myReal>>
evaluate_function(const Function<dim> &,
                  const Point<dim,
                  VectorizedArray<myReal>> &);


#endif