#ifndef EULERIAN_SPRAY_TYPES_DEFINITION_HH
#define EULERIAN_SPRAY_TYPES_DEFINITION_HH

#include <deal.II/lac/vector.h>

using namespace dealii;

using Number = double;

// Step 67 uses a LinearAlgebra::distributed::Vector<Number> 
// which needs #include<deal.II/lac/la_parallel_vector.h>
using SolutionType = Vector<Number>;

constexpr unsigned int testcase = 1;
constexpr int fe_degree = 2;
constexpr int n_global_refinements = 7;
constexpr unsigned int n_q_points_1d = fe_degree + 2;

// I define this enumerator even thoug it has only one element since I may
// decide to implement other types of numerical flux
enum NumericalFlux{
    local_lax_friedrichs,
};
constexpr NumericalFlux numerical_flux_type = local_lax_friedrichs;

#endif