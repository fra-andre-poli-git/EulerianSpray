#ifndef EULERIAN_SPRAY_TYPES_DEFINITION_HH
#define EULERIAN_SPRAY_TYPES_DEFINITION_HH

#include <deal.II/lac/vector.h>
#include<deal.II/lac/la_parallel_vector.h>

using namespace dealii;

using Number = double;

// Step 67 uses a LinearAlgebra::distributed::Vector<Number> 
// which needs #include<deal.II/lac/la_parallel_vector.h>
using SolutionType = LinearAlgebra::distributed::Vector<Number>;
// I have to use distributed one since I use solution.local_element(i), for
// instance in line 121 of EulerianSprayOperator.cpp
// TODO: make the code work for both Vector and distributed::Vector

constexpr unsigned int testcase = 1;
constexpr int fe_degree = 2;
constexpr int n_global_refinements = 7;
constexpr unsigned int n_q_points_1d = fe_degree + 2;
enum RungeKuttaScheme
{
    stage_1
};
constexpr RungeKuttaScheme scheme = stage_1;

// I define this enumerator even thoug it has only one element since I may
// decide to implement other types of numerical flux
enum NumericalFlux
{
    local_lax_friedrichs,
};
constexpr NumericalFlux numerical_flux_type = local_lax_friedrichs;

#endif