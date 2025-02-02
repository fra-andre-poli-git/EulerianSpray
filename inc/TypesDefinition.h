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
constexpr int fe_degree = 0;
constexpr int n_global_refinements = 7;
constexpr unsigned int n_q_points_1d = fe_degree + 2;
constexpr double parameter_final_time = 0.5;
constexpr double snapshot = 0.05;
enum RungeKuttaScheme
{
  lsrk_stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
  lsrk_stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
  lsrk_stage_7_order_4, /* Tselios, Simos, 2007 */
  lsrk_stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
  forward_euler,
  ssp_stage_2_order_2,
  ssp_stage_3_order_3,
};
constexpr RungeKuttaScheme scheme = ssp_stage_3_order_3;

// I define this enumerator even thoug it has only one element since I may
// decide to implement other types of numerical flux
enum NumericalFlux
{
  local_lax_friedrichs,
};
constexpr NumericalFlux numerical_flux_type = local_lax_friedrichs;

#endif