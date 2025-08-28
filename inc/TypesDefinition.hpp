#ifndef EULERIAN_SPRAY_TYPES_DEFINITION_HH
#define EULERIAN_SPRAY_TYPES_DEFINITION_HH

#include <deal.II/lac/vector.h>
#include<deal.II/lac/la_parallel_vector.h>

// TODO: this shouldn't be here...
using namespace dealii;

using Number = double;

// Step 67 uses a LinearAlgebra::distributed::Vector<Number> 
// which needs #include<deal.II/lac/la_parallel_vector.h>
using SolutionType = LinearAlgebra::distributed::Vector<Number>;
// I have to use distributed one since I use solution.local_element(i), for
// instance in line 121 of EulerianSprayOperator.cpp
// TODO: make the code work for both Vector and distributed::Vector


enum NumericalFlux
{
  local_lax_friedrichs,
  local_lax_friedrichs_modified,
  harten_lax_vanleer,
  godunov,
};

enum RungeKuttaScheme
{
  lsrk_stage_3_order_3,
  lsrk_stage_5_order_4,
  lsrk_stage_7_order_4,
  lsrk_stage_9_order_5,
  forward_euler,
  ssp_stage_2_order_2,
  ssp_stage_3_order_3,
};

#endif