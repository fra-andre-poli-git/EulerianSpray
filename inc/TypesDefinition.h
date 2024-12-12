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



#endif
