#ifndef PARAMETERS_EULERIAN_SPRAY_HH
#define PARAMETERS_EULERIAN_SPRAY_HH

#include"TypesDefinition.h"
#include<deal.II/base/parameter_handler.h>


struct Parameters
{
  unsigned int testcase;
  // unsigned int fe_degree;
  // unsigned int n_q_points_1d;
  unsigned int n_el_x_direction;
  double final_time;
  double snapshot;
  RungeKuttaScheme scheme;
  NumericalFlux numerical_flux_type;

  static void declare_parameters(ParameterHandler & prm);
  void parse_parameters(ParameterHandler & prm);
};
#endif