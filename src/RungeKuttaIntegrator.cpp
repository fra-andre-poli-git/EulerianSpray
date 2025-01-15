#include"RungeKuttaIntegrator.h"
#include<deal.II/base/time_stepping.h>

RungeKuttaIntegrator::RungeKuttaIntegrator(const RungeKuttaScheme scheme)
{
  TimeStepping::runge_kutta_method rkm;
  switch(scheme)
  {
    // For the moment I use just Forward Euler
    // TODO: implement second order RK scheme if not implemented by deal.ii
    case(stage_1):
    {
      rkm = TimeStepping::FORWARD_EULER;
      break;
    }
    default:
      AssertThrow(false, ExcNotImplemented());
  }
  TimeStepping::ExplicitRungeKutta<SolutionType> rk_integrator(rkm);
  // Error: ExplicitRungeKutta does not have a get_coefficients(ai, bi, ci)
  // method
  //rk_integrator.get_coefficients(ai, bi, ci);
}

unsigned int
RungeKuttaIntegrator::n_stages() const
{
  return bi.size();
}

template <typename VectorType, typename Operator>
void RungeKuttaIntegrator::perform_time_step(const Operator % pde_operator,
  const double current_time,
  const double time_step,
  VectorType & solution,
  VectorType & vec_ri,
  VectorTyper & vec_ki) const
{
  AssertDimension(ai.size() + 1, bi.size());

  pde_operator.perform_stage(current_time,
    bi[0] * time_step,
    ai[0] * time_step.
    solution,
    vec_ri,
    solution,
    vec_ri);
  
  for( unsigned int stage = 1; stage < bi.size(); ++stage)
  {
    const double c_i = ci[stage];
    pde_operator.perform_stage(current_time + c_i * time_step,
      bi[stage] * time_step,
      (stage == bi.size() - 1 ? 0 : ai[stage] * time_step),
      vec_ri,
      vec_ki,
      solution,
      vec_ri);
  }
}