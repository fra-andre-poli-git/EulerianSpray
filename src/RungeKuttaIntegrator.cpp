//
#include"RungeKuttaIntegrator.h"
#include"TypesDefinition.h"
#include"EulerianSprayOperator.h"
#include<deal.II/base/time_stepping.h>
RungeKuttaIntegrator::RungeKuttaIntegrator(const RungeKuttaScheme scheme)
{
  TimeStepping::runge_kutta_method lsrk;
  switch(scheme)
  {
    case stage_3_order_3:
    {
      lsrk = TimeStepping::LOW_STORAGE_RK_STAGE3_ORDER3;
      break;
    }
    case stage_5_order_4:
    {
      lsrk = TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4;
      break;
    }
    case stage_7_order_4:
    {
      lsrk = TimeStepping::LOW_STORAGE_RK_STAGE7_ORDER4;
      break;
    }
    case stage_9_order_5:
    {
      lsrk = TimeStepping::LOW_STORAGE_RK_STAGE9_ORDER5;
      break;
    }
    default:
      AssertThrow(false, ExcNotImplemented());
  } 
  TimeStepping::LowStorageRungeKutta<SolutionType> rk_integrator(lsrk);
  rk_integrator.get_coefficients(ai, bi, ci);
}

unsigned int
RungeKuttaIntegrator::n_stages() const
{
  return bi.size();
}

// For the moment i will rely on the lsrk implementation by step 67. The problem
// is that it is not general to all explicit Runge Kutta, therefore I cannot
// adapt it to Explicit Euler or some second order scheme. The fact is that
// other RK schemes are already implemented by TimeStepping, but as far as I can
// understand the interface with a Matrix Free implementation is not possible.
// The future implementation may be very general to all explicit Runge Kutta
// schemes, but can be also very expansive if the method is a low storage one,
// since in this implementation I mantain al Ki up to the computation of the new
// solution
template <typename VectorType, typename Operator>
void RungeKuttaIntegrator::perform_time_step(const Operator &pde_operator,
  const double    current_time,
  const double    time_step,
  VectorType &    solution,
  VectorType &    vec_ri,
  VectorType &    vec_ki) const
{
  AssertDimension(ai.size() + 1, bi.size());

  pde_operator.perform_stage(current_time,
                              bi[0] * time_step,
                              ai[0] * time_step,
                              solution,
                              vec_ri,
                              solution,
                              vec_ri);

  for (unsigned int stage = 1; stage < bi.size(); ++stage)
    {
      const double c_i = ci[stage];
      pde_operator.perform_stage(current_time + c_i * time_step,
                                  bi[stage] * time_step,
                                  (stage == bi.size() - 1 ?
                                    0 :
                                    ai[stage] * time_step),
                                  vec_ri,
                                  vec_ki,
                                  solution,
                                  vec_ri);
    }
}

//Instantiations of the template function
template void RungeKuttaIntegrator::perform_time_step<SolutionType,
  EulerianSprayOperator<1,2,4>>(const EulerianSprayOperator<1,2,4> &,
    const double,
    const double,
    SolutionType &,
    SolutionType &,
    SolutionType &) const;

template void RungeKuttaIntegrator::perform_time_step<SolutionType,
  EulerianSprayOperator<2,2,4>>(const EulerianSprayOperator<2,2,4> &,
    const double,
    const double,
    SolutionType &,
    SolutionType &,
    SolutionType &) const;

template void RungeKuttaIntegrator::perform_time_step<SolutionType,
  EulerianSprayOperator<3,2,4>>(const EulerianSprayOperator<3,2,4> &,
    const double,
    const double,
    SolutionType &,
    SolutionType &,
    SolutionType &) const;