#ifndef RUNGE_KUTTA_INTEGRATOR_IMP_H
#define RUNGE_KUTTA_INTEGRATOR_IMP_H  
#include"RungeKuttaIntegrator.h"
#include"TypesDefinition.h"
#include"EulerianSprayOperator.h"
#include"EulerianSprayOperator_IMP.h"
#include<deal.II/base/time_stepping.h>

template <typename VectorType, typename Operator, int dim>
LSRungeKuttaIntegrator<VectorType,Operator, dim>::LSRungeKuttaIntegrator(
  const RungeKuttaScheme scheme)
{
  TimeStepping::runge_kutta_method lsrk;
  switch(scheme)
  {
    case lsrk_stage_3_order_3:
    {
      lsrk = TimeStepping::LOW_STORAGE_RK_STAGE3_ORDER3;
      break;
    }
    case lsrk_stage_5_order_4:
    {
      lsrk = TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4;
      break;
    }
    case lsrk_stage_7_order_4:
    {
      lsrk = TimeStepping::LOW_STORAGE_RK_STAGE7_ORDER4;
      break;
    }
    case lsrk_stage_9_order_5:
    {
      lsrk = TimeStepping::LOW_STORAGE_RK_STAGE9_ORDER5;
      break;
    }
    case forward_euler:
    case ssp_stage_2_order_2:
    case ssp_stage_3_order_3:
      AssertThrow(false,
        ExcMessage("You are using a wrong Runge Kutta scheme for this"
          " constructor"));
    default:
      AssertThrow(false, ExcNotImplemented());
  } 
  TimeStepping::LowStorageRungeKutta<SolutionType> rk_integrator(lsrk);
  rk_integrator.get_coefficients(ai, bi, ci);
}

template <typename VectorType, typename Operator, int dim>
unsigned int
LSRungeKuttaIntegrator<VectorType,Operator, dim>::n_stages() const
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
template <typename VectorType, typename Operator, int dim>
void LSRungeKuttaIntegrator<VectorType,Operator, dim>::perform_time_step(
  const Operator &pde_operator,
  const double    current_time,
  const double    time_step,
  VectorType &    solution,
  VectorType &    vec_ri,
  VectorType &    vec_ki,
  // These three are needed for the limiter
  const DoFHandler<dim> & dof_handler,
  const MappingQ1<dim> & mapping,
  const FESystem<dim> & fe) const
{
  AssertDimension(ai.size() + 1, bi.size());

  pde_operator.perform_lsrk_stage(current_time,
    bi[0] * time_step,
    ai[0] * time_step,
    solution,
    vec_ri,
    solution,
    vec_ri);
  pde_operator.apply_positivity_limiter(solution, dof_handler, mapping, fe);
  for (unsigned int stage = 1; stage < bi.size(); ++stage)
    {
      const double c_i = ci[stage];
      pde_operator.perform_lsrk_stage(current_time + c_i * time_step,
        bi[stage] * time_step,
        (stage == bi.size() - 1 ?
          0 :
          ai[stage] * time_step),
        vec_ri,
        vec_ki,
        solution,
        vec_ri);
      pde_operator.apply_positivity_limiter(solution, dof_handler, mapping, fe);
    }
}

template <typename VectorType, typename Operator, int dim>
SSPRungeKuttaIntegrator<VectorType,Operator, dim>::SSPRungeKuttaIntegrator(
  const RungeKuttaScheme scheme)
{
  switch(scheme)
  {
    case forward_euler:
    {
      factor={1};
      ci={0};
      break;
    }
    case ssp_stage_2_order_2:
    {
      factor={1,1./2};
      ci={0,1};
      break;
    }
    case ssp_stage_3_order_3:
    {
      factor={1,1./4,2./3};
      ci={0,1,1./2};
      break;
    }
    case lsrk_stage_3_order_3:
    case lsrk_stage_5_order_4:
    case lsrk_stage_7_order_4:
    case lsrk_stage_9_order_5:
    {
      AssertThrow(false,
        ExcMessage("You are using a wrong Runge Kutta scheme for this"
          " constructor"));
      break;
    }
    default:
      AssertThrow(false, ExcNotImplemented());
  }
}

// This is my version of perform_time_step. I have a different structure of 
// Butcher tableau (do not have the assumption on aij and bi that characterizes
// lsrk schemese introduced in step 67). I write a function that works on all
// SSP Runge Kutta, that have a different property, namely the fact that
// (indexes start from 1 for a and b, start from 0 for vector):
// - for SSP22:
//    - b[1]/a[2][1] = factor[1]
// - for SSP33:
//    - a[3][1] * a[2][1] = a[3][2] = factor[1]
//    - b[2]/a[3][2] = b[1]/(a[3][2] * a[2][1]) = b[3] = factor [2]
// I still use the two latter arguments, but one is to store the solution at
// previous time step, and not vec_ri, therefore it is copy initialized from
// solution every time step
template <typename VectorType, typename Operator, int dim>
void SSPRungeKuttaIntegrator<VectorType,Operator, dim>::perform_time_step(
  const Operator &pde_operator,
  const double current_time,
  const double time_step,
  VectorType & solution,
  VectorType & copy_solution,
  VectorType & vec_ki,
  // These three are needed for the limiter
  const DoFHandler<dim> & dof_handler,
  const MappingQ1<dim> & mapping,
  const FESystem<dim> & fe) const
{
  copy_solution.reinit(solution);
  copy_solution=solution;
  vec_ki.reinit(solution);

  unsigned int n_stages = factor.size();
  for(unsigned int stage = 0; stage<n_stages; ++stage)
  {
    pde_operator.apply(current_time + ci[stage]*time_step, solution, vec_ki);
    solution *= factor[stage];
    solution.add(factor[stage]*time_step, vec_ki);
    solution.add(1-factor[stage], copy_solution);
    // Flux limiter
    pde_operator.apply_positivity_limiter(solution, dof_handler, mapping, fe);
  }
  
}

template <typename VectorType, typename Operator, int dim>
unsigned int
SSPRungeKuttaIntegrator<VectorType,Operator, dim>::n_stages() const
{
  return factor.size();
}

#endif