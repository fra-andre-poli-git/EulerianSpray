//
#include"RungeKuttaIntegrator.h"
#include"TypesDefinition.h"
#include"EulerianSprayOperator.h"
#include<deal.II/base/time_stepping.h>
LSRungeKuttaIntegrator::LSRungeKuttaIntegrator(
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

unsigned int
LSRungeKuttaIntegrator::n_stages() const
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
void LSRungeKuttaIntegrator::perform_time_step(const Operator &pde_operator,
  const double    current_time,
  const double    time_step,
  VectorType &    solution,
  VectorType &    vec_ri,
  VectorType &    vec_ki) const
{
  AssertDimension(ai.size() + 1, bi.size());

  pde_operator.perform_lsrk_stage(current_time,
                              bi[0] * time_step,
                              ai[0] * time_step,
                              solution,
                              vec_ri,
                              solution,
                              vec_ri);

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
    }
}


SSPRungeKuttaIntegrator::SSPRungeKuttaIntegrator(const RungeKuttaScheme scheme)
{
  switch(scheme)
  {
    case forward_euler:
    {
      bi={1};
      aij={};
      ci={0};
      break;
    }
    case ssp_stage_2_order_2:
    {
      bi={1./2,1./2};
      aij={1};
      ci={0,1};
      break;
    }
    case ssp_stage_3_order_3:
    {
      bi={1./6,1./6,2./3};
      aij={1,1./4,1./4};
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
// general three step Runge Kutta. The implementation with nested if may not be
// the most elegant one, but for now I will keep this 
template <typename VectorType, typename Operator>
void SSPRungeKuttaIntegrator::perform_time_step(const Operator &pde_operator,
  const double    current_time,
  const double    time_step,
  VectorType &    solution,
  VectorType &    vec_ri,
  VectorType &    vec_ki) const
{
  // This is the classical implementation

  // (void) vec_ri;
  // (void) vec_ki;
  // unsigned int n_stages = bi.size();
  // stage 1
  // VectorType k1;
  // k1.reinit(solution);
  // pde_operator.apply(current_time+ci[0]*time_step,solution,k1);
  // if(n_stages == 1)
  // {
  //   //u^{n+1}=u^n + b_1 k_1
  //   solution.add(bi[0]*time_step, k1);
  // }
  // else
  // {
  //   //stage 2
  //   VectorType k2;
  //   k2.reinit(solution);
  //   // Here I will define a temp vector that will be used to store
  //   // u^n + \sum_{i=1}^{stage-1} \Delta t a[stage][i] ki
  //   VectorType tmp;
  //   tmp.reinit(solution);
  //   tmp=solution;
  //   tmp.add(aij[0]*time_step,k1);
  //   pde_operator.apply(current_time+ci[1]*time_step, tmp, k2);
  //   if(n_stages == 2)
  //   {
  //     //u^{n+1}=u^n + b_1 k_1 + b_2 k_2
  //     solution.add(bi[0]*time_step,k1);
  //     solution.add(bi[1]*time_step,k2);
  //   }
  //   else
  //   {
  //     //stage 3
  //     VectorType k3;
  //     k3.reinit(solution);
  //     tmp=solution;
  //     tmp.add(aij[1]*time_step, k1);
  //     tmp.add(aij[2]*time_step, k2);
  //     pde_operator.apply(current_time + ci[2]*time_step, tmp, k3);
  //     //u^{n+1}=u^n + b1 k1 + b2 k2 + b3 k3
  //     solution.add(bi[0]*time_step,k1);
  //     solution.add(bi[1]*time_step,k2);
  //     solution.add(bi[2]*time_step,k3);
  //   }
  // }

  // This is the implementation to comply with filtering technique

  (void) vec_ri;
  (void) vec_ki;
  unsigned int n_stages = bi.size();

  VectorType copy_solution;
  copy_solution.reinit(solution);
  copy_solution = solution;
  VectorType k;
  k.reinit(solution);

  // stage 1
  pde_operator.apply(current_time + ci[0] * time_step, solution, k);
  if(n_stages == 1)
  {
    solution.add(bi[0]*time_step, k);
    // filter solution
  }
  else
  {
    solution.add(aij[0]*time_step, k);
    // filter solution
    //stage 2
    pde_operator.apply(current_time + ci[1] * time_step, solution, k);
    if(n_stages == 2)
    {
      double factor = bi[0]/aij[0];
      solution*=factor;
      solution.add(1-factor, copy_solution);
      solution.add(bi[1]*time_step, k);
      // filter solution
    }
    else
    {
      double factor = aij[1]*aij[0];
      solution *=factor;
      solution.add(1-factor, copy_solution);
      solution.add(aij[2]*time_step, k);
      // filter solution
      pde_operator.apply(current_time + ci[2]*time_step, solution, k);
      factor = bi[1]/aij[2]; // =bi[0]/(aij[0]*aij[1])
      solution *= factor;
      solution.add(1-factor, copy_solution);
      solution.add(bi[2]*time_step,k);
      // filter solution
    }
  }
}

unsigned int
SSPRungeKuttaIntegrator::n_stages() const
{
  return bi.size();
}


//Instantiations of the template function
template void LSRungeKuttaIntegrator::perform_time_step<SolutionType,
  EulerianSprayOperator<2,0,2>>(const EulerianSprayOperator<2,0,2> &,
    const double,
    const double,
    SolutionType &,
    SolutionType &,
    SolutionType &) const;

template void LSRungeKuttaIntegrator::perform_time_step<SolutionType,
  EulerianSprayOperator<2,1,3>>(const EulerianSprayOperator<2,1,3> &,
    const double,
    const double,
    SolutionType &,
    SolutionType &,
    SolutionType &) const;


template void LSRungeKuttaIntegrator::perform_time_step<SolutionType,
  EulerianSprayOperator<2,2,4>>(const EulerianSprayOperator<2,2,4> &,
    const double,
    const double,
    SolutionType &,
    SolutionType &,
    SolutionType &) const;

template void SSPRungeKuttaIntegrator::perform_time_step<SolutionType,
  EulerianSprayOperator<2,0,2>>(const EulerianSprayOperator<2,0,2> &,
    const double,
    const double,
    SolutionType &,
    SolutionType &,
    SolutionType &) const;

template void SSPRungeKuttaIntegrator::perform_time_step<SolutionType,
  EulerianSprayOperator<2,1,3>>(const EulerianSprayOperator<2,1,3> &,
    const double,
    const double,
    SolutionType &,
    SolutionType &,
    SolutionType &) const;

template void SSPRungeKuttaIntegrator::perform_time_step<SolutionType,
  EulerianSprayOperator<2,2,4>>(const EulerianSprayOperator<2,2,4> &,
    const double,
    const double,
    SolutionType &,
    SolutionType &,
    SolutionType &) const;