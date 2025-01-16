//

#include<array>
#include"RungeKuttaIntegrator.h"
RungeKuttaIntegrator::RungeKuttaIntegrator(const RungeKuttaScheme scheme)
{
  switch(scheme)
  {
    // For the moment I use just Forward Euler
    case(stage_1):
    {
      aij={0};
      bi={1};
      ci={0};      
      break;
    }
    default:
      AssertThrow(false, ExcNotImplemented());
  }  
}

unsigned int
RungeKuttaIntegrator::n_stages() const
{
  return bi.size();
}

// The following implementation may be very general to all explicit Runge Kutta
// schemes, but can be also very expansive if the method is a low storage one,
// since in this implementation I mantain al Ki up to the computation of the new
// solution
template <typename VectorType, typename Operator>
void RungeKuttaIntegrator::perform_time_step(const Operator & pde_operator,
  const double current_time,
  const double time_step,
  VectorType & solution,
  VectorType & previous) const
{
  no_stages = this->n_stages();
  std::array<VectorType, no_stages + 1> Ki;
  pde_operator.perform_stage()
  Ki[0] = pde_operator
  for(unsigned int stage = 1; stage =< this->n_stages(); stage ++)
  {

  }



}