#ifndef RUNGE_KUTTA_INTEGRATOR
#define RUNGE_KUTTA_INTEGRATOR
#include"TypesDefinition.h"

class RungeKuttaIntegrator
{
  public:
    RungeKuttaIntegrator(const RungeKuttaScheme scheme);

    unsigned int n_stages() const;

    template<typename VectorType, typename Operator>
    void perform_time_step(const Operator & pde_operator,
      const double current_time,
      const double time_step,
      VectorType & solution,
      VectorType & vec_ri,
      VectorType & vec_ki) const;

  private:
  std::vector<double> bi;
  std::vector<double> ai;
  std::vector<double> ci;
};


#endif