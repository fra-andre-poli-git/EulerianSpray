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
      VectorType & previous) const;

  private:
    std::vector<double> bi;
    // TODO: modify the following rule to exclude the central diagonal, since I
    // am not interested in semi-explicit Runge Kutta methods
    // aij will be assumed as the concatenation of the rows of the lower
    // triangular part of the full matrix. Therefore a[i][j] =
    // aij[i*(i+1)/2 + j]
    std::vector<double> aij;
    std::vector<double> ci;
};


#endif