#ifndef RUNGE_KUTTA_INTEGRATOR
#define RUNGE_KUTTA_INTEGRATOR
#include"TypesDefinition.h"

class LSRungeKuttaIntegrator
{
  public:
    LSRungeKuttaIntegrator(const RungeKuttaScheme scheme);

    unsigned int n_stages() const;

    template<typename VectorType, typename Operator>
    void perform_time_step(const Operator &pde_operator,
      const double    current_time,
      const double    time_step,
      VectorType &    solution,
      VectorType &    vec_ri,
      VectorType &    vec_ki) const;

  private:
    std::vector<double> bi;
    std::vector<double> ai;
    std::vector<double> ci;
};

class SSPRungeKuttaIntegrator
{
  public:
    SSPRungeKuttaIntegrator(const RungeKuttaScheme scheme);
  
  template<typename VectorType, typename Operator>
  void perform_time_step(const Operator &pde_operator,
    const double current_time,
    const double time_step,
    VectorType & solution,
    VectorType & vec_ri,
    VectorType & vec_ki) const;

  unsigned int n_stages() const;
  
  private:
    std::vector<double> bi;
    // aij will be assumed as the concatenation of the rows of the lower
    // triangular part of the full matrix (diagonal excluded). Therefore
    // a[i][j] = aij[i*(i-1)/2 + j]
    // MATRIX INDICES START FROM 0
    std::vector<double> aij;
    std::vector<double> ci;
};
#endif