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
    // Instead of storing the whole matrix we store directly the vector of
    // factors, which are multiplication/division of original Butcher tableau
    // aijs and bis
    std::vector<double> factor;
    std::vector<double> ci;
};
#endif