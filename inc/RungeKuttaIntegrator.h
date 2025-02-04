#ifndef RUNGE_KUTTA_INTEGRATOR
#define RUNGE_KUTTA_INTEGRATOR
#include"TypesDefinition.h"

template<typename VectorType, typename Operator>
class RungeKuttaIntegrator
{
  public:
    virtual unsigned int n_stages() const = 0;
    virtual void perform_time_step(
      const Operator &pde_operator,
      const double    current_time,
      const double    time_step,
      VectorType &    solution,
      VectorType &    vec_ri,
      VectorType &    vec_ki
    ) const = 0 ;
    virtual ~RungeKuttaIntegrator() = default;  
};
template<typename VectorType, typename Operator>
class LSRungeKuttaIntegrator : public RungeKuttaIntegrator<VectorType,Operator>
{
  public:
    LSRungeKuttaIntegrator(const RungeKuttaScheme scheme);

    unsigned int n_stages() const override;

    void perform_time_step(const Operator &pde_operator,
      const double    current_time,
      const double    time_step,
      VectorType &    solution,
      VectorType &    vec_ri,
      VectorType &    vec_ki) const override;

  private:
    std::vector<double> bi;
    std::vector<double> ai;
    std::vector<double> ci;
};

template<typename VectorType, typename Operator>
class SSPRungeKuttaIntegrator : public RungeKuttaIntegrator<VectorType,Operator>
{
  public:
    SSPRungeKuttaIntegrator(const RungeKuttaScheme scheme);
  
  void perform_time_step(const Operator &pde_operator,
    const double current_time,
    const double time_step,
    VectorType & solution,
    VectorType & vec_ri,
    VectorType & vec_ki) const override;

  unsigned int n_stages() const override;
  
  private:
    // Instead of storing the whole matrix we store directly the vector of
    // factors, which are multiplication/division of original Butcher tableau
    // aijs and bis -> for more info check in the implementation of
    // perform_time_step
    std::vector<double> factor;
    std::vector<double> ci;
};
#endif