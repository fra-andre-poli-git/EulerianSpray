#ifndef RUNGE_KUTTA_INTEGRATOR
#define RUNGE_KUTTA_INTEGRATOR
#include"TypesDefinition.hpp"

#include<deal.II/fe/fe_system.h> 
#include<deal.II/fe/mapping_q1.h>
#include<deal.II/dofs/dof_handler.h>

template<typename VectorType, typename Operator, int dim>
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
      VectorType &    vec_ki,
      // These three are needed for the limiter
      const DoFHandler<dim> & dof_handler,
      const MappingQ1<dim> & mapping,
      const FESystem<dim> & fe) const = 0 ;
    virtual ~RungeKuttaIntegrator() = default;  
};
template<typename VectorType, typename Operator, int dim>
class LSRungeKuttaIntegrator : public RungeKuttaIntegrator<VectorType,Operator,dim>
{
  public:
    LSRungeKuttaIntegrator(const RungeKuttaScheme scheme);

    unsigned int n_stages() const override;

    void perform_time_step(const Operator &pde_operator,
      const double    current_time,
      const double    time_step,
      VectorType &    solution,
      VectorType &    vec_ri,
      VectorType &    vec_ki,
      // These three are needed for the limiter
      const DoFHandler<dim> & dof_handler,
      const MappingQ1<dim> & mapping,
      const FESystem<dim> & fe) const override;

  private:
    std::vector<double> bi;
    std::vector<double> ai;
    std::vector<double> ci;
};

template<typename VectorType, typename Operator, int dim>
class SSPRungeKuttaIntegrator : public RungeKuttaIntegrator<VectorType,Operator, dim>
{
  public:
    SSPRungeKuttaIntegrator(const RungeKuttaScheme scheme);
  
  void perform_time_step(const Operator &pde_operator,
    const double current_time,
    const double time_step,
    VectorType & solution,
    VectorType & vec_ri,
    VectorType & vec_ki,
    // These three are needed for the limiter
    const DoFHandler<dim> & dof_handler,
    const MappingQ1<dim> & mapping,
    const FESystem<dim> & fe) const override;

  unsigned int n_stages() const override;
  
  private:
    // Instead of storing the whole matrix we store directly the vector of
    // factors, which are multiplication/division of original Butcher tableau
    // aijs and bis -> for more info check in the implementation of
    // perform_time_step
    std::vector<double> factor;
    std::vector<double> ci;
};

#include"RungeKuttaIntegrator_IMP.hpp"
#endif