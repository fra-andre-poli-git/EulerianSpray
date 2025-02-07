#ifndef EULERIAN_SPRAY_INITIAL_SOLUTION_HH
#define EULERIAN_SPRAY_INITIAL_SOLUTION_HH

#include"Parameters.h"
#include<deal.II/base/function.h>


using namespace dealii;

template<int dim>
class InitialSolution : public Function<dim>
{
  public:
    InitialSolution(const Parameters & params): Function<dim>(dim + 1),
      parameters(params){};
    virtual double value(const Point<dim> & p, const unsigned int component) 
      const override;
  private:
    Parameters parameters;
};

template<int dim>
class FinalSolution : public Function<dim>
{
  public:
    FinalSolution (const Parameters & params): Function<dim>(dim + 1),
      parameters(params){};
    virtual double value(const Point<dim> & p, const unsigned int component)
      const override;
  private:
    Parameters parameters;
};

template<int dim>
class DirichletFunction : public Function<dim>
{
  public:
    DirichletFunction(const double time, const Parameters & params):
      Function<dim>(dim + 1, time), parameters(params){};
    virtual double value(const Point<dim> & p, const unsigned int component)
      const override;
  private:
    Parameters parameters;
};

template<int dim>
class ExactSolution : public Function<dim>
{
  public:
    ExactSolution(const double time, const Parameters & params):
      Function<dim>(dim, time), parameters(params){};
    virtual double value(const Point<dim> & p, const unsigned int component)
      const override;
  private:
    Parameters parameters;
};

// template<int dim>
// class ExternalFlux : public Function<dim>
// {
//   public:
//     ExternalFlux(const double time):Function<dim>(dim, time){};
//     virtual double value(const Point<dim> & p, const unsigned int component)
//       const override;
// };
#endif