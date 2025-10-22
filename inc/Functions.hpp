#ifndef EULERIAN_SPRAY_INITIAL_SOLUTION_HH
#define EULERIAN_SPRAY_INITIAL_SOLUTION_HH

#include"Parameters.hpp"

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

// I create this FinalSolutionVelocity that expresses the solution in term of
// the velocity, which is useful for the plot since it's impossible to retrieve
// the velocity when the density is null (or it may be error prone when it's
// small).
template<int dim>
class FinalSolutionVelocity : public Function<dim>
{
  public:
    FinalSolutionVelocity (const Parameters & params): Function<dim>(dim + 1),
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

#include"Functions_IMP.hpp"

#endif