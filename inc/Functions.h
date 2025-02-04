#ifndef EULERIAN_SPRAY_INITIAL_SOLUTION_HH
#define EULERIAN_SPRAY_INITIAL_SOLUTION_HH

#include<deal.II/base/function.h>

using namespace dealii;

template<int dim>
class InitialSolution : public Function<dim>
{
  public:
    InitialSolution(): Function<dim>(dim + 1){};
    virtual double value(const Point<dim> & p, const unsigned int component) 
      const override;
};

template<int dim>
class FinalSolution : public Function<dim>
{
  public:
    FinalSolution (): Function<dim>(dim + 1){};
    virtual double value(const Point<dim> & p, const unsigned int component)
      const override;
};

template<int dim>
class DirichletFunction : public Function<dim>
{
  public:
    DirichletFunction(const double time):Function<dim>(dim + 1, time){};
    virtual double value(const Point<dim> & p, const unsigned int component)
      const override;
};

template<int dim>
class ExactSolution : public Function<dim>
{
  public:
    ExactSolution(const double time):Function<dim>(dim, time){};
    virtual double value(const Point<dim> & p, const unsigned int component)
      const override;
};

template<int dim>
class ExternalFlux : public Function<dim>
{
  public:
    ExternalFlux(const double time):Function<dim>(dim, time){};
    virtual double value(const Point<dim> & p, const unsigned int component)
      const override;
};
#endif