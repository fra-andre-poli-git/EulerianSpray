#ifndef EULERIAN_SPRAY_INITIAL_SOLUTION_HH
#define EULERIAN_SPRAY_INITIAL_SOLUTION_HH

#include<deal.II/base/function.h>

using namespace dealii;

template<int dim>
class InitialSolution : public Function<dim>{
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

// template<int dim>
// class ExternalFlow : public Function<dim>

#endif