#ifndef EULERIAN_SPRAY_INITIAL_SOLUTION_HH
#define EULERIAN_SPRAY_INITIAL_SOLUTION_HH

#include<deal.II/base/function.h>

using namespace dealii;

template<int dim>
class InitialSolution : public Function<dim>{
    public:
        virtual double value(const Point<dim> & p, const unsigned int component) const override;
};

#endif