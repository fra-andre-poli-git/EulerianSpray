#include<deal.II/base/function.h>

using namespace dealii;

template<int dim, int degree, int n_points_1d>
class EulerianSprayOperator{
    public:
        void project(const Function<dim> & function,
                     Vector<Number>)
}