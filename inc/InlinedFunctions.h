#ifndef EULERIAN_SPRAY_INLINED FUNCTIONS
#define EULERIAN_SPRAY_INLINED FUNCTIONS

#include<deal.II/base/tensor.h>

// This file contains some useful inlined functions used in the computations
// made by EulerianSprayOperator 

// This function returns the velocity $\underline{v}$ from the vector of
// conserved quantities $\undeline{w}=[ \rho, \rho u_1, ..., \rho u_d]
template< int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE
Tensor<1, dim, Number> euler_velocity(
    const Tensor<1, dim + 1, Number> & conserved_variables){

    const Number inverse_density = Number(1.) / conserved_variables[0];

    Tensor<1, dim, Number> velocity;
    for(unsigned int d = 0; d < dim; d++)
        velocity[d] = conserved_variables[1+d] * inverse_density;
    
    return velocity;
}

// This template function returns the Eulerian spray flux (in analogy to Euler
// flux from step 67) 
template <int dim, typename Number>
// I still have to understand deepely why I use DEAL_II_ALWAYS_INLINE
inline DEAL_II_ALWAYS_INLINE
Tensor<1, dim + 1, Tensor<1, dim, Number>>
eulerian_spray_flux(const Tensor<1, dim+1, Number> & conserved_variables){
    const Tensor<1, dim, Number> velocity =
        euler_velocity<dim>(conserved_variables);

    Tensor<1, dim + 1, T
}





#endif