#ifndef EULERIAN_SPRAY_INLINED_FUNCTIONS
#define EULERIAN_SPRAY_INLINED_FUNCTIONS

#include<deal.II/base/tensor.h>

// This file contains some useful inlined functions used in the computations
// made by EulerianSprayOperator 

// This function returns the velocity $\underline{v}$ from the vector of
// conserved quantities $\undeline{w}=[ \rho, \rho u_1, ..., \rho u_d]
template< int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE
Tensor<1, dim, Number> eulerian_spray_velocity(
    const Tensor<1, dim + 1, Number> & conserved_variables){

    const Number inverse_density = Number(1.) / conserved_variables[0];

    Tensor<1, dim, Number> velocity;
    for(unsigned int d = 0; d < dim; d++)
        velocity[d] = conserved_variables[1+d] * inverse_density;
    
    return velocity;
}

// This template function returns the Eulerian spray flux (in analogy to Euler
// flux from step 67)
// TODO: why I use a return type a tensor of order 1 of tensors of order 1
// and not a tensor of order two?
template <int dim, typename Number>
// TODO: understand deepely why I use DEAL_II_ALWAYS_INLINE
inline DEAL_II_ALWAYS_INLINE
Tensor<1, dim + 1, Tensor<1, dim, Number>>
eulerian_spray_flux(const Tensor<1, dim+1, Number> & conserved_variables){
    const Tensor<1, dim, Number> velocity =
        eulerian_spray_velocity<dim>(conserved_variables);

    Tensor<1, dim + 1, Tensor<1, dim, Number>> flux;
    for(unsigned int d = 0; d < dim; ++d){
        // The first row is the vector of momenti in the dim directions
        flux[0][d] = conserved_variables[1 + d];
        for( unsigned int e = 0; e < dim; ++e)
            flux[e+1][d] = conserved_variables[e+1] * velocity[d];
    }
    return flux;
}





#endif