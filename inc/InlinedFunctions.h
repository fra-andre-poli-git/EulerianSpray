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
  const Tensor<1, dim + 1, Number> & conserved_variables)
{
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
// TODO: understand why I use DEAL_II_ALWAYS_INLINE
inline DEAL_II_ALWAYS_INLINE
Tensor<1, dim + 1, Tensor<1, dim, Number>>
  eulerian_spray_flux(const Tensor<1, dim+1, Number> & conserved_variables)
{
  const Tensor<1, dim, Number> velocity =
    eulerian_spray_velocity<dim>(conserved_variables);

  Tensor<1, dim + 1, Tensor<1, dim, Number>> flux;
  for(unsigned int d = 0; d < dim; ++d)
  {
    // The first row is the vector of momenti in the dim directions
    flux[0][d] = conserved_variables[1 + d];
    for( unsigned int e = 0; e < dim; ++e)
      flux[e+1][d] = conserved_variables[e+1] * velocity[d];
  }
  return flux;
}

template <int n_components, int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE
Tensor<1, n_components, Number>
operator * ( const Tensor<1, n_components, Tensor<1, dim, Number>> & matrix,
  const Tensor<1, dim, Number> & vector)
{
  Tensor<1, n_components, Number> result;
  for(unsigned int d = 0; d < n_components; ++d)
    result[d] = matrix[d] * vector;
  return result;
}

// This function returns the numerical flux already multiplied by the normal
// vecotr. I am using local Lax-Friedrichs, but the structure can accept other
// flux definitions through the switch.
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE
Tensor<1, dim + 1, Number>
eulerian_spray_numerical_flux(const Tensor<1, dim + 1, Number> & w_minus,
  const Tensor<1, dim + 1, Number> & w_plus,
  const Tensor<1, dim, Number> & normal,
  const NumericalFlux numerical_flux_type)
{
  const auto velocity_minus = eulerian_spray_velocity<dim>(w_minus);
  const auto velocity_plus = eulerian_spray_velocity<dim>(w_plus);

  const auto flux_minus = eulerian_spray_flux<dim>(w_minus);
  const auto flux_plus = eulerian_spray_flux<dim>(w_plus);

  switch (numerical_flux_type)
  {
    case local_lax_friedrichs:
    {
      // auto v_p_times_n = static_cast<Number>(velocity_plus * normal);
      // auto v_m_times_n = static_cast<Number>(velocity_minus * normal);
      // const auto delta = std::max(v_p_times_n.norm() ,
      //   std::abs(v_m_times_n));
      const auto delta = std::max(velocity_plus.norm() , velocity_minus.norm());
      return 0.5 * (flux_minus * normal + flux_plus * normal) +
        0.5 * delta * (w_minus - w_plus);
    }
    case godunov:
    {
      
    }
    default:{
      Assert(false, ExcNotImplemented());
      return{};
    }
  }
}





#endif