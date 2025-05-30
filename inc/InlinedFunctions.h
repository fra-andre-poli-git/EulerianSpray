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

// This template function returns the Eulerian spray flux 
template <int dim, typename Number>
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

// This functions implements the matrix-vector where matrix is a
// (n_components x dim) and the vector is (dim x 1)
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
// vector. I am using local Lax-Friedrichs, but the structure can accept other
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
      // These is according to Forcella
      // auto v_p_times_n = static_cast<Number>(velocity_plus * normal);
      // auto v_m_times_n = static_cast<Number>(velocity_minus * normal);
      // const auto delta = std::max(std::abs(v_p_times_n) ,
      //   std::abs(v_m_times_n));

      // This is according to Sabat et al. TODO: I can't find it on Sabat et al, find it
      const auto delta = std::max(velocity_plus.norm() , velocity_minus.norm());
      return 0.5 * (flux_minus * normal + flux_plus * normal) +
        0.5 * delta * (w_minus - w_plus);
    }
    case godunov:
    {
      // Taken from Bouchout Jin Li
      const Number density_minus = w_minus[0];
      const Number density_plus = w_plus[0];
      const Number rho_m_sqrt = std::sqrt(density_minus);
      const Number rho_p_sqrt = std::sqrt(density_plus);
      const Number u_delta = ((rho_m_sqrt * (velocity_minus * normal) +
        rho_p_sqrt * (velocity_plus * normal))/(rho_m_sqrt + rho_p_sqrt));

      // Now, one may think, like me, that u_delta is a Number, therefore a
      // double. Well, THEY ARE WRONG, because, thanks to the macro 
      // DEAL_II_ALWAYS_INLINE, it is a dealii::VectorizedArray<double, 2>
      // Vectorization should speed up the performance, so I will keep it,
      // therefore now I have to deal with it

      Tensor<1, dim + 1, Number> flux;

      unsigned int vectorization_dimension = u_delta.size();

      for(unsigned int v=0; v<vectorization_dimension; ++v)
      {
        if(u_delta[v]>1e-16)
        {//flux_minus*normal
          auto normal_flux = flux_minus*normal;
          for(unsigned int d=0; d<dim+1; ++d)
          {
            flux[d][v]=normal_flux[d][v];
          }
        }
        else if(-u_delta[v]>1e-16)
        {//flux_plus*normal
          auto normal_flux = flux_plus*normal;
          for(unsigned int d=0; d<dim+1; ++d)
          {
            flux[d][v]=normal_flux[d][v];
          }
        }
        else
        {//0.5*(flux_plus * normal + flux_minus * normal)
          auto normal_flux = 0.5*(flux_plus * normal + flux_minus * normal);
          for(unsigned int d=0; d<dim+1; ++d)
          {
            flux[d][v]=normal_flux[d][v];
          }
        }
      }
      return flux;
    }
    default:
    {
      Assert(false, ExcNotImplemented());
      return{};
    }
  }
}

#endif