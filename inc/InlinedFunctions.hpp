#ifndef EULERIAN_SPRAY_INLINED_FUNCTIONS
#define EULERIAN_SPRAY_INLINED_FUNCTIONS

#include<deal.II/base/tensor.h>

// This file contains some useful inlined functions used in the computations
// made by EulerianSprayOperator to compute some quantities.
// TODO: I should opt for a more meaningful name for this file

// This function returns the velocity $\underline{v}$ from the vector of
// conserved quantities $\undeline{w}=[ \rho, \rho u_1, ..., \rho u_d]
template< int dim, typename myReal>
inline DEAL_II_ALWAYS_INLINE
Tensor<1, dim, myReal> eulerian_spray_velocity(
  const Tensor<1, dim + 1, myReal> & conserved_variables)
{
  const myReal inverse_density = myReal(1.) / conserved_variables[0];

  Tensor<1, dim, myReal> velocity;
  for(unsigned int d = 0; d < dim; d++)
    velocity[d] = conserved_variables[1+d] * inverse_density;
  
  return velocity;
}

// This template function returns the Eulerian spray flux 
template <int dim, typename myReal>
inline DEAL_II_ALWAYS_INLINE
Tensor<1, dim + 1, Tensor<1, dim, myReal>>
  eulerian_spray_flux(const Tensor<1, dim+1, myReal> & conserved_variables)
{
  const Tensor<1, dim, myReal> velocity =
    eulerian_spray_velocity<dim>(conserved_variables);

  Tensor<1, dim + 1, Tensor<1, dim, myReal>> flux;
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
template <int n_components, int dim, typename myReal>
inline DEAL_II_ALWAYS_INLINE
Tensor<1, n_components, myReal>
operator * ( const Tensor<1, n_components, Tensor<1, dim, myReal>> & matrix,
  const Tensor<1, dim, myReal> & vector)
{
  Tensor<1, n_components, myReal> result;
  for(unsigned int d = 0; d < n_components; ++d)
    result[d] = matrix[d] * vector;
  return result;
}

// This function returns the numerical flux already multiplied by the normal
// vector. "normal" refers to the outward one. I started using local
// Lax-Friedrichs, but the structure can accept other flux definitions
// through the switch.
template <int dim, typename myReal>
inline DEAL_II_ALWAYS_INLINE
Tensor<1, dim + 1, myReal>
eulerian_spray_numerical_flux(const Tensor<1, dim + 1, myReal> & w_minus,
  const Tensor<1, dim + 1, myReal> & w_plus,
  const Tensor<1, dim, myReal> & normal,
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
      // According to ... Forcella? TODO add reference
      auto v_p_times_n = (velocity_plus * normal);
      auto v_m_times_n = (velocity_minus * normal);
      const auto delta = std::max(abs(v_p_times_n) , abs(v_m_times_n));

      // This is according to step 67
      // const auto delta = std::max(velocity_plus.norm(), velocity_minus.norm());

      return 0.5 * (flux_minus * normal + flux_plus * normal) +
        0.5 * delta * (w_minus - w_plus);
    }
    case godunov:
    {
      // Taken from Yang, Wei, Shu, 2013 and Bouchou, Jin, Li, ????
      const myReal density_minus = w_minus[0];
      const myReal density_plus = w_plus[0];

      const auto normal_velocity_minus = (velocity_minus * normal);
      const auto normal_velocity_plus = (velocity_plus * normal);

      // Now, one may think, like me, that normal_velocity is a myReal,
      // therefore a double. Well, THEY ARE WRONG, because, thanks to the macro 
      // DEAL_II_ALWAYS_INLINE, it is a dealii::VectorizedArray<double, 2>
      // Vectorization should speed up the performance, so I will keep it,
      // therefore now I have to deal with it

      // unsigned int vectorization_dimension = normal_velocity_minus.size();
      
      constexpr unsigned int vectorization_dimension = VectorizedArray<myReal>::size();

      Tensor<1, dim + 1, myReal> flux;
    
      for(unsigned int v=0; v<vectorization_dimension; ++v)
      {
        if(normal_velocity_minus[v] > 0 && normal_velocity_plus[v] > 0)
        {
          auto normal_flux = flux_minus*normal;
          for(unsigned int d=0; d<dim+1; ++d)
            flux[d][v]= normal_flux[d][v];
        }
        else if(normal_velocity_minus[v] <= 0 && normal_velocity_plus[v] <=0)
        {
          auto normal_flux = flux_plus*normal;
          for(unsigned int d=0; d<dim+1; ++d)
            flux[d][v]= normal_flux[d][v];
        }
        else if(normal_velocity_minus[v] <= 0 && normal_velocity_plus[v] > 0)
        {
          for(unsigned int d=0; d<dim+1; ++d)
            flux[d][v]= 0.0;
        }
        else if(normal_velocity_minus[v] > 0 && normal_velocity_plus[v] <= 0)
        {
          // I compute u_delta here, since it is needed only in this case.
          // In this way, however, I may compute it more than once for each
          // vectorization dimension. TODO: check if this is a problem
          // performance-wise.
          const myReal rho_m_sqrt = std::sqrt(density_minus);
          const myReal rho_p_sqrt = std::sqrt(density_plus);
          const myReal u_delta = ((rho_m_sqrt * normal_velocity_minus +
            rho_p_sqrt * normal_velocity_plus)/(rho_m_sqrt + rho_p_sqrt));
          if(u_delta[v] >= 0)
          {
            auto normal_flux = flux_minus*normal;
            for(unsigned int d=0; d<dim+1; ++d)
              flux[d][v]= normal_flux[d][v];
          }
          else if(u_delta[v] < 0)
          {
            auto normal_flux = flux_plus*normal;
            for(unsigned int d=0; d<dim+1; ++d)
              flux[d][v]= normal_flux[d][v];
          }
          else
          {
            auto normal_flux = 0.5 * (flux_minus * normal + flux_plus * normal);
            for(unsigned int d=0; d<dim+1; ++d)
              flux[d][v]= normal_flux[d][v];
          }
        }
      }
      return flux;
    }
    case local_lax_friedrichs_modified:
    {
      // This is the modified version from tutorial
      const auto delta  = 0.5 * std::sqrt(std::max(velocity_plus.norm_square(),
        velocity_minus.norm_square()));

      return 0.5 * (flux_minus * normal + flux_plus * normal) +
        0.5 * delta * (w_minus - w_plus);
    }
    case harten_lax_vanleer:
    {
      const auto avg_velocity_normal = 
        0.5 * ((velocity_minus + velocity_plus) * normal);
      const myReal s_pos = std::max(myReal(), avg_velocity_normal);
      const myReal s_neg = std::min(myReal(), avg_velocity_normal);
      const myReal inverse_s = myReal(1.) / (s_pos - s_neg);

      return inverse_s *
        ((s_pos * (flux_minus * normal) - s_neg * (flux_plus * normal)) -
        s_pos * s_neg * (w_minus - w_plus));
    }
    default:
    {
      Assert(false, ExcNotImplemented());
      return{};
    }
  }
}

#endif