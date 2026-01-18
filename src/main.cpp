/*
	Advanced Programming for Scientific Computing project and thesis work of 
  Francesco Andreotti, PoliMi
	The reference for this program is Step-67 of deal.ii tutorials.
*/
#include<iostream>
#include"EulerianSprayProblem.hpp"
#include"Parameters.hpp"

#include<deal.II/base/parameter_handler.h>

// For the moment dimension will be a global const, 
// It must be 2, since the functions of the output are not meant to be used in
// dimension 1.
constexpr unsigned int dimension = 2;
constexpr unsigned int finite_element_degree = 2;

void test_numerical_flux();

int main(int argc, char ** argv){
	try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 8);
    deallog.depth_console(0);
    ParameterHandler prm;
    Parameters::declare_parameters(prm);
    prm.parse_input(argv[1]);
    Parameters parameters;
    parameters.parse_parameters(prm);
    // TODO verbose_cout
    EulerianSprayProblem<dimension,finite_element_degree>
      eulerian_spray_problem(parameters);
    eulerian_spray_problem.run();
    // test_numerical_flux();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
	return 0;
}


// This function is created to test the numerical flux
void test_numerical_flux() {
    // Stati sinistro e destro semplici
    double rho_m = 1.0, u_m = 0.1, v_m = 0. ;
    double rho_p = 1.0, u_p = -0.1, v_p = 0.;

    std::cout<< "rho_m = " << rho_m <<" u_m = " << u_m << " v_m " << v_m << std::endl;
    std::cout<< "rho_p = " << rho_p <<" u_p = " << u_p << " v_p " << v_p << std::endl;

    
    Tensor<1,dimension + 1> w_m, w_p;
    w_m[0] = rho_m; w_m[1] = rho_m*u_m; w_m[2] = rho_m*v_m;
    w_p[0] = rho_p; w_p[1] = rho_p*u_p; w_p[2] = rho_p*v_p;

    std::cout<< "w_m = [" << w_m[0] <<" , "<< w_m[1] <<" , "<< w_m[2] <<"]"<<std::endl;
    std::cout<< "w_p = [" << w_p[0] <<" , "<< w_p[1] <<" , "<< w_p[2] <<"]"<<std::endl;
    
    Tensor<1,2> normal({-1.0, 0.0});

    std::cout<< "normal = [" << normal[0] << ", " << normal[1] << "]"<<std::endl;
    
    // Calcola flusso con entrambi i metodi
    // auto flux_godunov = eulerian_spray_numerical_flux(w_m, w_p, normal, godunov);
    // auto flux_llf = eulerian_spray_numerical_flux(w_m, w_p, normal, local_lax_friedrichs);

    auto flux_godunov = eulerian_spray_numerical_flux(w_p, w_m, normal, godunov);
    auto flux_llf = eulerian_spray_numerical_flux(w_p, w_m, normal, local_lax_friedrichs);
    
    // Stampa e verifica manualmente
    std::cout << "Godunov flux: " << flux_godunov << std::endl;
    std::cout << "LLF flux: " << flux_llf << std::endl;
    
    // Per questo caso simmetrico, il flusso dovrebbe essere quello
    // con u_interface ≈ 0
}