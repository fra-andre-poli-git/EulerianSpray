/*
	Advanced Programming for Scientific Computing project and thesis work of 
  Francesco Andreotti, PoliMi
	The reference for this program is Step-67 of deal.ii tutorials.
*/
#include<iostream>
#include"EulerianSprayProblem.h"

// For the moment dimension will be a global const, 
// It must be 2, since the functions of the output are not meant to be used in
// dimension 1.
constexpr unsigned int dimension=2;




int main(int argc, char ** argv){
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 8);
  
	try{
      deallog.depth_console(0);

      EulerianSprayProblem<dimension> eulerian_spray_problem;
      eulerian_spray_problem.run();
    }
  	catch (std::exception &exc){
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
  	catch (...){
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
