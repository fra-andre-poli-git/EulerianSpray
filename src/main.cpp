/*
	Advanced Programming for Scientific Computing project and thesis work of Francesco Andreotti, PoliMi
	The reference for this program is Step-67 tutorial, even though I will try to rewrite it on my own
*/
#include<iostream>
#include"EulerianSprayProblem.h"

// For the moment dimension will be a global const, 

constexpr unsigned int dimension=1;



int main(/*int argc, char ** argv*/){
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
