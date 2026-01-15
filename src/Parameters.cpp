#include<map>
#include"Parameters.hpp"

void Parameters::declare_parameters(ParameterHandler & prm)
{
  prm.enter_subsection("Problem parameters");
  {
    prm.declare_entry("testcase",
      "1",
      Patterns::Integer(0), // Is it ok even if I want an unsigned?
      "number of case:"
        "- 2: vacuum formation"
        "- 3: moving delta shock"
        "- 4: 2d collapse");

    // prm.declare_entry("finite element degree",
    //   "0",
    //   Patterns::Integer(0),
    //   "Polynomial degree of finite element basis function for the elements");

    prm.declare_entry("number of elements in x direction",
      "400",
      Patterns::Integer(0),
      "number of elements along the x direction: for 1d cases elements along y "
        "direction will be set as a twentieth of the ones along x, whereas in "
        "2d cases they will be set identical as this parameter");

    prm.declare_entry("final time",
      "0.5",
      Patterns::Double(0),
      "Final value of time");

    prm.declare_entry("snapshot instant",
      "0.05",
      Patterns::Double(0),
      "Time between every snapshot of the solution");

    prm.declare_entry("plot everything",
      "false",
      Patterns::Bool(),
      "True if you want to plot every time step");

    prm.declare_entry("CFL",
      "0.1",
      Patterns::Double(0),
      "CFL number");
  }
  prm.leave_subsection();

  prm.enter_subsection("Integrator parameters");
  {
    prm.declare_entry("Runge Kutta scheme",
      "forward_euler",
      Patterns::MultipleSelection("lsrk_stage_5_order_4|"
        "lsrk_stage_7_order_4|"
        "lsrk_stage_3_order_3|"
        "lsrk_stage_9_order_5|"
        "forward_euler|"
        "ssp_stage_2_order_2|"
        "ssp_stage_3_order_3"),
      "Runge Kutta scheme type");
  }
  prm.leave_subsection();

  prm.enter_subsection("Operator parameters");
  {
    prm.declare_entry("numerical flux type",
      "local_lax_friedrichs",
      Patterns::MultipleSelection("local_lax_friedrichs|"
        "harten_lax_vanleer|"
        "godunov"),
      "Type of the numerical flux used between the interfaces of the elements");
  }
  prm.leave_subsection();

  prm.enter_subsection("Limiter parameters");
  {
    prm.declare_entry("limiter",
      "none",
      Patterns::MultipleSelection("none|"
        "bound_preserving"),
      "Type of limiter selected");
    prm.declare_entry("epsilon",
      "1e-13",
      Patterns::Double(),
      "Numerical tolerance for defining the numerical admissibility region");
  }
  prm.leave_subsection();
}


void Parameters::parse_parameters(ParameterHandler & prm)
{
  prm.enter_subsection("Problem parameters");
  {
    testcase = prm.get_integer("testcase");
    // fe_degree = prm.get_integer("finite element degree");
    // n_q_points_1d = fe_degree + 2;
    n_el_x_direction = prm.get_integer("number of elements in x direction");
    final_time = prm.get_double("final time");
    snapshot_instant = prm.get_double("snapshot instant");
    plot_everything = prm.get_bool("plot everything");
    CFL = prm.get_double("CFL");
  }
  prm.leave_subsection();

  prm.enter_subsection("Integrator parameters");
  {

    static const std::map<std::string, RungeKuttaScheme> scheme_map =
    {
      {"lsrk_stage_3_order_3", lsrk_stage_3_order_3},
      {"lsrk_stage_5_order_4", lsrk_stage_5_order_4},
      {"lsrk_stage_7_order_4", lsrk_stage_7_order_4},
      {"lsrk_stage_9_order_5", lsrk_stage_9_order_5},
      {"forward_euler", forward_euler},
      {"ssp_stage_2_order_2", ssp_stage_2_order_2},
      {"ssp_stage_3_order_3", ssp_stage_3_order_3}
    };
    std::string selected_scheme = prm.get("Runge Kutta scheme");

    auto it = scheme_map.find(selected_scheme);
    if(it != scheme_map.end())
      scheme = it->second;
    else
      throw std::runtime_error("Unknown Runge Kutta scheme: " + selected_scheme);
    
    // // The parser gets a bit cumbersome if I have to deal with enum:
    // if (selected_scheme == "lsrk_stage_3_order_3")
    //     scheme = lsrk_stage_3_order_3;
    // else if (selected_scheme == "lsrk_stage_5_order_4")
    //     scheme = lsrk_stage_5_order_4;
    // else if (selected_scheme == "lsrk_stage_7_order_4")
    //     scheme = lsrk_stage_7_order_4;
    // else if (selected_scheme == "lsrk_stage_9_order_5")
    //     scheme = lsrk_stage_9_order_5;
    // else if (selected_scheme == "forward_euler")
    //     scheme = forward_euler;
    // else if (selected_scheme == "ssp_stage_2_order_2")
    //     scheme = ssp_stage_2_order_2;
    // else if (selected_scheme == "ssp_stage_3_order_3")
    //     scheme = ssp_stage_3_order_3;
  }
  prm.leave_subsection();

  prm.enter_subsection("Operator parameters");
  {
    static const std::map<std::string, NumericalFlux> flux_map =
    {
      {"local_lax_friedrichs", local_lax_friedrichs},
      {"godunov", godunov}
    };

    std::string selected_flux = prm.get("numerical flux type");

    auto it = flux_map.find(selected_flux);
    if(it != flux_map.end())
      numerical_flux_type = it->second;
    else
      throw std::runtime_error("Unknown numerical flux: " + selected_flux);
    
    // if(selected_flux == "local_lax_friedrichs")
    //   numerical_flux_type = local_lax_friedrichs;
    // else if(selected_flux == "godunov")
    //   numerical_flux_type = godunov;
  }
  prm.leave_subsection();

  prm.enter_subsection("Limiter parameters");
  {
    static const std::map<std::string, LimiterType> limiter_map =
    {
      {"none", none},
      {"bound_preserving", bound_preserving}
    };

    std::string selected_limiter = prm.get("limiter");

    auto it = limiter_map.find(selected_limiter);
    if (it != limiter_map.end())
      limiter_type = it->second;
    else
      throw std::runtime_error("Unknown limiter: " + selected_limiter);
    

    // if(selected_limiter == "none")
    //   limiter_type = none;
    // else if(selected_limiter == "bound_preserving")
    //   limiter_type = bound_preserving;
    
    epsilon = prm.get_double("epsilon");
  }
}