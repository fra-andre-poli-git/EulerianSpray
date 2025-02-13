#include"Parameters.h"

void Parameters::declare_parameters(ParameterHandler & prm)
{
  prm.enter_subsection("Problem parameters");
  {
    prm.declare_entry("testcase",
      "1",
      Patterns::Integer(0), // Is it ok even if I want an unsigned?
      "Number of case:"
        "- 1: vacuum formation"
        "- 2: delta shock"
        "- 3: 2d");

    // prm.declare_entry("finite element degree",
    //   "0",
    //   Patterns::Integer(0),
    //   "Polynomial degree of finite element basis function for the elements");

    prm.declare_entry("number of elements in x direction",
      "400",
      Patterns::Integer(0),
      "Number of elements along the x direction: for 1d cases elements along y "
        "direction will be set as 5, whereas in 2d cases they will be set "
        "identical as this parameter");

    prm.declare_entry("final time",
      "0.5",
      Patterns::Double(0),
      "Final value of time");

    prm.declare_entry("snapshot instant",
      "0.05",
      Patterns::Double(0),
      "Time between every snapshot of the solution");
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
      Patterns::MultipleSelection("local_lax_friedrichs|godunov"),
      "Type of the numerical flux used between the interfaces of the elements");
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
    snapshot = prm.get_double("snapshot instant");
  }
  prm.leave_subsection();

  prm.enter_subsection("Integrator parameters");
  {
    std::string selected_scheme = prm.get("Runge Kutta scheme");

    // The parser gets a bit cumbersome if I have to deal with enum:
    // honestly I made ChatGPT write this chain of else if because it was too
    // boring
    if (selected_scheme == "lsrk_stage_3_order_3")
        scheme = lsrk_stage_3_order_3;
    else if (selected_scheme == "lsrk_stage_5_order_4")
        scheme = lsrk_stage_5_order_4;
    else if (selected_scheme == "lsrk_stage_7_order_4")
        scheme = lsrk_stage_7_order_4;
    else if (selected_scheme == "lsrk_stage_9_order_5")
        scheme = lsrk_stage_9_order_5;
    else if (selected_scheme == "forward_euler")
        scheme = forward_euler;
    else if (selected_scheme == "ssp_stage_2_order_2")
        scheme = ssp_stage_2_order_2;
    else if (selected_scheme == "ssp_stage_3_order_3")
        scheme = ssp_stage_3_order_3;
  }
  prm.leave_subsection();

  prm.enter_subsection("Operator parameters");
  {
    std::string selected_flux = prm.get("numerical flux type");
    if(selected_flux == "local_lax_friedrichs")
      numerical_flux_type = local_lax_friedrichs;
    else if(selected_flux == "godunov")
      numerical_flux_type = godunov;
  }
  prm.leave_subsection();
}