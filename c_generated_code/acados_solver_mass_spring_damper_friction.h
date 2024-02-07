/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_mass_spring_damper_friction_H_
#define ACADOS_SOLVER_mass_spring_damper_friction_H_

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define MASS_SPRING_DAMPER_FRICTION_NX     15
#define MASS_SPRING_DAMPER_FRICTION_NZ     1
#define MASS_SPRING_DAMPER_FRICTION_NU     1
#define MASS_SPRING_DAMPER_FRICTION_NP     2
#define MASS_SPRING_DAMPER_FRICTION_NBX    0
#define MASS_SPRING_DAMPER_FRICTION_NBX0   15
#define MASS_SPRING_DAMPER_FRICTION_NBU    1
#define MASS_SPRING_DAMPER_FRICTION_NSBX   0
#define MASS_SPRING_DAMPER_FRICTION_NSBU   0
#define MASS_SPRING_DAMPER_FRICTION_NSH    0
#define MASS_SPRING_DAMPER_FRICTION_NSG    0
#define MASS_SPRING_DAMPER_FRICTION_NSPHI  0
#define MASS_SPRING_DAMPER_FRICTION_NSHN   0
#define MASS_SPRING_DAMPER_FRICTION_NSGN   0
#define MASS_SPRING_DAMPER_FRICTION_NSPHIN 0
#define MASS_SPRING_DAMPER_FRICTION_NSBXN  0
#define MASS_SPRING_DAMPER_FRICTION_NS     0
#define MASS_SPRING_DAMPER_FRICTION_NSN    0
#define MASS_SPRING_DAMPER_FRICTION_NG     0
#define MASS_SPRING_DAMPER_FRICTION_NBXN   0
#define MASS_SPRING_DAMPER_FRICTION_NGN    0
#define MASS_SPRING_DAMPER_FRICTION_NY0    16
#define MASS_SPRING_DAMPER_FRICTION_NY     16
#define MASS_SPRING_DAMPER_FRICTION_NYN    15
#define MASS_SPRING_DAMPER_FRICTION_N      40
#define MASS_SPRING_DAMPER_FRICTION_NH     0
#define MASS_SPRING_DAMPER_FRICTION_NPHI   0
#define MASS_SPRING_DAMPER_FRICTION_NHN    0
#define MASS_SPRING_DAMPER_FRICTION_NPHIN  0
#define MASS_SPRING_DAMPER_FRICTION_NR     0

#ifdef __cplusplus
extern "C" {
#endif

// ** capsule for solver data **
typedef struct mass_spring_damper_friction_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *impl_dae_fun;
    external_function_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_param_casadi *impl_dae_jac_x_xdot_u_z;




    // cost






    // constraints




} mass_spring_damper_friction_solver_capsule;

mass_spring_damper_friction_solver_capsule * mass_spring_damper_friction_acados_create_capsule(void);
int mass_spring_damper_friction_acados_free_capsule(mass_spring_damper_friction_solver_capsule *capsule);

int mass_spring_damper_friction_acados_create(mass_spring_damper_friction_solver_capsule * capsule);
/**
 * Generic version of mass_spring_damper_friction_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
int mass_spring_damper_friction_acados_create_with_discretization(mass_spring_damper_friction_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
int mass_spring_damper_friction_acados_update_time_steps(mass_spring_damper_friction_solver_capsule * capsule, int N, double* new_time_steps);
int mass_spring_damper_friction_acados_update_params(mass_spring_damper_friction_solver_capsule * capsule, int stage, double *value, int np);
int mass_spring_damper_friction_acados_solve(mass_spring_damper_friction_solver_capsule * capsule);
int mass_spring_damper_friction_acados_free(mass_spring_damper_friction_solver_capsule * capsule);
void mass_spring_damper_friction_acados_print_stats(mass_spring_damper_friction_solver_capsule * capsule);

ocp_nlp_in *mass_spring_damper_friction_acados_get_nlp_in(mass_spring_damper_friction_solver_capsule * capsule);
ocp_nlp_out *mass_spring_damper_friction_acados_get_nlp_out(mass_spring_damper_friction_solver_capsule * capsule);
ocp_nlp_solver *mass_spring_damper_friction_acados_get_nlp_solver(mass_spring_damper_friction_solver_capsule * capsule);
ocp_nlp_config *mass_spring_damper_friction_acados_get_nlp_config(mass_spring_damper_friction_solver_capsule * capsule);
void *mass_spring_damper_friction_acados_get_nlp_opts(mass_spring_damper_friction_solver_capsule * capsule);
ocp_nlp_dims *mass_spring_damper_friction_acados_get_nlp_dims(mass_spring_damper_friction_solver_capsule * capsule);
ocp_nlp_plan *mass_spring_damper_friction_acados_get_nlp_plan(mass_spring_damper_friction_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_mass_spring_damper_friction_H_
