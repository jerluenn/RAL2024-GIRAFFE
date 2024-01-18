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

// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "hysteresis_model_model/hysteresis_model_model.h"




#include "hysteresis_model_cost/hysteresis_model_cost_y_fun.h"
#include "hysteresis_model_cost/hysteresis_model_cost_y_0_fun.h"
#include "hysteresis_model_cost/hysteresis_model_cost_y_e_fun.h"

#include "acados_solver_hysteresis_model.h"

#define NX     HYSTERESIS_MODEL_NX
#define NZ     HYSTERESIS_MODEL_NZ
#define NU     HYSTERESIS_MODEL_NU
#define NP     HYSTERESIS_MODEL_NP
#define NBX    HYSTERESIS_MODEL_NBX
#define NBX0   HYSTERESIS_MODEL_NBX0
#define NBU    HYSTERESIS_MODEL_NBU
#define NSBX   HYSTERESIS_MODEL_NSBX
#define NSBU   HYSTERESIS_MODEL_NSBU
#define NSH    HYSTERESIS_MODEL_NSH
#define NSG    HYSTERESIS_MODEL_NSG
#define NSPHI  HYSTERESIS_MODEL_NSPHI
#define NSHN   HYSTERESIS_MODEL_NSHN
#define NSGN   HYSTERESIS_MODEL_NSGN
#define NSPHIN HYSTERESIS_MODEL_NSPHIN
#define NSBXN  HYSTERESIS_MODEL_NSBXN
#define NS     HYSTERESIS_MODEL_NS
#define NSN    HYSTERESIS_MODEL_NSN
#define NG     HYSTERESIS_MODEL_NG
#define NBXN   HYSTERESIS_MODEL_NBXN
#define NGN    HYSTERESIS_MODEL_NGN
#define NY0    HYSTERESIS_MODEL_NY0
#define NY     HYSTERESIS_MODEL_NY
#define NYN    HYSTERESIS_MODEL_NYN
// #define N      HYSTERESIS_MODEL_N
#define NH     HYSTERESIS_MODEL_NH
#define NPHI   HYSTERESIS_MODEL_NPHI
#define NHN    HYSTERESIS_MODEL_NHN
#define NPHIN  HYSTERESIS_MODEL_NPHIN
#define NR     HYSTERESIS_MODEL_NR


// ** solver data **

hysteresis_model_solver_capsule * hysteresis_model_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(hysteresis_model_solver_capsule));
    hysteresis_model_solver_capsule *capsule = (hysteresis_model_solver_capsule *) capsule_mem;

    return capsule;
}


int hysteresis_model_acados_free_capsule(hysteresis_model_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int hysteresis_model_acados_create(hysteresis_model_solver_capsule * capsule)
{
    int N_shooting_intervals = HYSTERESIS_MODEL_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return hysteresis_model_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}

int hysteresis_model_acados_update_time_steps(hysteresis_model_solver_capsule * capsule, int N, double* new_time_steps)
{
    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "hysteresis_model_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;
}

int hysteresis_model_acados_create_with_discretization(hysteresis_model_solver_capsule * capsule, int N, double* new_time_steps)
{
    int status = 0;
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != HYSTERESIS_MODEL_N && !new_time_steps) {
        fprintf(stderr, "hysteresis_model_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, HYSTERESIS_MODEL_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    /************************************************
    *  plan & config
    ************************************************/
    ocp_nlp_plan * nlp_solver_plan = ocp_nlp_plan_create(N);
    capsule->nlp_solver_plan = nlp_solver_plan;
    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    nlp_solver_plan->nlp_cost[0] = NONLINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = NONLINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = NONLINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = ERK;
    }

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;
    ocp_nlp_config * nlp_config = ocp_nlp_config_create(*nlp_solver_plan);
    capsule->nlp_config = nlp_config;


    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 17
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
    }

    // for initial state
    nbx[0]  = NBX0;
    nsbx[0] = 0;
    ns[0] = NS - NSBX;
    nbxe[0] = 2;
    ny[0] = NY0;

    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);
    capsule->nlp_dims = nlp_dims;

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);

    for (int i = 0; i < N; i++)
    {
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);

    free(intNp1mem);



    /************************************************
    *  external functions
    ************************************************/


    // explicit ode
    capsule->forw_vde_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->forw_vde_casadi[i].casadi_fun = &hysteresis_model_expl_vde_forw;
        capsule->forw_vde_casadi[i].casadi_n_in = &hysteresis_model_expl_vde_forw_n_in;
        capsule->forw_vde_casadi[i].casadi_n_out = &hysteresis_model_expl_vde_forw_n_out;
        capsule->forw_vde_casadi[i].casadi_sparsity_in = &hysteresis_model_expl_vde_forw_sparsity_in;
        capsule->forw_vde_casadi[i].casadi_sparsity_out = &hysteresis_model_expl_vde_forw_sparsity_out;
        capsule->forw_vde_casadi[i].casadi_work = &hysteresis_model_expl_vde_forw_work;
        external_function_param_casadi_create(&capsule->forw_vde_casadi[i], 2);
    }

    capsule->expl_ode_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->expl_ode_fun[i].casadi_fun = &hysteresis_model_expl_ode_fun;
        capsule->expl_ode_fun[i].casadi_n_in = &hysteresis_model_expl_ode_fun_n_in;
        capsule->expl_ode_fun[i].casadi_n_out = &hysteresis_model_expl_ode_fun_n_out;
        capsule->expl_ode_fun[i].casadi_sparsity_in = &hysteresis_model_expl_ode_fun_sparsity_in;
        capsule->expl_ode_fun[i].casadi_sparsity_out = &hysteresis_model_expl_ode_fun_sparsity_out;
        capsule->expl_ode_fun[i].casadi_work = &hysteresis_model_expl_ode_fun_work;
        external_function_param_casadi_create(&capsule->expl_ode_fun[i], 2);
    }


    // nonlinear least square function
    capsule->cost_y_0_fun.casadi_fun = &hysteresis_model_cost_y_0_fun;
    capsule->cost_y_0_fun.casadi_n_in = &hysteresis_model_cost_y_0_fun_n_in;
    capsule->cost_y_0_fun.casadi_n_out = &hysteresis_model_cost_y_0_fun_n_out;
    capsule->cost_y_0_fun.casadi_sparsity_in = &hysteresis_model_cost_y_0_fun_sparsity_in;
    capsule->cost_y_0_fun.casadi_sparsity_out = &hysteresis_model_cost_y_0_fun_sparsity_out;
    capsule->cost_y_0_fun.casadi_work = &hysteresis_model_cost_y_0_fun_work;
    external_function_param_casadi_create(&capsule->cost_y_0_fun, 2);

    capsule->cost_y_0_fun_jac_ut_xt.casadi_fun = &hysteresis_model_cost_y_0_fun_jac_ut_xt;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_n_in = &hysteresis_model_cost_y_0_fun_jac_ut_xt_n_in;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_n_out = &hysteresis_model_cost_y_0_fun_jac_ut_xt_n_out;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_sparsity_in = &hysteresis_model_cost_y_0_fun_jac_ut_xt_sparsity_in;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_sparsity_out = &hysteresis_model_cost_y_0_fun_jac_ut_xt_sparsity_out;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_work = &hysteresis_model_cost_y_0_fun_jac_ut_xt_work;
    external_function_param_casadi_create(&capsule->cost_y_0_fun_jac_ut_xt, 2);

    capsule->cost_y_0_hess.casadi_fun = &hysteresis_model_cost_y_0_hess;
    capsule->cost_y_0_hess.casadi_n_in = &hysteresis_model_cost_y_0_hess_n_in;
    capsule->cost_y_0_hess.casadi_n_out = &hysteresis_model_cost_y_0_hess_n_out;
    capsule->cost_y_0_hess.casadi_sparsity_in = &hysteresis_model_cost_y_0_hess_sparsity_in;
    capsule->cost_y_0_hess.casadi_sparsity_out = &hysteresis_model_cost_y_0_hess_sparsity_out;
    capsule->cost_y_0_hess.casadi_work = &hysteresis_model_cost_y_0_hess_work;
    external_function_param_casadi_create(&capsule->cost_y_0_hess, 2);
    // nonlinear least squares cost
    capsule->cost_y_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        capsule->cost_y_fun[i].casadi_fun = &hysteresis_model_cost_y_fun;
        capsule->cost_y_fun[i].casadi_n_in = &hysteresis_model_cost_y_fun_n_in;
        capsule->cost_y_fun[i].casadi_n_out = &hysteresis_model_cost_y_fun_n_out;
        capsule->cost_y_fun[i].casadi_sparsity_in = &hysteresis_model_cost_y_fun_sparsity_in;
        capsule->cost_y_fun[i].casadi_sparsity_out = &hysteresis_model_cost_y_fun_sparsity_out;
        capsule->cost_y_fun[i].casadi_work = &hysteresis_model_cost_y_fun_work;

        external_function_param_casadi_create(&capsule->cost_y_fun[i], 2);
    }

    capsule->cost_y_fun_jac_ut_xt = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        capsule->cost_y_fun_jac_ut_xt[i].casadi_fun = &hysteresis_model_cost_y_fun_jac_ut_xt;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_n_in = &hysteresis_model_cost_y_fun_jac_ut_xt_n_in;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_n_out = &hysteresis_model_cost_y_fun_jac_ut_xt_n_out;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_sparsity_in = &hysteresis_model_cost_y_fun_jac_ut_xt_sparsity_in;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_sparsity_out = &hysteresis_model_cost_y_fun_jac_ut_xt_sparsity_out;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_work = &hysteresis_model_cost_y_fun_jac_ut_xt_work;

        external_function_param_casadi_create(&capsule->cost_y_fun_jac_ut_xt[i], 2);
    }

    capsule->cost_y_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        capsule->cost_y_hess[i].casadi_fun = &hysteresis_model_cost_y_hess;
        capsule->cost_y_hess[i].casadi_n_in = &hysteresis_model_cost_y_hess_n_in;
        capsule->cost_y_hess[i].casadi_n_out = &hysteresis_model_cost_y_hess_n_out;
        capsule->cost_y_hess[i].casadi_sparsity_in = &hysteresis_model_cost_y_hess_sparsity_in;
        capsule->cost_y_hess[i].casadi_sparsity_out = &hysteresis_model_cost_y_hess_sparsity_out;
        capsule->cost_y_hess[i].casadi_work = &hysteresis_model_cost_y_hess_work;

        external_function_param_casadi_create(&capsule->cost_y_hess[i], 2);
    }
    // nonlinear least square function
    capsule->cost_y_e_fun.casadi_fun = &hysteresis_model_cost_y_e_fun;
    capsule->cost_y_e_fun.casadi_n_in = &hysteresis_model_cost_y_e_fun_n_in;
    capsule->cost_y_e_fun.casadi_n_out = &hysteresis_model_cost_y_e_fun_n_out;
    capsule->cost_y_e_fun.casadi_sparsity_in = &hysteresis_model_cost_y_e_fun_sparsity_in;
    capsule->cost_y_e_fun.casadi_sparsity_out = &hysteresis_model_cost_y_e_fun_sparsity_out;
    capsule->cost_y_e_fun.casadi_work = &hysteresis_model_cost_y_e_fun_work;
    external_function_param_casadi_create(&capsule->cost_y_e_fun, 2);

    capsule->cost_y_e_fun_jac_ut_xt.casadi_fun = &hysteresis_model_cost_y_e_fun_jac_ut_xt;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_n_in = &hysteresis_model_cost_y_e_fun_jac_ut_xt_n_in;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_n_out = &hysteresis_model_cost_y_e_fun_jac_ut_xt_n_out;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_sparsity_in = &hysteresis_model_cost_y_e_fun_jac_ut_xt_sparsity_in;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_sparsity_out = &hysteresis_model_cost_y_e_fun_jac_ut_xt_sparsity_out;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_work = &hysteresis_model_cost_y_e_fun_jac_ut_xt_work;
    external_function_param_casadi_create(&capsule->cost_y_e_fun_jac_ut_xt, 2);

    capsule->cost_y_e_hess.casadi_fun = &hysteresis_model_cost_y_e_hess;
    capsule->cost_y_e_hess.casadi_n_in = &hysteresis_model_cost_y_e_hess_n_in;
    capsule->cost_y_e_hess.casadi_n_out = &hysteresis_model_cost_y_e_hess_n_out;
    capsule->cost_y_e_hess.casadi_sparsity_in = &hysteresis_model_cost_y_e_hess_sparsity_in;
    capsule->cost_y_e_hess.casadi_sparsity_out = &hysteresis_model_cost_y_e_hess_sparsity_out;
    capsule->cost_y_e_hess.casadi_work = &hysteresis_model_cost_y_e_hess_work;
    external_function_param_casadi_create(&capsule->cost_y_e_hess, 2);

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
    capsule->nlp_in = nlp_in;

    // set up time_steps
    

    if (new_time_steps) {
        hysteresis_model_acados_update_time_steps(capsule, N, new_time_steps);
    } else {// all time_steps are identical
        double time_step = 0.05;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &time_step);
        }
    }

    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "expl_vde_forw", &capsule->forw_vde_casadi[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "expl_ode_fun", &capsule->expl_ode_fun[i]);
    
    }


    /**** Cost ****/

    double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 10;
    W_0[1+(NY0) * 1] = 0.01;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);

    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);



    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    
    W[0+(NY) * 0] = 10;
    W[1+(NY) * 1] = 0.01;

    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(W);
    free(yref);


    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun", &capsule->cost_y_0_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun_jac", &capsule->cost_y_0_fun_jac_ut_xt);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "nls_y_hess", &capsule->cost_y_0_hess);
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun", &capsule->cost_y_fun[i-1]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun_jac", &capsule->cost_y_fun_jac_ut_xt[i-1]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "nls_y_hess", &capsule->cost_y_hess[i-1]);
    }




    // terminal cost


    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    
    W_e[0+(NYN) * 0] = 100;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun", &capsule->cost_y_e_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun_jac", &capsule->cost_y_e_fun_jac_ut_xt);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_hess", &capsule->cost_y_e_hess);



    /**** Constraints ****/

    // bounds for initial stage

    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);


    // idxbxe_0
    int* idxbxe_0 = malloc(2 * sizeof(int));
    
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);


    /* constraints that are the same for initial and intermediate */



    // u
    int* idxbu = malloc(NBU * sizeof(int));
    
    idxbu[0] = 0;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    
    lbu[0] = -30;
    ubu[0] = 30;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);


















    /* terminal constraints */

















    /************************************************
    *  opts
    ************************************************/

    capsule->nlp_opts = ocp_nlp_solver_opts_create(nlp_config, nlp_dims);


    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization", "fixed_step");

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 10;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 10;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);


    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double nlp_solver_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "step_length", &nlp_solver_step_length);

    double levenberg_marquardt = 1;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;
    // NOTE: there is no condensing happening here!
    qp_solver_cond_N = N;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);


    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_iter_max", &qp_solver_iter_max);

    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "print_level", &print_level);


    int ext_cost_num_hess = 0;


    /* out */
    ocp_nlp_out * nlp_out = ocp_nlp_out_create(nlp_config, nlp_dims);
    capsule->nlp_out = nlp_out;

    /* sens_out */
    ocp_nlp_out *sens_out = ocp_nlp_out_create(nlp_config, nlp_dims);
    capsule->sens_out = sens_out;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0
    


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    free(xu0);
    
    capsule->nlp_solver = ocp_nlp_solver_create(nlp_config, nlp_dims, capsule->nlp_opts);



    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));
    
    p[0] = 0.5;
    p[1] = 0.5;

    for (int i = 0; i <= N; i++)
    {
        hysteresis_model_acados_update_params(capsule, i, p, NP);
    }
    free(p);

    status = ocp_nlp_precompute(capsule->nlp_solver, nlp_in, nlp_out);

    if (status != ACADOS_SUCCESS)
    {
        printf("\nocp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int hysteresis_model_acados_update_params(hysteresis_model_solver_capsule * capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 2;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    const int N = capsule->nlp_solver_plan->N;
    if (stage < N && stage >= 0)
    {
        capsule->forw_vde_casadi[stage].set_param(capsule->forw_vde_casadi+stage, p);
        capsule->expl_ode_fun[stage].set_param(capsule->expl_ode_fun+stage, p);
    

        // constraints
    

        // cost
        if (stage == 0)
        {
            capsule->cost_y_0_fun.set_param(&capsule->cost_y_0_fun, p);
            capsule->cost_y_0_fun_jac_ut_xt.set_param(&capsule->cost_y_0_fun_jac_ut_xt, p);
            capsule->cost_y_0_hess.set_param(&capsule->cost_y_0_hess, p);
        }
        else // 0 < stage < N
        {
            capsule->cost_y_fun[stage-1].set_param(capsule->cost_y_fun+stage-1, p);
            capsule->cost_y_fun_jac_ut_xt[stage-1].set_param(capsule->cost_y_fun_jac_ut_xt+stage-1, p);
            capsule->cost_y_hess[stage-1].set_param(capsule->cost_y_hess+stage-1, p);
        }
    }

    else // stage == N
    {
        // terminal shooting node has no dynamics
        // cost
        capsule->cost_y_e_fun.set_param(&capsule->cost_y_e_fun, p);
        capsule->cost_y_e_fun_jac_ut_xt.set_param(&capsule->cost_y_e_fun_jac_ut_xt, p);
        capsule->cost_y_e_hess.set_param(&capsule->cost_y_e_hess, p);
        // constraints
    
    }


    return solver_status;
}



int hysteresis_model_acados_solve(hysteresis_model_solver_capsule * capsule)
{
    // solve NLP 
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}


int hysteresis_model_acados_free(hysteresis_model_solver_capsule * capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_param_casadi_free(&capsule->forw_vde_casadi[i]);
        external_function_param_casadi_free(&capsule->expl_ode_fun[i]);
    }
    free(capsule->forw_vde_casadi);
    free(capsule->expl_ode_fun);

    // cost
    external_function_param_casadi_free(&capsule->cost_y_0_fun);
    external_function_param_casadi_free(&capsule->cost_y_0_fun_jac_ut_xt);
    external_function_param_casadi_free(&capsule->cost_y_0_hess);
    for (int i = 0; i < N - 1; i++)
    {
        external_function_param_casadi_free(&capsule->cost_y_fun[i]);
        external_function_param_casadi_free(&capsule->cost_y_fun_jac_ut_xt[i]);
        external_function_param_casadi_free(&capsule->cost_y_hess[i]);
    }
    free(capsule->cost_y_fun);
    free(capsule->cost_y_fun_jac_ut_xt);
    free(capsule->cost_y_hess);
    external_function_param_casadi_free(&capsule->cost_y_e_fun);
    external_function_param_casadi_free(&capsule->cost_y_e_fun_jac_ut_xt);
    external_function_param_casadi_free(&capsule->cost_y_e_hess);

    // constraints

    return 0;
}

ocp_nlp_in *hysteresis_model_acados_get_nlp_in(hysteresis_model_solver_capsule * capsule) { return capsule->nlp_in; }
ocp_nlp_out *hysteresis_model_acados_get_nlp_out(hysteresis_model_solver_capsule * capsule) { return capsule->nlp_out; }
ocp_nlp_out *hysteresis_model_acados_get_sens_out(hysteresis_model_solver_capsule * capsule) { return capsule->sens_out; }
ocp_nlp_solver *hysteresis_model_acados_get_nlp_solver(hysteresis_model_solver_capsule * capsule) { return capsule->nlp_solver; }
ocp_nlp_config *hysteresis_model_acados_get_nlp_config(hysteresis_model_solver_capsule * capsule) { return capsule->nlp_config; }
void *hysteresis_model_acados_get_nlp_opts(hysteresis_model_solver_capsule * capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *hysteresis_model_acados_get_nlp_dims(hysteresis_model_solver_capsule * capsule) { return capsule->nlp_dims; }
ocp_nlp_plan *hysteresis_model_acados_get_nlp_plan(hysteresis_model_solver_capsule * capsule) { return capsule->nlp_solver_plan; }


void hysteresis_model_acados_print_stats(hysteresis_model_solver_capsule * capsule)
{
    int sqp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_m", &stat_m);

    
    double stat[1000];
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "statistics", stat);

    int nrow = sqp_iter+1 < stat_m ? sqp_iter+1 : stat_m;

    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j > 4)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }
}

