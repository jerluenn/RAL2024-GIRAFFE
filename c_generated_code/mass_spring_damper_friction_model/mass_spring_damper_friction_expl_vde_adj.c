/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) mass_spring_damper_friction_expl_vde_adj_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[13] = {9, 1, 0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s3[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

/* mass_spring_damper_friction_expl_vde_adj:(i0[9],i1[9],i2,i3[2])->(o0[10]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=3.7699111843077517e+04;
  a1=1.1477519934511685e+05;
  a2=arg[1]? arg[1][4] : 0;
  a2=(a1*a2);
  a3=arg[1]? arg[1][3] : 0;
  a3=(a1*a3);
  a4=(a2-a3);
  a5=(a0*a4);
  if (res[0]!=0) res[0][0]=a5;
  a5=arg[1]? arg[1][5] : 0;
  a1=(a1*a5);
  a5=(a1-a2);
  a6=(a0*a5);
  a7=-3.7699111843077517e+04;
  a4=(a7*a4);
  a6=(a6+a4);
  if (res[0]!=0) res[0][1]=a6;
  a7=(a7*a5);
  a0=(a0*a1);
  a7=(a7-a0);
  if (res[0]!=0) res[0][2]=a7;
  a7=750000.;
  a0=arg[1]? arg[1][6] : 0;
  a0=(a7*a0);
  a5=arg[0]? arg[0][3] : 0;
  a6=(a5+a5);
  a4=arg[0]? arg[0][6] : 0;
  a8=50.;
  a9=(a8*a0);
  a4=(a4*a9);
  a5=casadi_sq(a5);
  a10=1.0000000000000000e-10;
  a5=(a5+a10);
  a5=sqrt(a5);
  a11=(a5+a5);
  a4=(a4/a11);
  a6=(a6*a4);
  a0=(a0-a6);
  a6=100.;
  a4=(a6*a2);
  a0=(a0+a4);
  a4=-100.;
  a4=(a4*a3);
  a0=(a0+a4);
  a4=arg[1]? arg[1][0] : 0;
  a0=(a0+a4);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a7*a0);
  a4=arg[0]? arg[0][4] : 0;
  a11=(a4+a4);
  a12=arg[0]? arg[0][7] : 0;
  a13=(a8*a0);
  a12=(a12*a13);
  a4=casadi_sq(a4);
  a4=(a4+a10);
  a4=sqrt(a4);
  a14=(a4+a4);
  a12=(a12/a14);
  a11=(a11*a12);
  a0=(a0-a11);
  a11=(a6*a1);
  a0=(a0+a11);
  a11=-200.;
  a12=(a11*a2);
  a0=(a0+a12);
  a12=(a6*a3);
  a0=(a0+a12);
  a12=arg[1]? arg[1][1] : 0;
  a0=(a0+a12);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[1]? arg[1][8] : 0;
  a7=(a7*a0);
  a0=arg[0]? arg[0][5] : 0;
  a12=(a0+a0);
  a14=arg[0]? arg[0][8] : 0;
  a8=(a8*a7);
  a14=(a14*a8);
  a0=casadi_sq(a0);
  a0=(a0+a10);
  a0=sqrt(a0);
  a10=(a0+a0);
  a14=(a14/a10);
  a12=(a12*a14);
  a7=(a7-a12);
  a11=(a11*a1);
  a7=(a7+a11);
  a6=(a6*a2);
  a7=(a7+a6);
  a6=arg[1]? arg[1][2] : 0;
  a7=(a7+a6);
  if (res[0]!=0) res[0][5]=a7;
  a5=(a5*a9);
  a5=(-a5);
  if (res[0]!=0) res[0][6]=a5;
  a4=(a4*a13);
  a4=(-a4);
  if (res[0]!=0) res[0][7]=a4;
  a0=(a0*a8);
  a0=(-a0);
  if (res[0]!=0) res[0][8]=a0;
  if (res[0]!=0) res[0][9]=a3;
  return 0;
}

CASADI_SYMBOL_EXPORT int mass_spring_damper_friction_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int mass_spring_damper_friction_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int mass_spring_damper_friction_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void mass_spring_damper_friction_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int mass_spring_damper_friction_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void mass_spring_damper_friction_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void mass_spring_damper_friction_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void mass_spring_damper_friction_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int mass_spring_damper_friction_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int mass_spring_damper_friction_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real mass_spring_damper_friction_expl_vde_adj_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* mass_spring_damper_friction_expl_vde_adj_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* mass_spring_damper_friction_expl_vde_adj_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* mass_spring_damper_friction_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* mass_spring_damper_friction_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int mass_spring_damper_friction_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
