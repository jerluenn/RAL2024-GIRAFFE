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
  #define CASADI_PREFIX(ID) asymmetric_hysteresis_model_impl_dae_fun_ ## ID
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

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[8] = {4, 1, 0, 4, 0, 1, 2, 3};

/* asymmetric_hysteresis_model_impl_dae_fun:(i0[3],i1[3],i2,i3,i4[])->(o0[4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[1]? arg[1][0] : 0;
  a1=arg[0]? arg[0][1] : 0;
  a2=(a0-a1);
  if (res[0]!=0) res[0][0]=a2;
  a2=arg[1]? arg[1][1] : 0;
  a3=arg[2]? arg[2][0] : 0;
  a4=arg[0]? arg[0][0] : 0;
  a5=2.;
  a5=(a5*a0);
  a4=(a4+a5);
  a3=(a3-a4);
  a3=(a2-a3);
  if (res[0]!=0) res[0][1]=a3;
  a3=arg[1]? arg[1][2] : 0;
  a4=1.7570000000000000e+01;
  a5=arg[0]? arg[0][2] : 0;
  a4=(a4*a5);
  a5=casadi_sq(a1);
  a6=1.0000000000000000e-03;
  a5=(a5+a6);
  a5=sqrt(a5);
  a7=arg[3]? arg[3][0] : 0;
  a8=9.9999999999999995e-07;
  a8=(a7+a8);
  a5=(a5/a8);
  a4=(a4*a5);
  a0=(a0-a4);
  a3=(a3-a0);
  if (res[0]!=0) res[0][2]=a3;
  a3=5.0000000000000000e-01;
  a0=(a1/a6);
  a0=tanh(a0);
  a0=(a3*a0);
  a0=(a3+a0);
  a4=2.0970000000000000e+00;
  a5=1.3680000000000001e+00;
  a8=1.7199999999999999e-01;
  a9=casadi_sq(a1);
  a9=(a9+a6);
  a9=sqrt(a9);
  a8=(a8*a9);
  a9=1.2280000000000000e+00;
  a9=(a9*a2);
  a9=casadi_sq(a9);
  a9=(a9+a6);
  a9=sqrt(a9);
  a10=1.;
  a9=(a9+a10);
  a8=(a8/a9);
  a8=(-a8);
  a8=exp(a8);
  a5=(a5*a8);
  a4=(a4+a5);
  a5=(a2/a6);
  a5=tanh(a5);
  a5=(a3*a5);
  a5=(a3+a5);
  a8=1.5970000000000000e+00;
  a9=1.6000000000000000e-02;
  a9=(a9*a2);
  a9=casadi_sq(a9);
  a9=(a9+a6);
  a9=sqrt(a9);
  a9=exp(a9);
  a10=(a10-a9);
  a8=(a8*a10);
  a5=(a5*a8);
  a4=(a4+a5);
  a0=(a0*a4);
  a7=(a7-a0);
  a0=(a1/a6);
  a0=(-a0);
  a0=tanh(a0);
  a0=(a3*a0);
  a3=(a3+a0);
  a0=3.9600000000000000e+00;
  a4=(a2/a6);
  a4=tanh(a4);
  a5=-1.8550000000000000e+00;
  a8=5.5490000000000004e+00;
  a1=casadi_sq(a1);
  a1=(a1+a6);
  a1=sqrt(a1);
  a8=(a8*a1);
  a2=casadi_sq(a2);
  a2=(a2+a6);
  a2=sqrt(a2);
  a6=5.0000000000000001e-03;
  a2=(a2+a6);
  a8=(a8/a2);
  a8=(-a8);
  a8=exp(a8);
  a5=(a5*a8);
  a4=(a4*a5);
  a0=(a0+a4);
  a3=(a3*a0);
  a7=(a7-a3);
  if (res[0]!=0) res[0][3]=a7;
  return 0;
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int asymmetric_hysteresis_model_impl_dae_fun_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int asymmetric_hysteresis_model_impl_dae_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real asymmetric_hysteresis_model_impl_dae_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* asymmetric_hysteresis_model_impl_dae_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* asymmetric_hysteresis_model_impl_dae_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* asymmetric_hysteresis_model_impl_dae_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* asymmetric_hysteresis_model_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
