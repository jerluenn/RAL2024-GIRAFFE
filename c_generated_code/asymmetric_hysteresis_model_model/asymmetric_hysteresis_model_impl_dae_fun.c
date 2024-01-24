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
static const casadi_int casadi_s2[8] = {4, 1, 0, 4, 0, 1, 2, 3};

/* asymmetric_hysteresis_model_impl_dae_fun:(i0[3],i1[3],i2,i3,i4)->(o0[4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[1]? arg[1][0] : 0;
  a1=arg[0]? arg[0][1] : 0;
  a2=(a0-a1);
  if (res[0]!=0) res[0][0]=a2;
  a2=arg[1]? arg[1][1] : 0;
  a3=-4.;
  a4=arg[0]? arg[0][0] : 0;
  a5=(a3*a4);
  a6=6.;
  a7=(a6*a0);
  a5=(a5-a7);
  a7=arg[2]? arg[2][0] : 0;
  a5=(a5+a7);
  a2=(a2-a5);
  if (res[0]!=0) res[0][1]=a2;
  a2=arg[1]? arg[1][2] : 0;
  a5=2.0050000000000001e+01;
  a8=arg[0]? arg[0][2] : 0;
  a9=casadi_sq(a1);
  a10=1.0000000000000000e-10;
  a9=(a9+a10);
  a9=sqrt(a9);
  a8=(a8*a9);
  a9=arg[3]? arg[3][0] : 0;
  a11=9.9999999999999995e-07;
  a11=(a9+a11);
  a8=(a8/a11);
  a8=(a1-a8);
  a5=(a5*a8);
  a2=(a2-a5);
  if (res[0]!=0) res[0][2]=a2;
  a2=5.0000000000000000e-01;
  a5=(a1/a10);
  a5=tanh(a5);
  a5=(a2*a5);
  a5=(a2+a5);
  a8=9.9999999999999978e-02;
  a11=1.;
  a12=-4.5000000000000001e-01;
  a13=arg[4]? arg[4][0] : 0;
  a12=(a12*a13);
  a12=exp(a12);
  a12=(a11-a12);
  a12=(a4*a12);
  a8=(a8*a12);
  a13=9.0000000000000002e-01;
  a13=(a13*a12);
  a14=7.1999999999999995e-02;
  a15=casadi_sq(a1);
  a15=(a15+a10);
  a15=sqrt(a15);
  a14=(a14*a15);
  a15=2.2280000000000002e+00;
  a3=(a3*a4);
  a6=(a6*a0);
  a3=(a3-a6);
  a3=(a3+a7);
  a15=(a15*a3);
  a15=casadi_sq(a15);
  a15=(a15+a10);
  a15=sqrt(a15);
  a15=(a15+a11);
  a14=(a14/a15);
  a14=(-a14);
  a14=exp(a14);
  a13=(a13*a14);
  a8=(a8+a13);
  a13=(a3/a10);
  a13=tanh(a13);
  a13=(a2*a13);
  a13=(a2+a13);
  a14=-1.0000000000000000e-02;
  a15=1.8559999999999999e-01;
  a15=(a15*a3);
  a15=casadi_sq(a15);
  a15=(a15+a10);
  a15=sqrt(a15);
  a15=exp(a15);
  a11=(a11-a15);
  a14=(a14*a11);
  a13=(a13*a14);
  a8=(a8+a13);
  a5=(a5*a8);
  a9=(a9-a5);
  a5=(a1/a10);
  a5=(-a5);
  a5=tanh(a5);
  a5=(a2*a5);
  a5=(a2+a5);
  a2=(a2*a12);
  a8=(a3/a10);
  a8=(-a8);
  a8=tanh(a8);
  a13=-5.0000000000000000e-01;
  a13=(a13*a12);
  a12=1.5489999999999999e+00;
  a1=casadi_sq(a1);
  a1=(a1+a10);
  a1=sqrt(a1);
  a12=(a12*a1);
  a3=casadi_sq(a3);
  a3=(a3+a10);
  a3=sqrt(a3);
  a10=5.0000000000000001e-03;
  a3=(a3+a10);
  a12=(a12/a3);
  a12=(-a12);
  a12=exp(a12);
  a13=(a13*a12);
  a8=(a8*a13);
  a2=(a2+a8);
  a5=(a5*a2);
  a9=(a9-a5);
  if (res[0]!=0) res[0][3]=a9;
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
    case 4: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* asymmetric_hysteresis_model_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
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
