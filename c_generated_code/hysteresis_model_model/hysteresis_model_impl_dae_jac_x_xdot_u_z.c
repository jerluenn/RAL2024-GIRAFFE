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
  #define CASADI_PREFIX(ID) hysteresis_model_impl_dae_jac_x_xdot_u_z_ ## ID
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
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
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

static const casadi_int casadi_s0[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[9] = {3, 2, 0, 3, 4, 0, 1, 2, 1};
static const casadi_int casadi_s4[7] = {3, 2, 0, 1, 2, 0, 1};
static const casadi_int casadi_s5[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s6[5] = {3, 1, 0, 1, 2};

/* hysteresis_model_impl_dae_jac_x_xdot_u_z:(i0[2],i1[2],i2,i3,i4[])->(o0[3x2,4nz],o1[3x2,2nz],o2[3],o3[3x1,1nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a4, a5, a6, a7, a8, a9;
  a0=3.3333333333333331e-01;
  if (res[0]!=0) res[0][0]=a0;
  a1=3.;
  a2=-3.3333333333333331e-01;
  a3=arg[0]? arg[0][1] : 0;
  a4=casadi_sq(a3);
  a5=1.0000000000000000e-10;
  a4=(a4+a5);
  a6=sqrt(a4);
  a7=(a6*a4);
  a8=6.9999999999999996e-01;
  a9=arg[2]? arg[2][0] : 0;
  a10=arg[0]? arg[0][0] : 0;
  a11=(a9-a10);
  a11=(a0*a11);
  a12=(a11+a11);
  a12=(a2*a12);
  a13=casadi_sq(a11);
  a13=(a13+a5);
  a13=sqrt(a13);
  a14=(a13+a13);
  a12=(a12/a14);
  a12=(a8*a12);
  a12=(a3*a12);
  a12=(a7*a12);
  a12=(a2-a12);
  a14=1.0000000000000001e-01;
  a15=casadi_sq(a3);
  a15=(a15+a5);
  a16=casadi_sq(a15);
  a17=(a14*a16);
  a12=(a12-a17);
  a12=(a1*a12);
  a12=(-a12);
  if (res[0]!=0) res[0][1]=a12;
  a12=9.0000000000000002e-01;
  a17=5.9343034025940089e-01;
  a17=(a17*a10);
  a10=(a12*a17);
  a14=(a14*a17);
  a18=casadi_sq(a9);
  a18=(a18+a5);
  a18=sqrt(a18);
  a19=(-a18);
  a19=exp(a19);
  a20=(a14*a19);
  a10=(a10+a20);
  a20=5.0000000000000000e-01;
  a21=-3.3333333333333330e+09;
  a22=1.;
  a23=(a11/a5);
  a23=tanh(a23);
  a24=casadi_sq(a23);
  a24=(a22-a24);
  a24=(a21*a24);
  a24=(a20*a24);
  a24=(a10*a24);
  a25=(a20*a23);
  a25=(a20+a25);
  a26=5.3408730623346079e-01;
  a27=5.9343034025940093e-02;
  a27=(a27*a19);
  a27=(a26+a27);
  a27=(a25*a27);
  a24=(a24+a27);
  a27=9.9999999999999978e-02;
  a27=(a27*a17);
  a12=(a12*a17);
  a17=casadi_sq(a9);
  a17=(a17+a5);
  a17=sqrt(a17);
  a28=(-a17);
  a28=exp(a28);
  a29=(a12*a28);
  a27=(a27+a29);
  a29=3.3333333333333330e+09;
  a5=(a11/a5);
  a5=(-a5);
  a5=tanh(a5);
  a30=casadi_sq(a5);
  a30=(a22-a30);
  a30=(a29*a30);
  a30=(a20*a30);
  a30=(a27*a30);
  a31=(a20*a5);
  a31=(a20+a31);
  a32=5.9343034025940079e-02;
  a26=(a26*a28);
  a32=(a32+a26);
  a32=(a31*a32);
  a30=(a30+a32);
  a24=(a24+a30);
  a24=(-a24);
  if (res[0]!=0) res[0][2]=a24;
  a24=(a8*a13);
  a30=(a7*a24);
  a24=(a24*a3);
  a32=(a3/a6);
  a4=(a4*a32);
  a32=(a3+a3);
  a6=(a6*a32);
  a4=(a4+a6);
  a24=(a24*a4);
  a30=(a30+a24);
  a24=-3.0000000000000004e-01;
  a24=(a24*a11);
  a15=(a15+a15);
  a4=(a3+a3);
  a15=(a15*a4);
  a24=(a24*a15);
  a30=(a30+a24);
  a30=(a1*a30);
  if (res[0]!=0) res[0][3]=a30;
  if (res[1]!=0) res[1][0]=a22;
  if (res[1]!=0) res[1][1]=a22;
  if (res[2]!=0) res[2][0]=a2;
  a11=(a11+a11);
  a11=(a0*a11);
  a13=(a13+a13);
  a11=(a11/a13);
  a8=(a8*a11);
  a3=(a3*a8);
  a7=(a7*a3);
  a0=(a0-a7);
  a7=-1.0000000000000001e-01;
  a7=(a7*a16);
  a0=(a0-a7);
  a1=(a1*a0);
  a1=(-a1);
  if (res[2]!=0) res[2][1]=a1;
  a23=casadi_sq(a23);
  a23=(a22-a23);
  a29=(a29*a23);
  a29=(a20*a29);
  a10=(a10*a29);
  a18=(a9/a18);
  a19=(a19*a18);
  a14=(a14*a19);
  a25=(a25*a14);
  a10=(a10-a25);
  a5=casadi_sq(a5);
  a5=(a22-a5);
  a21=(a21*a5);
  a20=(a20*a21);
  a27=(a27*a20);
  a9=(a9/a17);
  a28=(a28*a9);
  a12=(a12*a28);
  a31=(a31*a12);
  a27=(a27-a31);
  a10=(a10+a27);
  a10=(-a10);
  if (res[2]!=0) res[2][2]=a10;
  if (res[3]!=0) res[3][0]=a22;
  return 0;
}

CASADI_SYMBOL_EXPORT int hysteresis_model_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int hysteresis_model_impl_dae_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int hysteresis_model_impl_dae_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void hysteresis_model_impl_dae_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int hysteresis_model_impl_dae_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void hysteresis_model_impl_dae_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void hysteresis_model_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void hysteresis_model_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int hysteresis_model_impl_dae_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int hysteresis_model_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real hysteresis_model_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* hysteresis_model_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* hysteresis_model_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* hysteresis_model_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* hysteresis_model_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int hysteresis_model_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
