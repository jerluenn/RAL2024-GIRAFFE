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
  #define CASADI_PREFIX(ID) asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_ ## ID
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
#define casadi_s7 CASADI_PREFIX(s7)
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
static const casadi_int casadi_s4[11] = {4, 3, 0, 1, 4, 5, 1, 0, 2, 3, 2};
static const casadi_int casadi_s5[12] = {4, 3, 0, 3, 5, 6, 0, 1, 2, 1, 3, 2};
static const casadi_int casadi_s6[5] = {4, 1, 0, 1, 1};
static const casadi_int casadi_s7[6] = {4, 1, 0, 2, 2, 3};

/* asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z:(i0[3],i1[3],i2,i3,i4[])->(o0[4],o1[4x3,5nz],o2[4x3,6nz],o3[4x1,1nz],o4[4x1,2nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a5, a6, a7, a8, a9;
  a0=arg[1]? arg[1][0] : 0;
  a1=arg[0]? arg[0][1] : 0;
  a2=(a0-a1);
  if (res[0]!=0) res[0][0]=a2;
  a2=arg[1]? arg[1][1] : 0;
  a3=arg[2]? arg[2][0] : 0;
  a4=arg[0]? arg[0][0] : 0;
  a5=2.;
  a6=(a5*a0);
  a4=(a4+a6);
  a3=(a3-a4);
  a3=(a2-a3);
  if (res[0]!=0) res[0][1]=a3;
  a3=arg[1]? arg[1][2] : 0;
  a4=1.7570000000000000e+01;
  a6=arg[0]? arg[0][2] : 0;
  a6=(a4*a6);
  a7=casadi_sq(a1);
  a8=1.0000000000000000e-03;
  a7=(a7+a8);
  a7=sqrt(a7);
  a9=arg[3]? arg[3][0] : 0;
  a10=9.9999999999999995e-07;
  a10=(a9+a10);
  a11=(a7/a10);
  a12=(a6*a11);
  a0=(a0-a12);
  a3=(a3-a0);
  if (res[0]!=0) res[0][2]=a3;
  a3=5.0000000000000000e-01;
  a0=(a1/a8);
  a0=tanh(a0);
  a12=(a3*a0);
  a12=(a3+a12);
  a13=2.0970000000000000e+00;
  a14=1.3680000000000001e+00;
  a15=1.7199999999999999e-01;
  a16=casadi_sq(a1);
  a16=(a16+a8);
  a16=sqrt(a16);
  a17=(a15*a16);
  a18=1.2280000000000000e+00;
  a19=(a18*a2);
  a20=casadi_sq(a19);
  a20=(a20+a8);
  a20=sqrt(a20);
  a21=1.;
  a22=(a20+a21);
  a17=(a17/a22);
  a23=(-a17);
  a23=exp(a23);
  a24=(a14*a23);
  a13=(a13+a24);
  a24=(a2/a8);
  a24=tanh(a24);
  a25=(a3*a24);
  a25=(a3+a25);
  a26=1.5970000000000000e+00;
  a27=1.6000000000000000e-02;
  a28=(a27*a2);
  a29=casadi_sq(a28);
  a29=(a29+a8);
  a29=sqrt(a29);
  a30=exp(a29);
  a31=(a21-a30);
  a31=(a26*a31);
  a32=(a25*a31);
  a13=(a13+a32);
  a32=(a12*a13);
  a9=(a9-a32);
  a32=(a1/a8);
  a32=(-a32);
  a32=tanh(a32);
  a33=(a3*a32);
  a33=(a3+a33);
  a34=3.9600000000000000e+00;
  a35=(a2/a8);
  a35=tanh(a35);
  a36=-1.8550000000000000e+00;
  a37=5.5490000000000004e+00;
  a38=casadi_sq(a1);
  a38=(a38+a8);
  a38=sqrt(a38);
  a39=(a37*a38);
  a40=casadi_sq(a2);
  a40=(a40+a8);
  a40=sqrt(a40);
  a8=5.0000000000000001e-03;
  a8=(a40+a8);
  a39=(a39/a8);
  a41=(-a39);
  a41=exp(a41);
  a42=(a36*a41);
  a43=(a35*a42);
  a34=(a34+a43);
  a43=(a33*a34);
  a9=(a9-a43);
  if (res[0]!=0) res[0][3]=a9;
  if (res[1]!=0) res[1][0]=a21;
  a9=-1.;
  if (res[1]!=0) res[1][1]=a9;
  a7=(a1/a7);
  a7=(a7/a10);
  a7=(a6*a7);
  if (res[1]!=0) res[1][2]=a7;
  a7=1000.;
  a0=casadi_sq(a0);
  a0=(a21-a0);
  a0=(a7*a0);
  a0=(a3*a0);
  a13=(a13*a0);
  a16=(a1/a16);
  a15=(a15*a16);
  a15=(a15/a22);
  a15=(a23*a15);
  a15=(a14*a15);
  a15=(a12*a15);
  a13=(a13-a15);
  a15=-1000.;
  a32=casadi_sq(a32);
  a32=(a21-a32);
  a15=(a15*a32);
  a15=(a3*a15);
  a34=(a34*a15);
  a1=(a1/a38);
  a37=(a37*a1);
  a37=(a37/a8);
  a37=(a41*a37);
  a37=(a36*a37);
  a37=(a35*a37);
  a37=(a33*a37);
  a34=(a34-a37);
  a13=(a13+a34);
  a13=(-a13);
  if (res[1]!=0) res[1][3]=a13;
  a4=(a4*a11);
  if (res[1]!=0) res[1][4]=a4;
  if (res[2]!=0) res[2][0]=a21;
  if (res[2]!=0) res[2][1]=a5;
  if (res[2]!=0) res[2][2]=a9;
  if (res[2]!=0) res[2][3]=a21;
  a17=(a17/a22);
  a19=(a19+a19);
  a18=(a18*a19);
  a20=(a20+a20);
  a18=(a18/a20);
  a17=(a17*a18);
  a23=(a23*a17);
  a14=(a14*a23);
  a24=casadi_sq(a24);
  a24=(a21-a24);
  a24=(a7*a24);
  a3=(a3*a24);
  a31=(a31*a3);
  a28=(a28+a28);
  a27=(a27*a28);
  a29=(a29+a29);
  a27=(a27/a29);
  a30=(a30*a27);
  a26=(a26*a30);
  a25=(a25*a26);
  a31=(a31-a25);
  a14=(a14+a31);
  a12=(a12*a14);
  a14=casadi_sq(a35);
  a14=(a21-a14);
  a7=(a7*a14);
  a42=(a42*a7);
  a39=(a39/a8);
  a2=(a2/a40);
  a39=(a39*a2);
  a41=(a41*a39);
  a36=(a36*a41);
  a35=(a35*a36);
  a42=(a42+a35);
  a33=(a33*a42);
  a12=(a12+a33);
  a12=(-a12);
  if (res[2]!=0) res[2][4]=a12;
  if (res[2]!=0) res[2][5]=a21;
  if (res[3]!=0) res[3][0]=a9;
  a11=(a11/a10);
  a6=(a6*a11);
  a6=(-a6);
  if (res[4]!=0) res[4][0]=a6;
  if (res[4]!=0) res[4][1]=a21;
  return 0;
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    case 4: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_fun_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
