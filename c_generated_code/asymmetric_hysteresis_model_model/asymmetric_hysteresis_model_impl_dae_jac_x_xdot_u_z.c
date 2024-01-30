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
  #define CASADI_PREFIX(ID) asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_ ## ID
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
static const casadi_int casadi_s2[12] = {4, 3, 0, 2, 5, 6, 1, 3, 0, 2, 3, 2};
static const casadi_int casadi_s3[11] = {4, 3, 0, 3, 4, 5, 0, 1, 3, 1, 2};
static const casadi_int casadi_s4[6] = {4, 1, 0, 2, 1, 3};
static const casadi_int casadi_s5[6] = {4, 1, 0, 2, 2, 3};

/* asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z:(i0[3],i1[3],i2,i3,i4)->(o0[4x3,6nz],o1[4x3,5nz],o2[4x1,2nz],o3[4x1,2nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a5, a6, a7, a8, a9;
  a0=4.;
  if (res[0]!=0) res[0][0]=a0;
  a0=5.0000000000000000e-01;
  a1=arg[0]? arg[0][1] : 0;
  a2=1.0000000000000000e-10;
  a3=(a1/a2);
  a3=tanh(a3);
  a4=(a0*a3);
  a4=(a0+a4);
  a5=1.3680000000000001e+00;
  a6=1.7199999999999999e-01;
  a7=casadi_sq(a1);
  a7=(a7+a2);
  a7=sqrt(a7);
  a8=(a6*a7);
  a9=1.5790000000000000e+00;
  a10=-4.;
  a11=arg[0]? arg[0][0] : 0;
  a11=(a10*a11);
  a12=6.;
  a13=arg[1]? arg[1][0] : 0;
  a13=(a12*a13);
  a11=(a11-a13);
  a13=arg[2]? arg[2][0] : 0;
  a11=(a11+a13);
  a13=(a9*a11);
  a14=casadi_sq(a13);
  a14=(a14+a2);
  a14=sqrt(a14);
  a15=1.;
  a16=(a14+a15);
  a8=(a8/a16);
  a17=(-a8);
  a17=exp(a17);
  a18=(a8/a16);
  a19=-6.3159999999999998e+00;
  a20=(a13+a13);
  a19=(a19*a20);
  a20=(a14+a14);
  a19=(a19/a20);
  a18=(a18*a19);
  a18=(a17*a18);
  a18=(a5*a18);
  a19=-1.8550000000000000e+00;
  a20=(a19*a11);
  a21=casadi_sq(a20);
  a21=(a21+a2);
  a21=sqrt(a21);
  a22=exp(a21);
  a23=(a15-a22);
  a23=(a9*a23);
  a24=-40000000000.;
  a25=(a11/a2);
  a25=tanh(a25);
  a26=casadi_sq(a25);
  a26=(a15-a26);
  a24=(a24*a26);
  a24=(a0*a24);
  a24=(a23*a24);
  a26=(a0*a25);
  a26=(a0+a26);
  a27=7.4199999999999999e+00;
  a28=(a20+a20);
  a27=(a27*a28);
  a28=(a21+a21);
  a27=(a27/a28);
  a27=(a22*a27);
  a27=(a9*a27);
  a27=(a26*a27);
  a24=(a24-a27);
  a18=(a18+a24);
  a18=(a4*a18);
  a24=(a1/a2);
  a24=(-a24);
  a24=tanh(a24);
  a27=(a0*a24);
  a27=(a0+a27);
  a28=3.9600000000000000e+00;
  a29=5.5490000000000004e+00;
  a30=casadi_sq(a1);
  a30=(a30+a2);
  a30=sqrt(a30);
  a31=(a29*a30);
  a32=casadi_sq(a11);
  a32=(a32+a2);
  a32=sqrt(a32);
  a33=5.0000000000000001e-03;
  a33=(a32+a33);
  a31=(a31/a33);
  a34=(-a31);
  a34=exp(a34);
  a35=(a28*a34);
  a36=40000000000.;
  a37=(a11/a2);
  a37=(-a37);
  a37=tanh(a37);
  a38=casadi_sq(a37);
  a38=(a15-a38);
  a36=(a36*a38);
  a36=(a35*a36);
  a38=(a31/a33);
  a39=(a11+a11);
  a10=(a10*a39);
  a39=(a32+a32);
  a10=(a10/a39);
  a38=(a38*a10);
  a38=(a34*a38);
  a38=(a28*a38);
  a38=(a37*a38);
  a36=(a36+a38);
  a36=(a27*a36);
  a18=(a18+a36);
  a18=(-a18);
  if (res[0]!=0) res[0][1]=a18;
  a18=-1.;
  if (res[0]!=0) res[0][2]=a18;
  a36=1.7570000000000000e+01;
  a38=arg[0]? arg[0][2] : 0;
  a10=casadi_sq(a1);
  a10=(a10+a2);
  a10=sqrt(a10);
  a2=(a1/a10);
  a2=(a38*a2);
  a39=arg[3]? arg[3][0] : 0;
  a40=9.9999999999999995e-07;
  a39=(a39+a40);
  a2=(a2/a39);
  a2=(a15-a2);
  a2=(a36*a2);
  a2=(-a2);
  if (res[0]!=0) res[0][3]=a2;
  a2=2.0970000000000000e+00;
  a40=(a5*a17);
  a2=(a2+a40);
  a40=(a26*a23);
  a2=(a2+a40);
  a40=10000000000.;
  a3=casadi_sq(a3);
  a3=(a15-a3);
  a3=(a40*a3);
  a3=(a0*a3);
  a2=(a2*a3);
  a7=(a1/a7);
  a6=(a6*a7);
  a6=(a6/a16);
  a6=(a17*a6);
  a6=(a5*a6);
  a6=(a4*a6);
  a2=(a2-a6);
  a6=(a37*a35);
  a6=(a28+a6);
  a7=-10000000000.;
  a24=casadi_sq(a24);
  a24=(a15-a24);
  a24=(a7*a24);
  a24=(a0*a24);
  a6=(a6*a24);
  a1=(a1/a30);
  a29=(a29*a1);
  a29=(a29/a33);
  a29=(a34*a29);
  a29=(a28*a29);
  a29=(a37*a29);
  a29=(a27*a29);
  a6=(a6-a29);
  a2=(a2+a6);
  a2=(-a2);
  if (res[0]!=0) res[0][4]=a2;
  a2=(a10/a39);
  a2=(a36*a2);
  if (res[0]!=0) res[0][5]=a2;
  if (res[1]!=0) res[1][0]=a15;
  if (res[1]!=0) res[1][1]=a12;
  a12=(a8/a16);
  a2=-9.4740000000000002e+00;
  a6=(a13+a13);
  a2=(a2*a6);
  a6=(a14+a14);
  a2=(a2/a6);
  a12=(a12*a2);
  a12=(a17*a12);
  a12=(a5*a12);
  a2=-60000000000.;
  a6=casadi_sq(a25);
  a6=(a15-a6);
  a2=(a2*a6);
  a2=(a0*a2);
  a2=(a23*a2);
  a6=1.1129999999999999e+01;
  a29=(a20+a20);
  a6=(a6*a29);
  a29=(a21+a21);
  a6=(a6/a29);
  a6=(a22*a6);
  a6=(a9*a6);
  a6=(a26*a6);
  a2=(a2-a6);
  a12=(a12+a2);
  a12=(a4*a12);
  a2=60000000000.;
  a6=casadi_sq(a37);
  a6=(a15-a6);
  a2=(a2*a6);
  a2=(a35*a2);
  a6=(a31/a33);
  a29=-6.;
  a1=(a11+a11);
  a29=(a29*a1);
  a1=(a32+a32);
  a29=(a29/a1);
  a6=(a6*a29);
  a6=(a34*a6);
  a6=(a28*a6);
  a6=(a37*a6);
  a2=(a2+a6);
  a2=(a27*a2);
  a12=(a12+a2);
  a12=(-a12);
  if (res[1]!=0) res[1][2]=a12;
  if (res[1]!=0) res[1][3]=a15;
  if (res[1]!=0) res[1][4]=a15;
  if (res[2]!=0) res[2][0]=a18;
  a8=(a8/a16);
  a13=(a13+a13);
  a13=(a9*a13);
  a14=(a14+a14);
  a13=(a13/a14);
  a8=(a8*a13);
  a17=(a17*a8);
  a5=(a5*a17);
  a25=casadi_sq(a25);
  a25=(a15-a25);
  a40=(a40*a25);
  a0=(a0*a40);
  a23=(a23*a0);
  a20=(a20+a20);
  a19=(a19*a20);
  a21=(a21+a21);
  a19=(a19/a21);
  a22=(a22*a19);
  a9=(a9*a22);
  a26=(a26*a9);
  a23=(a23-a26);
  a5=(a5+a23);
  a4=(a4*a5);
  a5=casadi_sq(a37);
  a5=(a15-a5);
  a7=(a7*a5);
  a35=(a35*a7);
  a31=(a31/a33);
  a11=(a11/a32);
  a31=(a31*a11);
  a34=(a34*a31);
  a28=(a28*a34);
  a37=(a37*a28);
  a35=(a35+a37);
  a27=(a27*a35);
  a4=(a4+a27);
  a4=(-a4);
  if (res[2]!=0) res[2][1]=a4;
  a38=(a38*a10);
  a38=(a38/a39);
  a38=(a38/a39);
  a36=(a36*a38);
  a36=(-a36);
  if (res[3]!=0) res[3][0]=a36;
  if (res[3]!=0) res[3][1]=a15;
  return 0;
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    case 4: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int asymmetric_hysteresis_model_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
