#ifndef AUTO_DIFF_INCLUDED
#define AUTO_DIFF_INCLUDED
// dual number arithmetic for automatic differentiation
// a dual number is represented by a float2/3/4 in 1/2/3D
// addition, substraction
float REAL_ADD(float a, float b) { return a + b; }
float DUAL_ADD(float2 a, float2 b) { return a.y + b.y; }
float REAL_SUB(float a, float b) { return a - b; }
float DUAL_SUB(float2 a, float2 b) { return a.y - b.y; }
// multiplication
float REAL_MUL(float a, float b) { return a * b; }
float DUAL_MUL(float2 a, float2 b) { return a.y * b.x + a.x * b.y; }
// division
float REAL_DIV(float a, float b) { return a / b; }
float DUAL_DIV(float2 a, float2 b) { return (a.y * b.x - a.x * b.y) / (b.x * b.x); }
// min, max
float REAL_MIN(float a, float b) { return a <= b ? a : b; }
float DUAL_MIN(float2 a, float2 b) { return a.x < b.x ? a.y : b.y; }
float REAL_MAX(float a, float b) { return a >= b ? a : b; }
float DUAL_MAX(float2 a, float2 b) { return a.x > b.x ? a.y : b.y; }
// helpers to define a fucntion with arity 1
#define DUAL_FUN_A1(dfun, a) (a.y*(dfun(a.x)))
#define DUAL_1D_FUN(fun, dfun, a) float2(fun(a.x), DUAL_FUN_A1(dfun, a.xy))
#define DUAL_2D_FUN(fun, dfun, a) float3(DUAL_1D_FUN(fun, dfun, a), DUAL_FUN_A1(dfun, a.xz))
#define DUAL_3D_FUN(fun, dfun, a) float4(DUAL_2D_FUN(fun, dfun, a), DUAL_FUN_A1(dfun, a.xw))
// helpers to define a function with arity 2 (operator)
#define DUAL_1D_OP(op, dop, a, b) float2(op(a.x, b.x), dop(a.xy, b.xy))
#define DUAL_2D_OP(op, dop, a, b) float3(DUAL_1D_OP(op, dop, a, b), dop(a.xz, b.xz))
#define DUAL_3D_OP(op, dop, a, b) float4(DUAL_2D_OP(op, dop, a, b), dop(a.xw, b.xw))
// helpers to define a function with arity 1 and an extra non-variable parameter
#define DUAL_FUN_A1_PAR(dfun, a, p) (a.y*(dfun(a.x, p)))
#define DUAL_1D_FUN_PAR(fun, dfun, a, p) float2(fun(a.x, p), DUAL_FUN_A1_PAR(dfun, a.xy, p))
#define DUAL_2D_FUN_PAR(fun, dfun, a, p) float3(DUAL_1D_FUN_PAR(fun, dfun, a, p), DUAL_FUN_A1_PAR(dfun, a.xz, p))
#define DUAL_3D_FUN_PAR(fun, dfun, a, p) float4(DUAL_2D_FUN_PAR(fun, dfun, a, p), DUAL_FUN_A1_PAR(dfun, a.xw, p))

// defines and overloads the arity 1 function `NAME` for usage with dual numbers
#define DEF_DUAL_FUN(NAME, fun, dfun) \
float  NAME(float  a){return fun(a);} \
float2 NAME(float2 a){return DUAL_1D_FUN(fun, dfun, a);} \
float3 NAME(float3 a){return DUAL_2D_FUN(fun, dfun, a);} \
float4 NAME(float4 a){return DUAL_3D_FUN(fun, dfun, a);}
// defines and overloads the arity 2 function `NAME` for usage with dual numbers
// also defines the versions between dual number and a regular number
#define DEF_DUAL_OP(NAME, op, dop) \
float  NAME(float  a, float  b){return op(a, b);} \
float2 NAME(float2 a, float2 b){return DUAL_1D_OP(op, dop, a, b);} \
float2 NAME(float  a, float2 b){return NAME(CONST_1D(a), b);} \
float2 NAME(float2 a, float  b){return NAME(a, CONST_1D(b));} \
float3 NAME(float3 a, float3 b){return DUAL_2D_OP(op, dop, a, b);} \
float3 NAME(float  a, float3 b){return NAME(CONST_2D(a), b);} \
float3 NAME(float3 a, float  b){return NAME(a, CONST_2D(b));} \
float4 NAME(float4 a, float4 b){return DUAL_3D_OP(op, dop, a, b);} \
float4 NAME(float  a, float4 b){return NAME(CONST_3D(a), b);} \
float4 NAME(float4 a, float  b){return NAME(a, CONST_3D(b));}
// defines and overloads the arity 1 (+ a non-variable parameter) function `NAME`
#define DEF_DUAL_FUN_PAR(NAME, fun, dfun, type) \
float  NAME(float  a, type par){return fun(a, par);} \
float2 NAME(float2 a, type par){return DUAL_1D_FUN_PAR(fun, dfun, a, par);} \
float3 NAME(float3 a, type par){return DUAL_2D_FUN_PAR(fun, dfun, a, par);} \
float4 NAME(float4 a, type par){return DUAL_3D_FUN_PAR(fun, dfun, a, par);}
// like above, but the parameter is fixed
#define DEF_DUAL_FUN_FIXPAR(NAME, fun, dfun, par) \
float  NAME(float  a){return fun(a, par);} \
float2 NAME(float2 a){return DUAL_1D_FUN_PAR(fun, dfun, a, par);} \
float3 NAME(float3 a){return DUAL_2D_FUN_PAR(fun, dfun, a, par);} \
float4 NAME(float4 a){return DUAL_3D_FUN_PAR(fun, dfun, a, par);}
// defines and overloads a function called `NAME` for usage with dual numbers
#define TEMPLATE_DUAL_FUN_A1(NAME, def, p1) \
float  NAME(float  p1){return def;} \
float2 NAME(float2 p1){return def;} \
float3 NAME(float3 p1){return def;} \
float4 NAME(float4 p1){return def;}


// constant valued and variable dual numbers in different dimensions
#define CONST_1D(a) float2(a, 0)
#define CONST_2D(a) float3(a, 0, 0)
#define CONST_3D(a) float4(a, 0, 0, 0)
#define VAR1_1D(a)  float2(a, 1)
#define VAR1_2D(a)  float3(a, 1, 0)
#define VAR2_2D(a)  float3(a, 0, 1)
#define VAR1_3D(a)  float4(a, 1, 0, 0)
#define VAR2_3D(a)  float4(a, 0, 1, 0)
#define VAR3_3D(a)  float4(a, 0, 0, 1)
#define CONST CONST_1D
#define VAR VAR1_1D
// function defines for use with dual numbers
DEF_DUAL_OP(ADD, REAL_ADD, DUAL_ADD)
DEF_DUAL_OP(SUB, REAL_SUB, DUAL_SUB)
DEF_DUAL_OP(MUL, REAL_MUL, DUAL_MUL)
DEF_DUAL_OP(DIV, REAL_DIV, DUAL_DIV)
DEF_DUAL_OP(MIN, REAL_MIN, DUAL_MIN)
DEF_DUAL_OP(MAX, REAL_MAX, DUAL_MAX)
DEF_DUAL_FUN(SIN, sin, cos)
DEF_DUAL_FUN(COS, cos, -sin)
DEF_DUAL_FUN(EXP, exp, exp)
float dlog(float a) { return 1 / a; }
DEF_DUAL_FUN(LOG, log, dlog)
DEF_DUAL_FUN(ABS, abs, sign)
float dsqrt(float a) { return 0.5 / sqrt(a); }
DEF_DUAL_FUN(SQRT, sqrt, dsqrt)
float dpow(float a, float b) { return b * pow(a, b - 1); }
DEF_DUAL_FUN_PAR(POW, pow, dpow, float)
DEF_DUAL_FUN_FIXPAR(POW2, pow, dpow, 2)
DEF_DUAL_FUN_FIXPAR(POW3, pow, dpow, 3)
DEF_DUAL_FUN_FIXPAR(POW4, pow, dpow, 4)
float mod_(float a, float b) { return a % b; }
float dmod_(float a, float b) { return 1; }
DEF_DUAL_FUN_PAR(MOD, mod_, dmod_, float)
#define LERP(x, y, s) ADD(x,MUL(s,SUB(y,x)))
#define MIX LERP
#define CLAMP(x, a, b) MIN(MAX(x,a),b)
TEMPLATE_DUAL_FUN_A1(SATURATE, CLAMP(x,0,1), x)
TEMPLATE_DUAL_FUN_A1(HERMITE3, MUL(MUL(t,t),SUB(3,MUL(2,t))), t)
#define PAR_BETWEEN(e1, e2, x) SATURATE(DIV(SUB(x,e1),SUB(e2,e1)))
#define SMOOTHSTEP(e1, e2, x) HERMITE3(PAR_BETWEEN(e1,e2,x))
TEMPLATE_DUAL_FUN_A1(HERMITE5, MUL(MUL(t,t),MUL(t,ADD(MUL(t,SUB(MUL(6,t), 15)),10))), t)
#define SMOOTHERSTEP(e1, e2, x) HERMITE5(PAR_BETWEEN(e1,e2,x))


#endif
