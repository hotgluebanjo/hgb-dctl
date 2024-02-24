// HGB DCTL library 0.1.0
// ----------------------
// Wheel reinvention!
// Attempting to be compatible with all DCTL targets.
//
// - https://github.com/hotgluebanjo
//
// References:
//
// - https://registry.khronos.org/OpenCL/specs/3.0-unified/pdf/OpenCL_C.pdf
// - https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
// - https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
//
// Since unions can't have anonymous structs it seems like a good idea to
// simply transmute floatN instances for indexing rather than make a custom
// VecN type. This also preserves the operator overloading.

#ifndef HGB_SPOW
    #define HGB_SPOW HGB_Spow_Mirror
#endif

// TODO: OpenCL.
#if defined(_WIN32) || defined(_WIN64)
    #define HGB_OS_WINDOWS
#elif defined(__APPLE__) && defined(__MACH__)
    #define HGB_OS_MAC
#elif defined(__linux__)
    #define HGB_OS_LINUX
#else
    #error "HGB: Unsupported OS"
#endif

#if defined(DEVICE_IS_METAL)
    #define HGB_DEVICE_METAL
#elif defined(DEVICE_IS_CUDA)
    #define HGB_DEVICE_CUDA
#elif defined(DEVICE_IS_OPENCL)
    #define HGB_DEVICE_OPENCL
#else
    #error "HGB: Unknown device framework"
#endif

#define nil 0
#define loop while (true)
#define for_range(i, min, max) for (hgb_usize i = min; i < max; i += 1)
#define cast(T) (T)
#define transmute(x, T) (*((T *)(&x)))

// Some, uh, assumptions. stdint is not included.
typedef unsigned char   hgb_u8;
typedef unsigned short hgb_u16;
typedef unsigned int   hgb_u32;
typedef unsigned long  hgb_u64;

typedef signed char   hgb_i8;
typedef signed short hgb_i16;
typedef signed int   hgb_i32;
typedef signed long  hgb_i64;

typedef size_t    hgb_usize;
typedef ptrdiff_t hgb_isize;

typedef float hgb_f32;

typedef hgb_u8 hgb_byte;

#define HGB_PI  3.14159265358979323846264338327950288f
#define HGB_TAU 6.28318530717958647692528676655900576f
#define HGB_E   2.71828182845904523536f

#define HGB_SQRT_TWO   1.41421356237309504880168872420969808f
#define HGB_SQRT_THREE 1.73205080756887729352744634150587236f

#define HGB_OK      hgb_vec3(0.0f, 1.0f, 0.0f)
#define HGB_WARNING hgb_vec3(1.0f, 1.0f, 0.0f)
#define HGB_ERROR   hgb_vec3(1.0f, 0.0f, 0.0f)

#define hgb_square(x) ((x) * (x))
#define hgb_cube(x) ((x) * (x) * (x))

#define hgb_max(a, b) ((a) > (b) ? (a) : (b))
#define hgb_min(a, b) ((a) < (b) ? (a) : (b))

#define hgb_max3(a, b, c) hgb_max(a, hgb_max(b, c))
#define hgb_min3(a, b, c) hgb_min(a, hgb_min(b, c))

#define hgb_clamp(x, start, end) hgb_max(hgb_min((x), (end)), (start))
#define hgb_clamp01(x) hgb_clamp((x), 0, 1)
#define hgb_saturate hgb_clamp01

#define hgb_abs(x) ((x) < 0 ? -(x) : (x))
#define hgb_sign(x) ((x) < 0 ? -1 : (x) > 0 ? 1 : 0)

__DEVICE__ inline hgb_f32 hgb_fdim(hgb_f32 x, hgb_f32 y) { return _fdimf(x, y); }
__DEVICE__ inline hgb_f32 hgb_fma(hgb_f32 x, hgb_f32 y, hgb_f32 z) { return _fmaf(x, y, z); }
__DEVICE__ inline hgb_f32 hgb_lgamma(hgb_f32 x) { return _lgammaf(x); }
__DEVICE__ inline hgb_f32 hgb_tgamma(hgb_f32 x) { return _tgammaf(x); }
__DEVICE__ inline hgb_f32 hgb_rsqrt(hgb_f32 x) { return _rsqrtf(x); }

__DEVICE__ inline bool hgb_isinf(hgb_f32 x) { return cast(bool)isinf(x); }
__DEVICE__ inline bool hgb_isnan(hgb_f32 x) { return cast(bool)isnan(x); }

__DEVICE__ inline bool hgb_signbit(hgb_f32 x) { return cast(bool)signbit(x); }
__DEVICE__ inline hgb_f32 hgb_copysign(hgb_f32 x, hgb_f32 y) { return _copysignf(x, y); }
__DEVICE__ inline hgb_f32 hgb_frexp(hgb_f32 x, hgb_i32 exp) { return _frexp(x, exp); }
__DEVICE__ inline hgb_f32 hgb_ldexp(hgb_f32 x, hgb_i32 exp) { return _ldexp(x, exp); }

typedef float2 HGB_Vec2;
typedef float3 HGB_Vec3;
typedef float4 HGB_Vec4;

__DEVICE__ inline HGB_Vec2 hgb_vec2(hgb_f32 x, hgb_f32 y) {
    HGB_Vec2 out = {x, y};
    return out;
}

__DEVICE__ inline HGB_Vec3 hgb_vec3(hgb_f32 x, hgb_f32 y, hgb_f32 z) {
    HGB_Vec3 out = {x, y, z};
    return out;
}

__DEVICE__ inline HGB_Vec4 hgb_vec4(hgb_f32 x, hgb_f32 y, hgb_f32 z, hgb_f32 w) {
    HGB_Vec4 out = {x, y, z, w};
    return out;
}

__DEVICE__ inline HGB_Vec2 hgb_repeat_vec2(hgb_f32 v) { return hgb_vec2(v, v); }
__DEVICE__ inline HGB_Vec3 hgb_repeat_vec3(hgb_f32 v) { return hgb_vec3(v, v, v); }
__DEVICE__ inline HGB_Vec4 hgb_repeat_vec4(hgb_f32 v) { return hgb_vec4(v, v, v, v); }

__DEVICE__ inline HGB_Vec2 hgb_zeros_vec2() { return hgb_repeat_vec2(0.0f); }
__DEVICE__ inline HGB_Vec3 hgb_zeros_vec3() { return hgb_repeat_vec3(0.0f); }
__DEVICE__ inline HGB_Vec4 hgb_zeros_vec4() { return hgb_repeat_vec4(0.0f); }

// No function pointers in OpenCL.
#define hgb_map_vec2(v, f) hgb_vec2(f(v.x), f(v.y))
#define hgb_map_vec3(v, f) hgb_vec3(f(v.x), f(v.y), f(v.z))
#define hgb_map_vec4(v, f) hgb_vec4(f(v.x), f(v.y), f(v.z), f(v.w))

__DEVICE__ HGB_Vec2 hgb_swizzle_vec2(HGB_Vec2 v, hgb_usize x, hgb_usize y) {
    x = hgb_clamp(x, 0, 1);
    y = hgb_clamp(y, 0, 1);
    hgb_f32 *it = &transmute(v, hgb_f32);
    return hgb_vec2(it[x], it[y]);
}

__DEVICE__ HGB_Vec3 hgb_swizzle_vec3(HGB_Vec3 v, hgb_usize x, hgb_usize y, hgb_usize z) {
    x = hgb_clamp(x, 0, 2);
    y = hgb_clamp(y, 0, 2);
    z = hgb_clamp(z, 0, 2);
    hgb_f32 *it = &transmute(v, hgb_f32);
    return hgb_vec3(it[x], it[y], it[z]);
}

__DEVICE__ HGB_Vec4 hgb_swizzle_vec4(HGB_Vec4 v, hgb_usize x, hgb_usize y, hgb_usize z, hgb_usize w) {
    x = hgb_clamp(x, 0, 3);
    y = hgb_clamp(y, 0, 3);
    z = hgb_clamp(z, 0, 3);
    w = hgb_clamp(w, 0, 3);
    hgb_f32 *it = &transmute(v, hgb_f32);
    return hgb_vec4(it[x], it[y], it[z], it[w]);
}

__DEVICE__ inline hgb_f32 hgb_pow(hgb_f32 v, hgb_f32 p) { return _powf(v, p); }
__DEVICE__ inline HGB_Vec2 hgb_pow_vec2(HGB_Vec2 v, float p) { return hgb_vec2(hgb_pow(v.x, p), hgb_pow(v.y, p)); }
__DEVICE__ inline HGB_Vec3 hgb_pow_vec3(HGB_Vec3 v, float p) { return hgb_vec3(hgb_pow(v.x, p), hgb_pow(v.y, p), hgb_pow(v.z, p)); }
__DEVICE__ inline HGB_Vec4 hgb_pow_vec4(HGB_Vec4 v, float p) { return hgb_vec4(hgb_pow(v.x, p), hgb_pow(v.y, p), hgb_pow(v.z, p), hgb_pow(v.w, p)); }

#define hgb_ln      hgb_log
#define hgb_ln_vec2 hgb_log_vec2
#define hgb_ln_vec3 hgb_log_vec3
#define hgb_ln_vec4 hgb_log_vec4
__DEVICE__ inline hgb_f32  hgb_log(hgb_f32 x) { return _logf(x); }
__DEVICE__ inline HGB_Vec2 hgb_log_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_log(v.x), hgb_log(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_log_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_log(v.x), hgb_log(v.y), hgb_log(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_log_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_log(v.x), hgb_log(v.y), hgb_log(v.z), hgb_log(v.w)); }

__DEVICE__ inline hgb_f32  hgb_log2(hgb_f32 x) { return _log2f(x); }
__DEVICE__ inline HGB_Vec2 hgb_log2_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_log2(v.x), hgb_log2(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_log2_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_log2(v.x), hgb_log2(v.y), hgb_log2(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_log2_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_log2(v.x), hgb_log2(v.y), hgb_log2(v.z), hgb_log2(v.w)); }

__DEVICE__ inline hgb_f32  hgb_log10(hgb_f32 x) { return _log10f(x); }
__DEVICE__ inline HGB_Vec2 hgb_log10_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_log10(v.x), hgb_log10(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_log10_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_log10(v.x), hgb_log10(v.y), hgb_log10(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_log10_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_log10(v.x), hgb_log10(v.y), hgb_log10(v.z), hgb_log10(v.w)); }

__DEVICE__ inline hgb_f32  hgb_exp(hgb_f32 x) { return _expf(x); }
__DEVICE__ inline HGB_Vec2 hgb_exp_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_exp(v.x), hgb_exp(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_exp_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_exp(v.x), hgb_exp(v.y), hgb_exp(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_exp_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_exp(v.x), hgb_exp(v.y), hgb_exp(v.z), hgb_exp(v.w)); }

__DEVICE__ inline hgb_f32  hgb_exp2(hgb_f32 x) { return _exp2f(x); }
__DEVICE__ inline HGB_Vec2 hgb_exp2_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_exp2(v.x), hgb_exp2(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_exp2_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_exp2(v.x), hgb_exp2(v.y), hgb_exp2(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_exp2_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_exp2(v.x), hgb_exp2(v.y), hgb_exp2(v.z), hgb_exp2(v.w)); }

__DEVICE__ inline hgb_f32  hgb_exp10(hgb_f32 x) { return _exp10f(x); }
__DEVICE__ inline HGB_Vec2 hgb_exp10_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_exp10(v.x), hgb_exp10(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_exp10_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_exp10(v.x), hgb_exp10(v.y), hgb_exp10(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_exp10_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_exp10(v.x), hgb_exp10(v.y), hgb_exp10(v.z), hgb_exp10(v.w)); }

__DEVICE__ inline hgb_f32  hgb_sqrt(hgb_f32 x) { return _sqrtf(x); }
__DEVICE__ inline HGB_Vec2 hgb_sqrt_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_sqrt(v.x), hgb_sqrt(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_sqrt_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_sqrt(v.x), hgb_sqrt(v.y), hgb_sqrt(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_sqrt_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_sqrt(v.x), hgb_sqrt(v.y), hgb_sqrt(v.z), hgb_sqrt(v.w)); }

__DEVICE__ inline hgb_f32  hgb_cbrt(hgb_f32 x) { return _cbrtf(x); }
__DEVICE__ inline HGB_Vec2 hgb_cbrt_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_cbrt(v.x), hgb_cbrt(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_cbrt_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_cbrt(v.x), hgb_cbrt(v.y), hgb_cbrt(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_cbrt_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_cbrt(v.x), hgb_cbrt(v.y), hgb_cbrt(v.z), hgb_cbrt(v.w)); }

__DEVICE__ inline hgb_f32  hgb_ceil(hgb_f32 x) { return _ceil(x); }
__DEVICE__ inline HGB_Vec2 hgb_ceil_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_ceil(v.x), hgb_ceil(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_ceil_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_ceil(v.x), hgb_ceil(v.y), hgb_ceil(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_ceil_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_ceil(v.x), hgb_ceil(v.y), hgb_ceil(v.z), hgb_ceil(v.w)); }

__DEVICE__ inline hgb_f32  hgb_floor(hgb_f32 x) { return _floor(x); }
__DEVICE__ inline HGB_Vec2 hgb_floor_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_floor(v.x), hgb_floor(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_floor_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_floor(v.x), hgb_floor(v.y), hgb_floor(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_floor_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_floor(v.x), hgb_floor(v.y), hgb_floor(v.z), hgb_floor(v.w)); }

__DEVICE__ inline hgb_f32  hgb_trunc(hgb_f32 x) { return _truncf(x); }
__DEVICE__ inline HGB_Vec2 hgb_trunc_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_trunc(v.x), hgb_trunc(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_trunc_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_trunc(v.x), hgb_trunc(v.y), hgb_trunc(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_trunc_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_trunc(v.x), hgb_trunc(v.y), hgb_trunc(v.z), hgb_trunc(v.w)); }

__DEVICE__ inline hgb_f32  hgb_round(hgb_f32 x) { return _round(x); }
__DEVICE__ inline HGB_Vec2 hgb_round_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_round(v.x), hgb_round(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_round_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_round(v.x), hgb_round(v.y), hgb_round(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_round_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_round(v.x), hgb_round(v.y), hgb_round(v.z), hgb_round(v.w)); }

__DEVICE__ inline hgb_f32  hgb_mod(hgb_f32 x, hgb_f32 y) { return _fmod(x, y); }
__DEVICE__ inline HGB_Vec2 hgb_mod_vec2(HGB_Vec2 vx, HGB_Vec2 vy) { return hgb_vec2(hgb_mod(vx.x, vy.x), hgb_mod(vx.y, vy.y)); }
__DEVICE__ inline HGB_Vec3 hgb_mod_vec3(HGB_Vec3 vx, HGB_Vec3 vy) { return hgb_vec3(hgb_mod(vx.x, vy.x), hgb_mod(vx.y, vy.y), hgb_mod(vx.z, vy.z)); }
__DEVICE__ inline HGB_Vec4 hgb_mod_vec4(HGB_Vec4 vx, HGB_Vec4 vy) { return hgb_vec4(hgb_mod(vx.x, vy.x), hgb_mod(vx.y, vy.y), hgb_mod(vx.z, vy.z), hgb_mod(vx.w, vy.w)); }

__DEVICE__ inline hgb_f32  hgb_remainder(hgb_f32 x, hgb_f32 y) { return _fremainder(x, y); }
__DEVICE__ inline HGB_Vec2 hgb_remainder_vec2(HGB_Vec2 vx, HGB_Vec2 vy) { return hgb_vec2(hgb_remainder(vx.x, vy.x), hgb_remainder(vx.y, vy.y)); }
__DEVICE__ inline HGB_Vec3 hgb_remainder_vec3(HGB_Vec3 vx, HGB_Vec3 vy) { return hgb_vec3(hgb_remainder(vx.x, vy.x), hgb_remainder(vx.y, vy.y), hgb_remainder(vx.z, vy.z)); }
__DEVICE__ inline HGB_Vec4 hgb_remainder_vec4(HGB_Vec4 vx, HGB_Vec4 vy) { return hgb_vec4(hgb_remainder(vx.x, vy.x), hgb_remainder(vx.y, vy.y), hgb_remainder(vx.z, vy.z), hgb_remainder(vx.w, vy.w)); }

__DEVICE__ inline hgb_f32  hgb_cos(hgb_f32 x) { return _cosf(x); }
__DEVICE__ inline HGB_Vec2 hgb_cos_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_cos(v.x), hgb_cos(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_cos_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_cos(v.x), hgb_cos(v.y), hgb_cos(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_cos_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_cos(v.x), hgb_cos(v.y), hgb_cos(v.z), hgb_cos(v.w)); }

__DEVICE__ inline hgb_f32  hgb_sin(hgb_f32 x) { return _sinf(x); }
__DEVICE__ inline HGB_Vec2 hgb_sin_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_sin(v.x), hgb_sin(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_sin_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_sin(v.x), hgb_sin(v.y), hgb_sin(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_sin_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_sin(v.x), hgb_sin(v.y), hgb_sin(v.z), hgb_sin(v.w)); }

__DEVICE__ inline hgb_f32  hgb_tan(hgb_f32 x) { return _tanf(x); }
__DEVICE__ inline HGB_Vec2 hgb_tan_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_tan(v.x), hgb_tan(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_tan_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_tan(v.x), hgb_tan(v.y), hgb_tan(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_tan_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_tan(v.x), hgb_tan(v.y), hgb_tan(v.z), hgb_tan(v.w)); }

__DEVICE__ inline hgb_f32  hgb_acos(hgb_f32 x) { return _acosf(x); }
__DEVICE__ inline HGB_Vec2 hgb_acos_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_acos(v.x), hgb_acos(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_acos_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_acos(v.x), hgb_acos(v.y), hgb_acos(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_acos_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_acos(v.x), hgb_acos(v.y), hgb_acos(v.z), hgb_acos(v.w)); }

__DEVICE__ inline hgb_f32  hgb_asin(hgb_f32 x) { return _asinf(x); }
__DEVICE__ inline HGB_Vec2 hgb_asin_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_asin(v.x), hgb_asin(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_asin_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_asin(v.x), hgb_asin(v.y), hgb_asin(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_asin_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_asin(v.x), hgb_asin(v.y), hgb_asin(v.z), hgb_asin(v.w)); }

__DEVICE__ inline hgb_f32  hgb_atan2(hgb_f32 y, hgb_f32 x) { return _atan2f(y, x); }
__DEVICE__ inline HGB_Vec2 hgb_atan2_vec2(HGB_Vec2 vy, HGB_Vec2 vx) { return hgb_vec2(hgb_atan2(vy.x, vx.x), hgb_atan2(vy.y, vx.y)); }
__DEVICE__ inline HGB_Vec3 hgb_atan2_vec3(HGB_Vec3 vy, HGB_Vec3 vx) { return hgb_vec3(hgb_atan2(vy.x, vx.x), hgb_atan2(vy.y, vx.y), hgb_atan2(vy.z, vx.z)); }
__DEVICE__ inline HGB_Vec4 hgb_atan2_vec4(HGB_Vec4 vy, HGB_Vec4 vx) { return hgb_vec4(hgb_atan2(vy.x, vx.x), hgb_atan2(vy.y, vx.y), hgb_atan2(vy.y, vx.y), hgb_atan2(vy.w, vx.w)); }

__DEVICE__ inline hgb_f32  hgb_acosh(hgb_f32 x) { return _acoshf(x); }
__DEVICE__ inline HGB_Vec2 hgb_acosh_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_acosh(v.x), hgb_acosh(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_acosh_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_acosh(v.x), hgb_acosh(v.y), hgb_acosh(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_acosh_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_acosh(v.x), hgb_acosh(v.y), hgb_acosh(v.z), hgb_acosh(v.w)); }

__DEVICE__ inline hgb_f32  hgb_asinh(hgb_f32 x) { return _asinhf(x); }
__DEVICE__ inline HGB_Vec2 hgb_asinh_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_asinh(v.x), hgb_asinh(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_asinh_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_asinh(v.x), hgb_asinh(v.y), hgb_asinh(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_asinh_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_asinh(v.x), hgb_asinh(v.y), hgb_asinh(v.z), hgb_asinh(v.w)); }

__DEVICE__ inline hgb_f32  hgb_atanh(hgb_f32 x) { return _atanhf(x); }
__DEVICE__ inline HGB_Vec2 hgb_atanh_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_atanh(v.x), hgb_atanh(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_atanh_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_atanh(v.x), hgb_atanh(v.y), hgb_atanh(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_atanh_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_atanh(v.x), hgb_atanh(v.y), hgb_atanh(v.z), hgb_atanh(v.w)); }

__DEVICE__ inline hgb_f32  hgb_cosh(hgb_f32 x) { return _coshf(x); }
__DEVICE__ inline HGB_Vec2 hgb_cosh_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_cosh(v.x), hgb_cosh(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_cosh_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_cosh(v.x), hgb_cosh(v.y), hgb_cosh(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_cosh_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_cosh(v.x), hgb_cosh(v.y), hgb_cosh(v.z), hgb_cosh(v.w)); }

__DEVICE__ inline hgb_f32  hgb_sinh(hgb_f32 x) { return _sinhf(x); }
__DEVICE__ inline HGB_Vec2 hgb_sinh_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_sinh(v.x), hgb_sinh(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_sinh_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_sinh(v.x), hgb_sinh(v.y), hgb_sinh(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_sinh_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_sinh(v.x), hgb_sinh(v.y), hgb_sinh(v.z), hgb_sinh(v.w)); }

__DEVICE__ inline hgb_f32  hgb_tanh(hgb_f32 x) { return _tanhf(x); }
__DEVICE__ inline HGB_Vec2 hgb_tanh_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_tanh(v.x), hgb_tanh(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_tanh_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_tanh(v.x), hgb_tanh(v.y), hgb_tanh(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_tanh_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_tanh(v.x), hgb_tanh(v.y), hgb_tanh(v.z), hgb_tanh(v.w)); }

#define hgb_hypot  hgb_length
#define hgb_hypot3 hgb_length3
__DEVICE__ inline hgb_f32 hgb_length(hgb_f32 x, hgb_f32 y) { return hgb_sqrt(hgb_square(x) + hgb_square(y)); }
__DEVICE__ inline hgb_f32 hgb_length3(hgb_f32 x, hgb_f32 y, hgb_f32 z) { return hgb_sqrt(hgb_square(x) + hgb_square(y) + hgb_square(z)); }
__DEVICE__ inline hgb_f32 hgb_length4(hgb_f32 x, hgb_f32 y, hgb_f32 z, hgb_f32 w) { return hgb_sqrt(hgb_square(x) + hgb_square(y) + hgb_square(z) + hgb_square(w)); }

#define hgb_hypot_vec2 hgb_length_vec2
#define hgb_hypot_vec3 hgb_length_vec3
__DEVICE__ inline hgb_f32 hgb_length_vec2(HGB_Vec2 v) { return hgb_length(v.x, v.y); }
__DEVICE__ inline hgb_f32 hgb_length_vec3(HGB_Vec3 v) { return hgb_length3(v.x, v.y, v.z); }
__DEVICE__ inline hgb_f32 hgb_length_vec4(HGB_Vec4 v) { return hgb_length4(v.x, v.y, v.z, v.w); }

__DEVICE__ inline hgb_f32  hgb_to_degrees(hgb_f32 radians) { return radians * 360.0f / HGB_TAU; }
__DEVICE__ inline HGB_Vec2 hgb_to_degrees_vec2(HGB_Vec2 radians) { return hgb_vec2(hgb_to_degrees(radians.x), hgb_to_degrees(radians.y)); }
__DEVICE__ inline HGB_Vec3 hgb_to_degrees_vec3(HGB_Vec3 radians) { return hgb_vec3(hgb_to_degrees(radians.x), hgb_to_degrees(radians.y), hgb_to_degrees(radians.z)); }
__DEVICE__ inline HGB_Vec4 hgb_to_degrees_vec4(HGB_Vec4 radians) { return hgb_vec4(hgb_to_degrees(radians.x), hgb_to_degrees(radians.y), hgb_to_degrees(radians.z), hgb_to_degrees(radians.w)); }

__DEVICE__ inline hgb_f32  hgb_to_radians(hgb_f32 degrees) { return degrees * HGB_TAU / 360.0f; }
__DEVICE__ inline HGB_Vec2 hgb_to_radians_vec2(HGB_Vec2 degrees) { return hgb_vec2(hgb_to_radians(degrees.x), hgb_to_radians(degrees.y)); }
__DEVICE__ inline HGB_Vec3 hgb_to_radians_vec3(HGB_Vec3 degrees) { return hgb_vec3(hgb_to_radians(degrees.x), hgb_to_radians(degrees.y), hgb_to_radians(degrees.z)); }
__DEVICE__ inline HGB_Vec4 hgb_to_radians_vec4(HGB_Vec4 degrees) { return hgb_vec4(hgb_to_radians(degrees.x), hgb_to_radians(degrees.y), hgb_to_radians(degrees.z), hgb_to_radians(degrees.w)); }

enum {
    HGB_Spow_Preserve,
    HGB_Spow_Clamp,
    HGB_Spow_Mirror,
};

__DEVICE__ inline hgb_f32 hgb_spow(hgb_f32 x, hgb_f32 p) {
    #if HGB_SPOW == HGB_Spow_Preserve
        if (x < 0.0f) {
            return x;
        }
        return hgb_pow(x, p);
    #elif HGB_SPOW == HGB_Spow_Clamp
        if (x < 0.0f) {
            return 0.0f;
        }
        return hgb_pow(x, p);
    #else
        return hgb_sign(x) * hgb_pow(hgb_abs(x), p);
    #endif
}
__DEVICE__ inline HGB_Vec2 hgb_spow_vec2(HGB_Vec2 v, hgb_f32 p) { return hgb_vec2(hgb_spow(v.x, p), hgb_spow(v.y, p)); }
__DEVICE__ inline HGB_Vec3 hgb_spow_vec3(HGB_Vec3 v, hgb_f32 p) { return hgb_vec3(hgb_spow(v.x, p), hgb_spow(v.y, p), hgb_spow(v.z, p)); }
__DEVICE__ inline HGB_Vec4 hgb_spow_vec4(HGB_Vec4 v, hgb_f32 p) { return hgb_vec4(hgb_spow(v.x, p), hgb_spow(v.y, p), hgb_spow(v.z, p), hgb_spow(v.w, p)); }

__DEVICE__ inline hgb_f32  hgb_fract(hgb_f32 x) { return x - hgb_floor(x); }
__DEVICE__ inline HGB_Vec2 hgb_fract_vec2(HGB_Vec2 v) { return hgb_vec2(hgb_fract(v.x), hgb_fract(v.y)); }
__DEVICE__ inline HGB_Vec3 hgb_fract_vec3(HGB_Vec3 v) { return hgb_vec3(hgb_fract(v.x), hgb_fract(v.y), hgb_fract(v.z)); }
__DEVICE__ inline HGB_Vec4 hgb_fract_vec4(HGB_Vec4 v) { return hgb_vec4(hgb_fract(v.x), hgb_fract(v.y), hgb_fract(v.z), hgb_fract(v.w)); }

__DEVICE__ inline hgb_f32  hgb_lerp(hgb_f32 a, hgb_f32 b, hgb_f32 t) { return (1.0f - t) * a + t * b; }
__DEVICE__ inline HGB_Vec2 hgb_lerp_vec2(HGB_Vec2 va, HGB_Vec2 vb, hgb_f32 t) { return hgb_vec2(hgb_lerp(va.x, vb.x, t), hgb_lerp(va.y, vb.y, t)); }
__DEVICE__ inline HGB_Vec3 hgb_lerp_vec3(HGB_Vec3 va, HGB_Vec3 vb, hgb_f32 t) { return hgb_vec3(hgb_lerp(va.x, vb.x, t), hgb_lerp(va.y, vb.y, t), hgb_lerp(va.z, vb.z, t)); }
__DEVICE__ inline HGB_Vec4 hgb_lerp_vec4(HGB_Vec4 va, HGB_Vec4 vb, hgb_f32 t) { return hgb_vec4(hgb_lerp(va.x, vb.x, t), hgb_lerp(va.y, vb.y, t), hgb_lerp(va.z, vb.z, t), hgb_lerp(va.w, vb.w, t)); }

__DEVICE__ inline hgb_f32  hgb_smoothstep(hgb_f32 a, hgb_f32 b, hgb_f32 x) { hgb_f32 t = (x - a) / (b - a); return hgb_square(t) * (3.0f - 2.0f * t); }
__DEVICE__ inline HGB_Vec2 hgb_smoothstep_vec2(HGB_Vec2 va, HGB_Vec2 vb, hgb_f32 x) { return hgb_vec2(hgb_smoothstep(va.x, vb.x, x), hgb_smoothstep(va.y, vb.y, x)); }
__DEVICE__ inline HGB_Vec3 hgb_smoothstep_vec3(HGB_Vec3 va, HGB_Vec3 vb, hgb_f32 x) { return hgb_vec3(hgb_smoothstep(va.x, vb.x, x), hgb_smoothstep(va.y, vb.y, x), hgb_smoothstep(va.z, vb.z, x)); }
__DEVICE__ inline HGB_Vec4 hgb_smoothstep_vec4(HGB_Vec4 va, HGB_Vec4 vb, hgb_f32 x) { return hgb_vec4(hgb_smoothstep(va.x, vb.x, x), hgb_smoothstep(va.y, vb.y, x), hgb_smoothstep(va.z, vb.z, x), hgb_smoothstep(va.w, vb.w, x)); }

__DEVICE__ inline hgb_f32  hgb_smootherstep(hgb_f32 a, hgb_f32 b, hgb_f32 x) { hgb_f32 t = (x - a) / (b - a); return hgb_cube(t) * (t * (6.0f * t - 15.0f) + 10.0f); }
__DEVICE__ inline HGB_Vec2 hgb_smootherstep_vec2(HGB_Vec2 va, HGB_Vec2 vb, hgb_f32 x) { return hgb_vec2(hgb_smootherstep(va.x, vb.x, x), hgb_smootherstep(va.y, vb.y, x)); }
__DEVICE__ inline HGB_Vec3 hgb_smootherstep_vec3(HGB_Vec3 va, HGB_Vec3 vb, hgb_f32 x) { return hgb_vec3(hgb_smootherstep(va.x, vb.x, x), hgb_smootherstep(va.y, vb.y, x), hgb_smootherstep(va.z, vb.z, x)); }
__DEVICE__ inline HGB_Vec4 hgb_smootherstep_vec4(HGB_Vec4 va, HGB_Vec4 vb, hgb_f32 x) { return hgb_vec4(hgb_smootherstep(va.x, vb.x, x), hgb_smootherstep(va.y, vb.y, x), hgb_smootherstep(va.z, vb.z, x), hgb_smootherstep(va.w, vb.w, x)); }

__DEVICE__ inline hgb_f32  hgb_step(hgb_f32 edge, hgb_f32 x) { if (x < edge) { return 0.0f; } return 1.0f; }
__DEVICE__ inline HGB_Vec2 hgb_step_vec2(HGB_Vec2 v, hgb_f32 x) { return hgb_vec2(hgb_step(v.x, x), hgb_step(v.y, x)); }
__DEVICE__ inline HGB_Vec3 hgb_step_vec3(HGB_Vec3 v, hgb_f32 x) { return hgb_vec3(hgb_step(v.x, x), hgb_step(v.y, x), hgb_step(v.z, x)); }
__DEVICE__ inline HGB_Vec4 hgb_step_vec4(HGB_Vec4 v, hgb_f32 x) { return hgb_vec4(hgb_step(v.x, x), hgb_step(v.y, x), hgb_step(v.z, x), hgb_step(v.w, x)); }

__DEVICE__ hgb_f32 hgb_dot_vec2(HGB_Vec2 a, HGB_Vec2 b) { return a.x * b.x + a.y * b.y; }
__DEVICE__ hgb_f32 hgb_dot_vec3(HGB_Vec3 a, HGB_Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__DEVICE__ hgb_f32 hgb_dot_vec4(HGB_Vec4 a, HGB_Vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

__DEVICE__ hgb_f32 hgb_cross_vec2(HGB_Vec2 a, HGB_Vec2 b) {
    return a.x * b.y - b.x * a.y;
}
__DEVICE__ HGB_Vec3 hgb_cross_vec3(HGB_Vec3 a, HGB_Vec3 b) {
    return hgb_vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

typedef struct HGB_Mat2 HGB_Mat2;
struct HGB_Mat2 {
    hgb_f32 v[2][2];
};

typedef struct HGB_Mat3 HGB_Mat3;
struct HGB_Mat3 {
    hgb_f32 v[3][3];
};

typedef struct HGB_Mat4 HGB_Mat4;
struct HGB_Mat4 {
    hgb_f32 v[4][4];
};

__DEVICE__ inline HGB_Mat2 hgb_mat2(
    hgb_f32 v00, hgb_f32 v01,
    hgb_f32 v10, hgb_f32 v11
) {
    HGB_Mat2 out = {{
        {v00, v01},
        {v10, v11}
    }};
    return out;
}

__DEVICE__ inline HGB_Mat3 hgb_mat3(
    hgb_f32 v00, hgb_f32 v01, hgb_f32 v02,
    hgb_f32 v10, hgb_f32 v11, hgb_f32 v12,
    hgb_f32 v20, hgb_f32 v21, hgb_f32 v22
) {
    HGB_Mat3 out = {{
        {v00, v01, v02},
        {v10, v11, v12},
        {v20, v21, v22}
    }};
    return out;
}

__DEVICE__ inline HGB_Mat4 hgb_mat4(
    hgb_f32 v00, hgb_f32 v01, hgb_f32 v02, hgb_f32 v03,
    hgb_f32 v10, hgb_f32 v11, hgb_f32 v12, hgb_f32 v13,
    hgb_f32 v20, hgb_f32 v21, hgb_f32 v22, hgb_f32 v23,
    hgb_f32 v30, hgb_f32 v31, hgb_f32 v32, hgb_f32 v33
) {
    HGB_Mat4 out = {{
        {v00, v01, v02, v03},
        {v10, v11, v12, v13},
        {v20, v21, v22, v23},
        {v30, v31, v32, v33}
    }};
    return out;
}

__DEVICE__ inline HGB_Mat2 hgb_repeat_mat2(hgb_f32 v) {
    return hgb_mat2(
        v, v,
        v, v
    );
}

__DEVICE__ inline HGB_Mat3 hgb_repeat_mat3(hgb_f32 v) {
    return hgb_mat3(
        v, v, v,
        v, v, v,
        v, v, v
    );
}

__DEVICE__ inline HGB_Mat4 hgb_repeat_mat4(hgb_f32 v) {
    return hgb_mat4(
        v, v, v, v,
        v, v, v, v,
        v, v, v, v,
        v, v, v, v
    );
}

__DEVICE__ inline HGB_Mat2 hgb_zeros_mat2() { return hgb_repeat_mat2(0.0f); }
__DEVICE__ inline HGB_Mat3 hgb_zeros_mat3() { return hgb_repeat_mat3(0.0f); }
__DEVICE__ inline HGB_Mat4 hgb_zeros_mat4() { return hgb_repeat_mat4(0.0f); }

__DEVICE__ inline HGB_Mat2 hgb_identity_mat2() {
    return hgb_mat2(
        1.0f, 0.0f,
        0.0f, 1.0f
    );
}
__DEVICE__ inline HGB_Mat3 hgb_identity_mat3() {
    return hgb_mat3(
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    );
}
__DEVICE__ inline HGB_Mat4 hgb_identity_mat4() {
    return hgb_mat4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

__DEVICE__ inline HGB_Mat2 hgb_transpose_mat2(HGB_Mat2 m) {
    for (hgb_usize j = 0; j < 2; j += 1) {
        for (hgb_usize i = j + 1; i < 2; i += 1) {
            hgb_f32 t = m.v[i][j];
            m.v[i][j] = m.v[j][i];
            m.v[j][i] = t;
        }
    }
    return m;
}
__DEVICE__ inline HGB_Mat3 hgb_transpose_mat3(HGB_Mat3 m) {
    for (hgb_usize j = 0; j < 3; j += 1) {
        for (hgb_usize i = j + 1; i < 3; i += 1) {
            hgb_f32 t = m.v[i][j];
            m.v[i][j] = m.v[j][i];
            m.v[j][i] = t;
        }
    }
    return m;
}
__DEVICE__ inline HGB_Mat4 hgb_transpose_mat4(HGB_Mat4 m) {
    for (hgb_usize j = 0; j < 4; j += 1) {
        for (hgb_usize i = j + 1; i < 4; i += 1) {
            hgb_f32 t = m.v[i][j];
            m.v[i][j] = m.v[j][i];
            m.v[j][i] = t;
        }
    }
    return m;
}

__DEVICE__ inline void hgb_transpose_mut_mat2(HGB_Mat2 *m) {
    for (hgb_usize j = 0; j < 2; j += 1) {
        for (hgb_usize i = j + 1; i < 2; i += 1) {
            hgb_f32 t = m->v[i][j];
            m->v[i][j] = m->v[j][i];
            m->v[j][i] = t;
        }
    }
}
__DEVICE__ inline void hgb_transpose_mut_mat3(HGB_Mat3 *m) {
    for (hgb_usize j = 0; j < 3; j += 1) {
        for (hgb_usize i = j + 1; i < 3; i += 1) {
            hgb_f32 t = m->v[i][j];
            m->v[i][j] = m->v[j][i];
            m->v[j][i] = t;
        }
    }
}
__DEVICE__ inline void hgb_transpose_mut_mat4(HGB_Mat4 *m) {
    for (hgb_usize j = 0; j < 4; j += 1) {
        for (hgb_usize i = j + 1; i < 4; i += 1) {
            hgb_f32 t = m->v[i][j];
            m->v[i][j] = m->v[j][i];
            m->v[j][i] = t;
        }
    }
}

__DEVICE__ hgb_f32 hgb_determinant_mat2(HGB_Mat2 m) {
    return m.v[0][0] * m.v[1][1] - m.v[1][0] * m.v[0][1];
}
__DEVICE__ hgb_f32 hgb_determinant_mat3(HGB_Mat3 m) {
    return m.v[0][0] * (m.v[1][1] * m.v[2][2] - m.v[1][2] * m.v[2][1])
         - m.v[0][1] * (m.v[1][0] * m.v[2][2] - m.v[1][2] * m.v[2][0])
         + m.v[0][2] * (m.v[1][0] * m.v[2][1] - m.v[1][1] * m.v[2][0]);
}

__DEVICE__ bool hgb_inverse_mat3(HGB_Mat3 m, HGB_Mat3 *out) {
    hgb_f32 d = hgb_determinant_mat3(m);
    if (d == 0.0f) {
        return false;
    }
    hgb_f32 ood = 1.0f / d;

    out->v[0][0] = (m.v[1][1] * m.v[2][2] - m.v[2][1] * m.v[1][2]) * ood;
    out->v[0][1] = (m.v[0][2] * m.v[2][1] - m.v[0][1] * m.v[2][2]) * ood;
    out->v[0][2] = (m.v[0][1] * m.v[1][2] - m.v[0][2] * m.v[1][1]) * ood;
    out->v[1][0] = (m.v[1][2] * m.v[2][0] - m.v[1][0] * m.v[2][2]) * ood;
    out->v[1][1] = (m.v[0][0] * m.v[2][2] - m.v[0][2] * m.v[2][0]) * ood;
    out->v[1][2] = (m.v[1][0] * m.v[0][2] - m.v[0][0] * m.v[1][2]) * ood;
    out->v[2][0] = (m.v[1][0] * m.v[2][1] - m.v[2][0] * m.v[1][1]) * ood;
    out->v[2][1] = (m.v[2][0] * m.v[0][1] - m.v[0][0] * m.v[2][1]) * ood;
    out->v[2][2] = (m.v[0][0] * m.v[1][1] - m.v[1][0] * m.v[0][1]) * ood;

    return true;
}

__DEVICE__ HGB_Mat3 hgb_mul_mat3(HGB_Mat3 a, HGB_Mat3 b) {
    HGB_Mat3 res = hgb_zeros_mat3();

    res.v[0][0] = (b.v[0][0] * a.v[0][0]) + (b.v[0][1] * a.v[1][0]) + (b.v[0][2] * a.v[2][0]);
    res.v[0][1] = (b.v[0][0] * a.v[0][1]) + (b.v[0][1] * a.v[1][1]) + (b.v[0][2] * a.v[2][1]);
    res.v[0][2] = (b.v[0][0] * a.v[0][2]) + (b.v[0][1] * a.v[1][2]) + (b.v[0][2] * a.v[2][2]);

    res.v[1][0] = (b.v[1][0] * a.v[0][0]) + (b.v[1][1] * a.v[1][0]) + (b.v[1][2] * a.v[2][0]);
    res.v[1][1] = (b.v[1][0] * a.v[0][1]) + (b.v[1][1] * a.v[1][1]) + (b.v[1][2] * a.v[2][1]);
    res.v[1][2] = (b.v[1][0] * a.v[0][2]) + (b.v[1][1] * a.v[1][2]) + (b.v[1][2] * a.v[2][2]);

    res.v[2][0] = (b.v[2][0] * a.v[0][0]) + (b.v[2][1] * a.v[1][0]) + (b.v[2][2] * a.v[2][0]);
    res.v[2][1] = (b.v[2][0] * a.v[0][1]) + (b.v[2][1] * a.v[1][1]) + (b.v[2][2] * a.v[2][1]);
    res.v[2][2] = (b.v[2][0] * a.v[0][2]) + (b.v[2][1] * a.v[1][2]) + (b.v[2][2] * a.v[2][2]);

    return res;
}

__DEVICE__ inline HGB_Vec3 hgb_mul_mat3_vec3(HGB_Mat3 m, HGB_Vec3 v) {
    return hgb_vec3(
        v.x * m.v[0][0] + v.y * m.v[0][1] + v.z * m.v[0][2],
        v.x * m.v[1][0] + v.y * m.v[1][1] + v.z * m.v[1][2],
        v.x * m.v[2][0] + v.y * m.v[2][1] + v.z * m.v[2][2]
    );
}

typedef struct HGB_Stack HGB_Stack;
struct HGB_Stack {
    __PRIVATE__ hgb_byte *data; // ?
    hgb_usize size;
    hgb_usize offset;
};

__DEVICE__ void hgb_stack_init(__PRIVATE__ HGB_Stack *s, __PRIVATE__ hgb_byte *backing, hgb_usize size) {
    s->data = backing;
    s->size = size;
    s->offset = 0;
}

__DEVICE__ void *hgb_stack_alloc(__PRIVATE__ HGB_Stack *s, hgb_usize amount) {
    void *ptr = s->data + s->offset;
    s->offset += amount;
    return ptr;
}

__DEVICE__ void hgb_stack_free_all(__PRIVATE__ HGB_Stack *s) {
    s->offset = 0;
}

__DEVICE__ void hgb_stack_free(__PRIVATE__ HGB_Stack *s, __PRIVATE__ void *ptr) {
    s->offset = cast(hgb_usize)ptr - cast(hgb_usize)s->data;
}

typedef struct HGB_Arena HGB_Arena;
struct HGB_Arena {
    __PRIVATE__ hgb_byte *data; // TODO
    hgb_usize size;
    hgb_usize offset;
};

__DEVICE__ void hgb_arena_init(__PRIVATE__ HGB_Arena *a, __PRIVATE__ hgb_byte *backing, hgb_usize size) {
    a->data = backing;
    a->size = size;
    a->offset = 0;
}

__DEVICE__ void *hgb_arena_alloc(__PRIVATE__ HGB_Arena *a, hgb_usize amount) {
    hgb_usize total = a->offset + amount;
    if (total <= a->size) {
        void *ptr = a->data + a->offset;
        a->offset = total;
        return ptr;
    }
    return nil;
}

__DEVICE__ void hgb_arena_free_all(__PRIVATE__ HGB_Arena *a) {
    a->offset = 0;
}

typedef struct HGB_Temp_Arena HGB_Temp_Arena;
struct HGB_Temp_Arena {
    __PRIVATE__ HGB_Arena *arena;
    hgb_usize old_offset;
};

__DEVICE__ HGB_Temp_Arena hgb_temp_arena_begin(__PRIVATE__ HGB_Arena *a) {
    HGB_Temp_Arena temp;
    temp.arena = a;
    temp.old_offset = a->offset;
    return temp;
}

__DEVICE__ void hgb_temp_arena_end(HGB_Temp_Arena temp) {
    temp.arena->offset = temp.old_offset;
}

typedef struct HGB_Linspace HGB_Linspace;
struct HGB_Linspace {
    hgb_f32 delta;
    hgb_f32 start;
    hgb_usize steps;
    hgb_usize current;
    bool done;
};

__DEVICE__ HGB_Linspace hgb_linspace_create(hgb_f32 start, hgb_f32 end, hgb_usize steps) {
    HGB_Linspace it;
    it.delta = (end - start) / cast(hgb_f32)(steps - 1);
    it.start = start;
    it.steps = steps;
    it.current = 0;
    it.done = false;
    return it;
}

__DEVICE__ hgb_f32 hgb_linspace_next(__PRIVATE__ HGB_Linspace *it) {
    if (it->current == it->steps - 1) {
        it->done = true;
    }
    hgb_f32 res = cast(hgb_f32)it->current * it->delta + it->start;
    it->current += 1;
    return res;
}

__DEVICE__ hgb_f32 *hgb_linspace_allocate(__PRIVATE__ HGB_Arena *arena, HGB_Linspace it) {
    hgb_f32 *array = (hgb_f32 *)hgb_arena_alloc(arena, sizeof(hgb_f32) * it.steps);
    if (array == nil) {
        return nil;
    }
    hgb_usize i = 0;
    while (!it.done) {
        array[i] = hgb_linspace_next(&it);
        i += 1;
    }
    return array;
}


__DEVICE__ hgb_usize _hgb_find_interval(hgb_f32 *points, hgb_usize n_pts, hgb_f32 x) {
    hgb_usize lower = 0;
    hgb_usize upper = n_pts - 1;
    while (lower != upper - 1) {
        hgb_usize center = lower + (upper - lower) / 2;
        if (x >= points[center]) {
            lower = center;
        } else {
            upper = center;
        }
    }
    return lower;
}

__DEVICE__ bool _hgb_handle_beyond_range(
    hgb_f32 x0, hgb_f32 xn_1,
    hgb_f32 y0, hgb_f32 yn_1,
    hgb_f32 m0, hgb_f32 mn_1,
    hgb_f32 x,
    bool extrapolate,
    hgb_f32 *res
) {
    if (x <= x0) {
        if (extrapolate) {
            *res = (x - x0) * m0 + y0;
        } else {
            *res = y0;
        }
        return true;
    }
    if (x >= xn_1) {
        if (extrapolate) {
            *res = (x - xn_1) * mn_1 + yn_1;
        } else {
            *res = yn_1;
        }
        return true;
    }
    return false;
}

typedef enum {
    HGB_Spline_End_Natural,
    HGB_Spline_End_Parabolic,
    HGB_Spline_End_Slope,
    HGB_Spline_End_Inner,
} HGB_Spline_End_Condition;

__DEVICE__ hgb_f32 _hgb_end_tangent(hgb_f32 x0, hgb_f32 x1, hgb_f32 y0, hgb_f32 y1, hgb_f32 m, HGB_Spline_End_Condition condition) {
    hgb_f32 res;
    switch (condition) {
        case HGB_Spline_End_Natural:   res = 3.0f * (y1 - y0) / (2.0f * (x1 - x0)) - m / 2.0f; break;
        case HGB_Spline_End_Parabolic: res = 2.0f * (y1 - y0) / (x1 - x0) - m; break;
        case HGB_Spline_End_Slope:     res = (y1 - y0) / (x1 - x0); break;
        case HGB_Spline_End_Inner:     res = m; break;
    }
    return res;
}

typedef enum {
    // Previous--next secant-line tangents. No tension parameter as it causes ripples
    // at any value other than `0.0` in 1D.
    //
    // https://www.youtube.com/watch?v=UCtmRJs726U
    // https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Cardinal_spline
    HGB_Hermite_Method_Cardinal,

    // https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Finite_difference
    HGB_Hermite_Method_Mean_Velocity,

    // Correctly derived non-uniform Catmull--Rom tangents.
    //
    // https://splines.readthedocs.io/en/latest/euclidean/catmull-rom-properties.html
    HGB_Hermite_Method_Catmull_Rom,

    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
    HGB_Hermite_Method_Pchip,

    // https://en.wikipedia.org/wiki/Akima_spline
    HGB_Hermite_Method_Akima,
} HGB_Hermite_Method;

typedef struct HGB_Cubic_Coeff HGB_Cubic_Coeff;
struct HGB_Cubic_Coeff {
    hgb_f32 a, b, c, d;
};

typedef struct HGB_Spline_Hermite HGB_Spline_Hermite;
struct HGB_Spline_Hermite {
    hgb_f32 *centers;
    hgb_f32 *values;
    HGB_Cubic_Coeff *coeff;
    hgb_usize n_pts;
    hgb_f32 *end_tangents;
    bool extrapolate;
};

// A Cubic Hermite Spline built with the auto-tangents `method`.
//
// https://en.wikipedia.org/wiki/Cubic_Hermite_spline
//
// UNCHECKED REQUIREMENTS:
//      - len(centers) == len(values)
//      - len(centers) == n_pts
//      - n_pts >= 4
__DEVICE__ HGB_Spline_Hermite hgb_spline_build_hermite(
    __PRIVATE__ HGB_Arena *arena,
    hgb_f32 *centers,
    hgb_f32 *values,
    hgb_usize n_pts,
    HGB_Hermite_Method method,
    HGB_Spline_End_Condition ends,
    bool extrapolate
) {
    HGB_Temp_Arena tmp = hgb_temp_arena_begin(arena);
    hgb_f32 *tangents = cast(hgb_f32 *)hgb_arena_alloc(tmp.arena, sizeof(hgb_f32) * n_pts);

    switch (method) {
        case HGB_Hermite_Method_Cardinal:
            for_range(i, 1, n_pts - 1) {
                tangents[i] = (values[i+1] - values[i-1]) / (centers[i+1] - centers[i-1]);
            }
            break;
        case HGB_Hermite_Method_Mean_Velocity:
            for_range(i, 1, n_pts - 1) {
                hgb_f32 v_1 = (values[i] - values[i-1]) / (centers[i] - centers[i-1]);
                hgb_f32 v0 = (values[i+1] - values[i]) / (centers[i+1] - centers[i]);
                tangents[i] = (v_1 + v0) / 2.0f;
            }
            break;
        case HGB_Hermite_Method_Catmull_Rom:
            for_range(i, 1, n_pts - 1) {
                hgb_f32 delta_1 = centers[i] - centers[i-1];
                hgb_f32 delta0 = centers[i+1] - centers[i];

                hgb_f32 v_1 = (values[i] - values[i-1]) / delta_1;
                hgb_f32 v0 = (values[i+1] - values[i]) / delta0;

                tangents[i] = (delta0 * v_1 + delta_1 * v0) / (delta0 + delta_1);
            }
            break;
        case HGB_Hermite_Method_Pchip:
            for_range(i, 1, n_pts - 1) {
                hgb_f32 delta_1 = centers[i] - centers[i-1];
                hgb_f32 delta0 = centers[i+1] - centers[i];

                hgb_f32 v_1 = (values[i] - values[i-1]) / delta_1;
                hgb_f32 v0 = (values[i+1] - values[i]) / delta0;

                hgb_f32 wl = 2.0f * delta0 + delta_1;
                hgb_f32 wr = delta0 + 2.0f * delta_1;

                tangents[i] = (wl + wr) / (wl / v_1 + wr / v0);
            }
            break;
        case HGB_Hermite_Method_Akima: {
            hgb_f32 *weights = cast(hgb_f32 *)hgb_arena_alloc(tmp.arena, sizeof(hgb_f32) * (n_pts - 1));

            for_range(i, 0, n_pts - 1) {
                weights[i] = (values[i+1] - values[i]) / (centers[i+1] - centers[i]);
            }

            // Second and last intervals.
            tangents[1] = (weights[0] + weights[1]) / 2.0f;
            tangents[n_pts-2] = (weights[n_pts-3] + weights[n_pts-2]) / 2.0f;

            for_range(i, 2, n_pts - 2) {
                hgb_f32 mn =
                    hgb_abs(weights[i+1] - weights[i]) * weights[i-1]
                  + hgb_abs(weights[i-1] - weights[i-2]) * weights[i];

                hgb_f32 md =
                    hgb_abs(weights[i+1] - weights[i])
                  + hgb_abs(weights[i-1] - weights[i-2]);

                if (md == 0.0f) {
                    tangents[i] = (weights[i-1] + weights[i]) / 2.0f;
                } else {
                    tangents[i] = mn / md;
                }
            }
            break;
        }
    }

    hgb_temp_arena_end(tmp);

    // First and last points.
    hgb_f32 *end_tangents = cast(hgb_f32 *)hgb_arena_alloc(arena, sizeof(hgb_f32) * 2);
    end_tangents[0] = _hgb_end_tangent(centers[0], centers[1], values[0], values[1], tangents[1], ends);
    end_tangents[1] = _hgb_end_tangent(centers[n_pts-2], centers[n_pts-1], values[n_pts-2], values[n_pts-1], tangents[n_pts-2], ends);

    tangents[0] = end_tangents[0];
    tangents[n_pts-1] = end_tangents[1];

    HGB_Cubic_Coeff *coeff = cast(HGB_Cubic_Coeff *)hgb_arena_alloc(arena, sizeof(HGB_Cubic_Coeff) * (n_pts - 1));

    for_range(i, 0, n_pts - 1) {
        hgb_f32 delta = centers[i+1] - centers[i];
        coeff[i].a = (2.0 * values[i] + tangents[i] * delta - 2.0 * values[i+1] + delta * tangents[i+1]);
        coeff[i].b = (-3.0 * values[i] + 3.0 * values[i+1] - 2.0 * delta * tangents[i] - delta * tangents[i+1]);
        coeff[i].c = delta * tangents[i];
        coeff[i].d = values[i];
    }

    HGB_Spline_Hermite res = {
        centers,
        values,
        coeff,
        n_pts,
        end_tangents,
        extrapolate
    };

    return res;
}

__DEVICE__ hgb_f32 hgb_spline_eval_hermite(HGB_Spline_Hermite *s, hgb_f32 x) {
    hgb_f32 beyond_val;
    bool is_beyond = _hgb_handle_beyond_range(
        s->centers[0],
        s->centers[s->n_pts-1],
        s->values[0],
        s->values[s->n_pts-1],
        s->end_tangents[0],
        s->end_tangents[1],
        x,
        s->extrapolate,
        &beyond_val
    );

    if (is_beyond) {
        return beyond_val;
    }

    hgb_usize i = _hgb_find_interval(s->centers, s->n_pts, x);
    hgb_f32 t = (x - s->centers[i]) / (s->centers[i+1] - s->centers[i]);

    return s->coeff[i].a * hgb_pow(t, 3.0f) + s->coeff[i].b * hgb_pow(t, 2.0f) + s->coeff[i].c * t + s->coeff[i].d;
}


__DEVICE__ inline hgb_f32 hgb_basis_gaussian(hgb_f32 x, hgb_f32 size) {
    return hgb_exp(hgb_pow(x / size, 2.0f));
}

__DEVICE__ inline hgb_f32 hgb_basis_tent(hgb_f32 x, hgb_f32 size) {
    return hgb_max(1.0f - hgb_abs(x / size), 0.0f);
}

__DEVICE__ inline hgb_f32 hgb_basis_exponential(hgb_f32 x, hgb_f32 size) {
    return hgb_exp(-hgb_abs(x / size));
}

// Default to knee = 1.0f.
__DEVICE__ inline hgb_f32 hgb_basis_falloff(hgb_f32 x, hgb_f32 size, hgb_f32 knee) {
    if (x == 0.0f) {
        return 1.0f;
    }
    return hgb_pow(1.0f / hgb_pow(x / size, 2.0f), knee);
}

__DEVICE__ inline hgb_f32 hgb_basis_boxcar(hgb_f32 x, hgb_f32 size) {
    if (x < -size || x > size) {
        return 0.0f;
    }
    return 1.0f;
}

__DEVICE__ inline hgb_f32 hgb_linear(hgb_f32 x) {
    return x;
}

// EI 800. TODO
__DEVICE__ hgb_f32 hgb_arri_logc_encode(hgb_f32 x) {
    if (x > 0.010591f) {
        return 0.247190f * hgb_log10(5.555556f * x + 0.052272f) + 0.385537f;
    }
    return 5.367655f * x + 0.092809f;
}

__DEVICE__ hgb_f32 hgb_arri_logc_decode(hgb_f32 x) {
    if (x > 0.1496582f){
        return (hgb_pow(10.0f, (x - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f;
    }
    return (x - 0.092809f) / 5.367655f;
}

__DEVICE__ hgb_f32 hgb_fuji_flog_encode(hgb_f32 x) {
    if (x >= 0.00089f) {
        return 0.344676f * _log10f(0.555556f * x + 0.009468f) + 0.790453f;
    }
    return 8.735631f * x + 0.092864f;
}

__DEVICE__ hgb_f32 hgb_fuji_flog_decode(hgb_f32 x) {
    if (x >= 0.100537775223865f) {
        return (_powf(10.0f, (x - 0.790453f) / 0.344676f)) / 0.555556f - 0.009468f / 0.555556f;
    }
    return (x - 0.092864f) / 8.735631f;
}

__DEVICE__ hgb_f32 hgb_nikon_nlog_encode(hgb_f32 x) {
    if (x > 0.328f) {
        return (150.0f / 1023.0f) * _logf(x) + (619.0f / 1023.0f);
    }
    return (650.0f / 1023.0f) * _powf((x + 0.0075f), 1.0f / 3.0f);
}

__DEVICE__ hgb_f32 hgb_nikon_nlog_decode(hgb_f32 x) {
    if (x > (452.0f / 1023.0f)){
        return _expf((x - 619.0f / 1023.0f) / (150.0f / 1023.0f));
    }
    return _powf(x / (650.0f / 1023.0f), 3.0f) - 0.0075f;
}

__DEVICE__ hgb_f32 hgb_panasonic_vlog_encode(hgb_f32 x) {
    if (x < 0.01f) {
        return 5.6f * x + 0.125f;
    }
    return 0.241514f * _log10f(x + 0.00873f) + 0.598206f;
}

__DEVICE__ hgb_f32 hgb_panasonic_vlog_decode(hgb_f32 x) {
    if (x < 0.181f) {
        return (x - 0.125f) / 5.6f;
    }
    return _powf(10.0f, ((x - 0.598206f) / 0.241514f)) - 0.00873f;
}

__DEVICE__ hgb_f32 hgb_sony_slog3_encode(hgb_f32 x) {
    if (x >= 0.01125000f) {
        return (420.0f + _log10f((x + 0.01f) / (0.18f + 0.01f)) * 261.5f) / 1023.0f;
    }
    return (x * (171.2102946929f - 95.0f) / 0.01125000f + 95.0f) / 1023.0f;
}

__DEVICE__ hgb_f32 hgb_sony_slog3_decode(hgb_f32 x) {
    if (x >= (171.2102946929f / 1023.0f)) {
        return _powf(10.0f, (x * 1023.0f - 420.0f) / 261.5f) * (0.18f + 0.01f) - 0.01f;
    }
    return (x * 1023.0f - 95.0f) * 0.01125000f / (171.2102946929f - 95.0f);
}

typedef struct HGB_Primaries_Whitepoint HGB_Primaries_Whitepoint;
struct HGB_Primaries_Whitepoint {
    HGB_Vec2 r;
    HGB_Vec2 g;
    HGB_Vec2 b;
    HGB_Vec2 w;
};

__DEVICE__ HGB_Mat3 hgb_npm(HGB_Primaries_Whitepoint pw) {
    hgb_f32 y = 1.0f;
    hgb_f32 x = pw.w.x * y / pw.w.y;
    hgb_f32 z = (1.0f - pw.w.x - pw.w.y) * y / pw.w.y;

    hgb_f32 d = pw.r.x * (pw.b.y - pw.g.y)
        + pw.b.x * (pw.g.y - pw.r.y)
        + pw.g.x * (pw.r.y - pw.b.y);

    hgb_f32 sr = (x * (pw.b.y - pw.g.y)
        - pw.g.x * (y * (pw.b.y - 1.0f) + pw.b.y * (x + z))
        + pw.b.x * (y * (pw.g.y - 1.0f) + pw.g.y * (x + z)))
        / d;
    hgb_f32 sg = (x * (pw.r.y - pw.b.y)
        + pw.r.x * (y * (pw.b.y - 1.0f) + pw.b.y * (x + z))
        - pw.b.x * (y * (pw.r.y - 1.0f) + pw.r.y * (x + z)))
        / d;
    hgb_f32 sb = (x * (pw.g.y - pw.r.y)
        - pw.r.x * (y * (pw.g.y - 1.0f) + pw.g.y * (x + z))
        + pw.g.x * (y * (pw.r.y - 1.0f) + pw.r.y * (x + z)))
        / d;

    return hgb_mat3(
        sr * pw.r.x,
        sg * pw.g.x,
        sb * pw.b.x,
        sr * pw.r.y,
        sg * pw.g.y,
        sb * pw.b.y,
        sr * (1.0f - pw.r.x - pw.r.y),
        sg * (1.0f - pw.g.x - pw.g.y),
        sb * (1.0f - pw.b.x - pw.b.y)
    );
}

__CONSTANT__ HGB_Primaries_Whitepoint HGB_BT_709 = {
    {0.640, 0.330},
    {0.300, 0.600},
    {0.150, 0.060},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_BT_2020 = {
    {0.708, 0.292},
    {0.170, 0.797},
    {0.131, 0.046},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_DCI_P3 = {
    {0.680, 0.320},
    {0.265, 0.690},
    {0.150, 0.060},
    {0.314, 0.351}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_DISPLAY_P3 = {
    {0.680, 0.320},
    {0.265, 0.690},
    {0.150, 0.060},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_ACES_AP0 = {
    {0.73470, 0.26530},
    {0.00000, 1.00000},
    {0.00010, -0.07700},
    {0.32168, 0.33767}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_ACES_AP1 = {
    {0.71300, 0.29300},
    {0.16500, 0.83000},
    {0.12800, 0.04400},
    {0.32168, 0.33767}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_ADOBE_RGB = {
    {0.6400, 0.3300},
    {0.2100, 0.7100},
    {0.1500, 0.0600},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_ADOBE_WIDE_GAMUT_RGB = {
    {0.7347, 0.2653},
    {0.1152, 0.8264},
    {0.1566, 0.0177},
    {0.3457, 0.3585}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_ARRI_WIDE_GAMUT_3 = {
    {0.6840, 0.3130},
    {0.2210, 0.8480},
    {0.0861, -0.1020},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_ARRI_WIDE_GAMUT_4 = {
    {0.7347, 0.2653},
    {0.1424, 0.8576},
    {0.0991, -0.0308},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_CANON_CINEMA_GAMUT = {
    {0.7400, 0.2700},
    {0.1700, 1.1400},
    {0.0800, -0.1000},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_DJI_D_GAMUT = {
    {0.7100, 0.3100},
    {0.2100, 0.8800},
    {0.0900, -0.0800},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_E_GAMUT = {
    {0.8000, 0.3177},
    {0.1800, 0.9000},
    {0.0650, -0.0805},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_PANASONIC_V_GAMUT = {
    {0.7300, 0.2800},
    {0.1650, 0.8400},
    {0.1000, -0.0300},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_PROPHOTO = {
    {0.734699, 0.265301},
    {0.159597, 0.840403},
    {0.036598, 0.000105},
    {0.345704, 0.358540}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_RED_WIDE_GAMUT_RGB = {
    {0.780308, 0.304253},
    {0.121595, 1.493994},
    {0.095612, -0.084589},
    {0.3127, 0.3290}
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_S_GAMUT = {
    {0.7300, 0.2800},
    {0.1400, 0.8550},
    {0.1000, -0.0500},
    {0.3127, 0.3290},
};

__CONSTANT__ HGB_Primaries_Whitepoint HGB_S_GAMUT3_CINE = {
    {0.7660, 0.2750},
    {0.2250, 0.8000},
    {0.0890, -0.0870},
    {0.3127, 0.3290},
};
