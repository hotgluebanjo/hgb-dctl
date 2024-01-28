// HGB DCTL library 0.1.0
// ----------------------
// Attempting to be compatible with all DCTL targets.
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
// VecN type.

#if defined(__APPLE__) || defined(__MACOSX)
    #define HGB_IS_MAC
#endif

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
    #define HGB_IS_WINDOWS
#endif

#if defined(linux) || defined(__linux__)
    #define HGB_IS_LINUX
#endif

#ifdef DEVICE_IS_METAL
    #define HGB_IS_METAL
#endif

#ifdef DEVICE_IS_CUDA
    #define HGB_IS_CUDA
#endif

#ifdef DEVICE_IS_OPENCL
    #define HGB_IS_OPENCL
#endif

// Some, uh, assumptions. stdint is not included.
typedef unsigned char   u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long  u64;

typedef signed char   i8;
typedef signed short i16;
typedef signed int   i32;
typedef signed long  i64;

typedef size_t    usize;
typedef ptrdiff_t isize;

typedef float f32;

typedef u8 byte;

typedef float2 Vec2;
typedef float3 Vec3;
typedef float4 Vec4;

typedef float Mat2[2][2];
typedef float Mat3[3][3];
typedef float Mat4[4][4];

#ifndef nil
    #define nil 0
#endif

#ifdef HGB_LANG
    #define loop while (true)
    #define for_range(i, min, max) for (usize i = min; i < max; i += 1)
    #define cast(T) (T)
    #define transmute(x, T) (*((T)(&x)))
#endif

#define HGB_PI 3.14159265358979323846264338327950288f
#define HGB_TAU 6.28318530717958647692528676655900576f

#define HGB_E 2.71828182845904523536f

#define HGB_SQRT_TWO 1.41421356237309504880168872420969808f
#define HGB_SQRT_THREE 1.73205080756887729352744634150587236f

#define HGB_ERROR make_float3(1.0f, 0.0f, 0.0f)
#define HGB_WARNING make_float3(1.0f, 1.0f, 0.0f)

// Computes x raised to the power of y
__DEVICE__ inline f32 hgb_pow(f32 x, f32 y) { return _powf(x, y); }

// Computes the value of the natural logarithm of x
__DEVICE__ inline f32 hgb_log(f32 x) { return _logf(x); }

// Computes the value of the logarithm of x to base 2
__DEVICE__ inline f32 hgb_log2(f32 x) { return _log2f(x); }

// Computes the value of the logarithm of x to base 10
__DEVICE__ inline f32 hgb_log10(f32 x) { return _log10f(x); }

// Computes e**x, the base-e exponential of x
__DEVICE__ inline f32 hgb_exp(f32 x) { return _expf(x); }

// Computes 2**x, the base-2 exponential of x
__DEVICE__ inline f32 hgb_exp2(f32 x) { return _exp2f(x); }

// Computes 10**x, the base-10 exponential of x
__DEVICE__ inline f32 hgb_exp10(f32 x) { return _exp10f(x); }

// Clamps x to be within the interval [min, max]
__DEVICE__ inline f32 hgb_clamp(f32 x, f32 min, f32 max) { return _clampf(x, min, max); }

// Clamps x to be within the interval [0.0f, 1.0f]
__DEVICE__ inline f32 hgb_saturate(f32 x) { return _saturatef(x); }

// Clamps x to be within the interval [0.0f, 1.0f]
__DEVICE__ inline f32 hgb_clamp01(f32 x) { return _saturatef(x); }

// Computes the non-negative square root of x
__DEVICE__ inline f32 hgb_sqrt(f32 x) { return _sqrtf(x); }

// Computes the cube root of x
__DEVICE__ inline f32 hgb_cbrt(f32 x) { return _cbrtf(x); }

// Returns the smallest integral value greater than or equal to x
__DEVICE__ inline f32 hgb_ceil(f32 x) { return _ceil(x); }

// Returns the largest integral value less than or equal to x
__DEVICE__ inline f32 hgb_floor(f32 x) { return _floor(x); }

// Returns the integral value nearest to but no larger in magnitude than x
__DEVICE__ inline f32 hgb_trunc(f32 x) { return _truncf(x); }

// Returns the integral value nearest to x rounding, with half-way cases rounded away from zero
__DEVICE__ inline f32 hgb_round(f32 x) { return _round(x); }

// Computes the floating-point remainder of x/y
__DEVICE__ inline f32 hgb_mod(f32 x, f32 y) { return _fmod(x, y); }

// Computes the value r such that r = x - n*y, where n is the integer nearest the exact value of x/y
__DEVICE__ inline f32 hgb_remainder(f32 x, f32 y) { return _fremainder(x, y); }

// Computes the square root of the sum of squares of x and y
__DEVICE__ inline f32 hgb_hypot(f32 x, f32 y) { return _hypotf(x, y); }

// Computes the cosine of x (measured in radians)
__DEVICE__ inline f32 hgb_cos(f32 x) { return _cosf(x); }

// Computes the sine of x (measured in radians)
__DEVICE__ inline f32 hgb_sin(f32 x) { return _sinf(x); }

// Computes the tangent of x (measured in radians)
__DEVICE__ inline f32 hgb_tan(f32 x) { return _tanf(x); }

// Computes the principle value of the arc cosine of x
__DEVICE__ inline f32 hgb_acos(f32 x) { return _acosf(x); }

// Computes the principle value of the arc sine of x
__DEVICE__ inline f32 hgb_asin(f32 x) { return _asinf(x); }

// Computes the principal value of the arc tangent of y/x, using the signs of
// both arguments to determine the quadrant of the return value
__DEVICE__ inline f32 hgb_atan2(f32 y, f32 x) { return _atan2f(y, x); }

// Computes the principle value of the inverse hyperbolic cosine of x
__DEVICE__ inline f32 hgb_acosh(f32 x) { return _acoshf(x); }

// Computes the principle value of the inverse hyperbolic sine of x
__DEVICE__ inline f32 hgb_asinh(f32 x) { return _asinhf(x); }

// Computes the inverse hyperbolic tangent of x
__DEVICE__ inline f32 hgb_atanh(f32 x) { return _atanhf(x); }

// Computes the hyperbolic cosine of x
__DEVICE__ inline f32 hgb_cosh(f32 x) { return _coshf(x); }

// Computes the hyperbolic sine of x
__DEVICE__ inline f32 hgb_sinh(f32 x) { return _sinhf(x); }

// Computes the hyperbolic tangent of x
__DEVICE__ inline f32 hgb_tanh(f32 x) { return _tanhf(x); }

// Returns the positive difference between x and y:  x - y if x > y, +0 if x is
// less than or equal to y
__DEVICE__ inline f32 hgb_fdim(f32 x, f32 y) { return _fdimf(x, y); }

// Computes (x * y) + z as a single operation
__DEVICE__ inline f32 hgb_fma(f32 x, f32 y, f32 z) { return _fmaf(x, y, z); }

// Computes the natural logorithm of the absolute value of the gamma function of x
__DEVICE__ inline f32 hgb_lgamma(f32 x) { return _lgammaf(x); }

// Computes the gamma function of x
__DEVICE__ inline f32 hgb_tgamma(f32 x) { return _tgammaf(x); }

// Computes the reciprocal of square root of x
__DEVICE__ inline f32 hgb_rsqrt(f32 x) { return _rsqrtf(x); }

// Returns a non-zero value if and only if x is an infinite value
__DEVICE__ inline bool hgb_isinf(f32 x) { return bool(isinf(x)); }

// Returns a non-zero value if and only if x is a NaN value
__DEVICE__ inline bool hgb_isnan(f32 x) { return bool(isnan(x)); }

// Returns a non-zero value if and only if sign bit of x is set
__DEVICE__ inline bool hgb_signbit(f32 x) { return bool(signbit(x)); }

// Returns x with its sign changed to y's
__DEVICE__ inline f32 hgb_copysign(f32 x, f32 y) { return _copysignf(x, y); }

// Extract mantissa and exponent from x. The mantissa m returned is a f32 with
// magnitude in the interval [1/2, 1) or 0, and exp is updated with integer
// exponent value, whereas x = m * 2^exp
__DEVICE__ inline f32 hgb_frexp(f32 x, i32 exp) { return _frexp(x, exp); }

// Returns (x * 2^exp)
__DEVICE__ inline f32 hgb_ldexp(f32 x, i32 exp) { return _ldexp(x, exp); }

#define hgb_square(x) ((x) * (x))
#define hgb_cube(x) ((x) * (x) * (x))

#define hgb_max(a, b) ((a) > (b) ? (a) : (b))
#define hgb_min(a, b) ((a) < (b) ? (a) : (b))

#define hgb_max3(a, b, c) hgb_max(a, hgb_max(b, c))
#define hgb_min3(a, b, c) hgb_min(a, hgb_min(b, c))

#define hgb_abs(x) ((x) < 0 ? -(x) : (x))
#define hgb_sign(x) ((x) < 0 ? -1 : (x) > 0 ? 1 : 0)

__DEVICE__ inline f32 hgb_to_degrees(f32 radians) {
    return radians * 360.0f / HGB_TAU;
}

__DEVICE__ inline f32 hgb_to_radians(f32 degrees) {
    return degrees * HGB_TAU / 360.0f;
}

// Just set `HGB_SPOW`.
enum Spow_Settings {
    Spow_Preserve,
    Spow_Clamp,
    Spow_Mirror,
};

__DEVICE__ inline f32 hgb_spow(f32 x, f32 p) {
    #if defined(HGB_SPOW) && HGB_SPOW == Spow_Preserve
        if (x < 0.0f) {
            return x;
        }
        return hgb_pow(x, p);
    #elif defined(HGB_SPOW) && HGB_SPOW == Spow_Clamp
        if (x < 0.0f) {
            return 0.0f;
        }
        return hgb_pow(x, p);
    #else
        return hgb_sign(x) * hgb_pow(hgb_abs(x), p);
    #endif
}

__DEVICE__ inline f32 hgb_lerp(f32 a, f32 b, f32 t) {
    return (1.0f - t) * a + t * b;
}

__DEVICE__ inline f32 smoothstep(f32 a, f32 b, f32 x) {
   f32 t = (x - a) / (b - a);
   return hgb_square(t) * (3.0f - 2.0f * t);
}

__DEVICE__ inline f32 smootherstep(f32 a, f32 b, f32 x) {
   f32 t = (x - a) / (b - a);
   return hgb_cube(t) * (t * (6.0f * t - 15.0f) + 10.0f);
}

__DEVICE__ inline f32 step(f32 edge, f32 x) {
    if (x < edge) {
        return 0.0f;
    }
    return 1.0f;
}

__DEVICE__ Vec2 hgb_vec2(f32 x, f32 y) {
    Vec2 out = {x, y};
    return out;
}
__DEVICE__ Vec3 hgb_vec3(f32 x, f32 y, f32 z) {
    Vec3 out = {x, y, z};
    return out;
}
__DEVICE__ Vec4 hgb_vec4(f32 x, f32 y, f32 z, f32 w) {
    Vec4 out = {x, y, z, w};
    return out;
}

__DEVICE__ Vec2 hgb_vec2_repeat(f32 v) { return hgb_vec2(v, v); }
__DEVICE__ Vec3 hgb_vec3_repeat(f32 v) { return hgb_vec3(v, v, v); }
__DEVICE__ Vec4 hgb_vec4_repeat(f32 v) { return hgb_vec4(v, v, v, v); }

__DEVICE__ Vec2 hgb_vec2_zeros() { return hgb_vec2_repeat(0.0f); }
__DEVICE__ Vec3 hgb_vec3_zeros() { return hgb_vec3_repeat(0.0f); }
__DEVICE__ Vec4 hgb_vec4_zeros() { return hgb_vec4_repeat(0.0f); }

__DEVICE__ Vec3 hgb_mat3_mul_vec3(Mat3 m, Vec3 v) {
    return make_float3(
        v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2],
        v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2],
        v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2]
    );
}

typedef struct Stack Stack;
struct Stack {
    __PRIVATE__ byte *data; // ?
    usize size;
    usize offset;
};

__DEVICE__ void hgb_stack_init(__PRIVATE__ Stack *s, __PRIVATE__ byte *backing, usize size) {
    s->data = backing;
    s->size = size;
    s->offset = 0;
}

__DEVICE__ void *hgb_stack_alloc(__PRIVATE__ Stack *s, usize amount) {
    void *ptr = s->data + s->offset;
    s->offset += amount;
    return ptr;
}

__DEVICE__ void hgb_stack_free_all(__PRIVATE__ Stack *s) {
    s->offset = 0;
}

__DEVICE__ void hgb_stack_free(__PRIVATE__ Stack *s, __PRIVATE__ void *ptr) {
    s->offset = usize(ptr) - usize(s->data);
}

typedef struct Arena Arena;
struct Arena {
    __PRIVATE__ byte *data; // TODO
    usize size;
    usize offset;
};

__DEVICE__ void arena_init(__PRIVATE__ Arena *a, __PRIVATE__ byte *backing, usize size) {
    a->data = backing;
    a->size = size;
    a->offset = 0;
}

__DEVICE__ void *arena_alloc(__PRIVATE__ Arena *a, usize amount) {
    usize total = a->offset + amount;
    if (total <= a->size) {
        void *ptr = a->data + a->offset;
        a->offset = total;
        return ptr;
    }
    return nil;
}

__DEVICE__ void arena_free_all(__PRIVATE__ Arena *a) {
    a->offset = 0;
}

typedef struct Linspace Linspace;
struct Linspace {
    f32 delta;
    f32 start;
    usize steps;
    usize current;
    bool done;
};

__DEVICE__ Linspace linspace_create(f32 start, f32 end, usize steps) {
    Linspace it;
    it.delta = (end - start) / f32(steps - 1);
    it.start = start;
    it.steps = steps;
    it.current = 0;
    it.done = false;
    return it;
}

__DEVICE__ f32 linspace_next(__PRIVATE__ Linspace *it) {
    if (it->current == it->steps - 1) {
        it->done = true;
    }
    f32 res = f32(it->current) * it->delta + it->start;
    it->current += 1;
    return res;
}

__DEVICE__ f32 *linspace_allocate(__PRIVATE__ Arena *arena, Linspace it) {
    f32 *array = (f32 *)arena_alloc(arena, sizeof(f32) * it.steps);
    if (array == nil) {
        return nil;
    }
    usize i = 0;
    while (!it.done) {
        array[i] = linspace_next(&it);
        i += 1;
    }
    return array;
}

__DEVICE__ inline f32 hgb_basis_gaussian(f32 x, f32 size) {
    return hgb_exp(hgb_pow(x / size, 2.0f));
}

__DEVICE__ inline f32 hgb_basis_tent(f32 x, f32 size) {
    return hgb_max(1.0f - hgb_abs(x / size), 0.0f);
}

__DEVICE__ inline f32 hgb_basis_exponential(f32 x, f32 size) {
    return hgb_exp(-hgb_abs(x / size));
}

// Default to knee = 1.0f.
__DEVICE__ inline f32 hgb_basis_falloff(f32 x, f32 size, f32 knee) {
    if (x == 0.0f) {
        return 1.0f;
    }
    return hgb_pow(1.0f / hgb_pow(x / size, 2.0f), knee);
}

__DEVICE__ inline f32 hgb_basis_boxcar(f32 x, f32 size) {
    if (x < -size || x > size) {
        return 0.0f;
    }
    return 1.0f;
}

// No function pointers in OpenCL.
#define hgb_apply_tf(f, v) make_float3(f(v.x), f(v.y), f(v.z))

__DEVICE__ float hgb_logc_encode(float x) {
    if (x > 0.010591f) {
        return 0.247190f * _log10f(5.555556f * x + 0.052272f) + 0.385537f;
    }
    return 5.367655f * x + 0.092809f;
}

__DEVICE__ float hgb_logc_decode(float x) {
    if (x > 0.1496582f){
        return (_powf(10.0f, (x - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f;
    }
    return (x - 0.092809f) / 5.367655f;
}
