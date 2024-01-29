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
    #define HGB_SPOW Spow_Mirror
#endif

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
#define for_range(i, min, max) for (usize i = min; i < max; i += 1)
#define cast(T) (T)
#define transmute(x, T) (*((T *)(&x)))

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

#define HGB_PI 3.14159265358979323846264338327950288f
#define HGB_TAU 6.28318530717958647692528676655900576f

#define HGB_E 2.71828182845904523536f

#define HGB_SQRT_TWO 1.41421356237309504880168872420969808f
#define HGB_SQRT_THREE 1.73205080756887729352744634150587236f

#define HGB_OK hgb_vec3(0.0f, 1.0f, 0.0f)
#define HGB_WARNING hgb_vec3(1.0f, 1.0f, 0.0f)
#define HGB_ERROR hgb_vec3(1.0f, 0.0f, 0.0f)

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

#define hgb_clamp(x, start, end) hgb_max(hgb_min((x), (end)), (start))
#define hgb_clamp01(x) hgb_clamp((x), 0, 1)
#define hgb_saturate(x) hgb_clamp((x), 0, 1)

#define hgb_abs(x) ((x) < 0 ? -(x) : (x))
#define hgb_sign(x) ((x) < 0 ? -1 : (x) > 0 ? 1 : 0)

__DEVICE__ inline f32 hgb_to_degrees(f32 radians) {
    return radians * 360.0f / HGB_TAU;
}

__DEVICE__ inline f32 hgb_to_radians(f32 degrees) {
    return degrees * HGB_TAU / 360.0f;
}

enum Spow_Settings {
    Spow_Preserve,
    Spow_Clamp,
    Spow_Mirror,
};

__DEVICE__ inline f32 hgb_spow(f32 x, f32 p) {
    #if HGB_SPOW == Spow_Preserve
        if (x < 0.0f) {
            return x;
        }
        return hgb_pow(x, p);
    #elif HGB_SPOW == Spow_Clamp
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

typedef float2 Vec2;
typedef float3 Vec3;
typedef float4 Vec4;

__DEVICE__ inline Vec2 hgb_vec2(f32 x, f32 y) {
    Vec2 out = {x, y};
    return out;
}

__DEVICE__ inline Vec3 hgb_vec3(f32 x, f32 y, f32 z) {
    Vec3 out = {x, y, z};
    return out;
}

__DEVICE__ inline Vec4 hgb_vec4(f32 x, f32 y, f32 z, f32 w) {
    Vec4 out = {x, y, z, w};
    return out;
}

__DEVICE__ inline Vec2 hgb_vec2_repeat(f32 v) { return hgb_vec2(v, v); }
__DEVICE__ inline Vec3 hgb_vec3_repeat(f32 v) { return hgb_vec3(v, v, v); }
__DEVICE__ inline Vec4 hgb_vec4_repeat(f32 v) { return hgb_vec4(v, v, v, v); }

__DEVICE__ inline Vec2 hgb_vec2_zeros() { return hgb_vec2_repeat(0.0f); }
__DEVICE__ inline Vec3 hgb_vec3_zeros() { return hgb_vec3_repeat(0.0f); }
__DEVICE__ inline Vec4 hgb_vec4_zeros() { return hgb_vec4_repeat(0.0f); }

// No function pointers in OpenCL.
#define hgb_vec2_map(v, f) hgb_vec2(f(v.x), f(v.y))
#define hgb_vec3_map(v, f) hgb_vec3(f(v.x), f(v.y), f(v.z))
#define hgb_vec4_map(v, f) hgb_vec4(f(v.x), f(v.y), f(v.z), f(v.w))

__DEVICE__ Vec2 hgb_vec2_swizzle(Vec2 v, usize x, usize y) {
    x = hgb_clamp(x, 0, 1);
    y = hgb_clamp(y, 0, 1);
    f32 *it = &transmute(v, f32);
    return hgb_vec2(it[x], it[y]);
}

__DEVICE__ Vec3 hgb_vec3_swizzle(Vec3 v, usize x, usize y, usize z) {
    x = hgb_clamp(x, 0, 2);
    y = hgb_clamp(y, 0, 2);
    z = hgb_clamp(z, 0, 2);
    f32 *it = &transmute(v, f32);
    return hgb_vec3(it[x], it[y], it[z]);
}

__DEVICE__ Vec4 hgb_vec4_swizzle(Vec4 v, usize x, usize y, usize z, usize w) {
    x = hgb_clamp(x, 0, 3);
    y = hgb_clamp(y, 0, 3);
    z = hgb_clamp(z, 0, 3);
    w = hgb_clamp(w, 0, 3);
    f32 *it = &transmute(v, f32);
    return hgb_vec4(it[x], it[y], it[z], it[w]);
}

__DEVICE__ f32 hgb_vec2_dot(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}

__DEVICE__ f32 hgb_vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__DEVICE__ f32 hgb_vec4_dot(Vec4 a, Vec4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__DEVICE__ f32 hgb_vec2_cross(Vec2 a, Vec2 b) {
    return a.x * b.y - b.x * a.y;
}

__DEVICE__ Vec3 hgb_vec3_cross(Vec3 a, Vec3 b) {
    return hgb_vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__DEVICE__ f32 hgb_vec2_length(Vec2 v) {
    return hgb_sqrt(hgb_square(v.x) + hgb_square(v.y));
}

__DEVICE__ f32 hgb_vec3_length(Vec3 v) {
    return hgb_sqrt(hgb_square(v.x) + hgb_square(v.y) + hgb_square(v.z));
}

__DEVICE__ f32 hgb_vec4_length(Vec4 v) {
    return hgb_sqrt(hgb_square(v.x) + hgb_square(v.y) + hgb_square(v.z) + hgb_square(v.w));
}

typedef struct Mat2 Mat2;
struct Mat2 {
    f32 v[2][2];
};

typedef struct Mat3 Mat3;
struct Mat3 {
    f32 v[3][3];
};

typedef struct Mat4 Mat4;
struct Mat4 {
    f32 v[4][4];
};

__DEVICE__ inline Mat2 hgb_mat2(
    f32 v00, f32 v01,
    f32 v10, f32 v11
) {
    Mat2 out = {{
        {v00, v01},
        {v10, v11}
    }};
    return out;
}

__DEVICE__ inline Mat3 hgb_mat3(
    f32 v00, f32 v01, f32 v02,
    f32 v10, f32 v11, f32 v12,
    f32 v20, f32 v21, f32 v22
) {
    Mat3 out = {{
        {v00, v01, v02},
        {v10, v11, v12},
        {v20, v21, v22}
    }};
    return out;
}

__DEVICE__ inline Mat4 hgb_mat4(
    f32 v00, f32 v01, f32 v02, f32 v03,
    f32 v10, f32 v11, f32 v12, f32 v13,
    f32 v20, f32 v21, f32 v22, f32 v23,
    f32 v30, f32 v31, f32 v32, f32 v33
) {
    Mat4 out = {{
        {v00, v01, v02, v03},
        {v10, v11, v12, v13},
        {v20, v21, v22, v23},
        {v30, v31, v32, v33}
    }};
    return out;
}

__DEVICE__ inline Mat2 hgb_mat2_repeat(f32 v) {
    return hgb_mat2(
        v, v,
        v, v
    );
}

__DEVICE__ inline Mat3 hgb_mat3_repeat(f32 v) {
    return hgb_mat3(
        v, v, v,
        v, v, v,
        v, v, v
    );
}

__DEVICE__ inline Mat4 hgb_mat4_repeat(f32 v) {
    return hgb_mat4(
        v, v, v, v,
        v, v, v, v,
        v, v, v, v,
        v, v, v, v
    );
}

__DEVICE__ inline Mat2 hgb_mat2_zeros() { return hgb_mat2_repeat(0.0f); }
__DEVICE__ inline Mat3 hgb_mat3_zeros() { return hgb_mat3_repeat(0.0f); }
__DEVICE__ inline Mat4 hgb_mat4_zeros() { return hgb_mat4_repeat(0.0f); }

__DEVICE__ inline Mat2 hgb_mat2_identity() {
    return hgb_mat2(
        1.0f, 0.0f,
        0.0f, 1.0f
    );
}

__DEVICE__ inline Mat3 hgb_mat3_identity() {
    return hgb_mat3(
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    );
}

__DEVICE__ inline Mat4 hgb_mat4_identity(f32 v) {
    return hgb_mat4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

__DEVICE__ inline Vec3 hgb_mat3_mul_vec3(Mat3 m, Vec3 v) {
    return hgb_vec3(
        v.x * m.v[0][0] + v.y * m.v[0][1] + v.z * m.v[0][2],
        v.x * m.v[1][0] + v.y * m.v[1][1] + v.z * m.v[1][2],
        v.x * m.v[2][0] + v.y * m.v[2][1] + v.z * m.v[2][2]
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

__DEVICE__ inline f32 hgb_linear(f32 x) {
    return x;
}

// EI 800. TODO
__DEVICE__ f32 hgb_arri_logc_encode(f32 x) {
    if (x > 0.010591f) {
        return 0.247190f * hgb_log10(5.555556f * x + 0.052272f) + 0.385537f;
    }
    return 5.367655f * x + 0.092809f;
}

__DEVICE__ f32 hgb_arri_logc_decode(f32 x) {
    if (x > 0.1496582f){
        return (hgb_pow(10.0f, (x - 0.385537f) / 0.2471896f) - 0.052272f) / 5.555556f;
    }
    return (x - 0.092809f) / 5.367655f;
}

__DEVICE__ f32 hgb_fuji_flog_encode(f32 x) {
    if (x >= 0.00089f) {
        return 0.344676f * _log10f(0.555556f * x + 0.009468f) + 0.790453f;
    }
    return 8.735631f * x + 0.092864f;
}

__DEVICE__ f32 hgb_fuji_flog_decode(f32 x) {
    if (x >= 0.100537775223865f) {
        return (_powf(10.0f, (x - 0.790453f) / 0.344676f)) / 0.555556f - 0.009468f / 0.555556f;
    }
    return (x - 0.092864f) / 8.735631f;
}

__DEVICE__ f32 hgb_nikon_nlog_encode(f32 x) {
    if (x > 0.328f) {
        return (150.0f / 1023.0f) * _logf(x) + (619.0f / 1023.0f);
    }
    return (650.0f / 1023.0f) * _powf((x + 0.0075f), 1.0f / 3.0f);
}

__DEVICE__ f32 hgb_nikon_nlog_decode(f32 x) {
    if (x > (452.0f / 1023.0f)){
        return _expf((x - 619.0f / 1023.0f) / (150.0f / 1023.0f));
    }
    return _powf(x / (650.0f / 1023.0f), 3.0f) - 0.0075f;
}

__DEVICE__ f32 hgb_panasonic_vlog_encode(f32 x) {
    if (x < 0.01f) {
        return 5.6f * x + 0.125f;
    }
    return 0.241514f * _log10f(x + 0.00873f) + 0.598206f;
}

__DEVICE__ f32 hgb_panasonic_vlog_decode(f32 x) {
    if (x < 0.181f) {
        return (x - 0.125f) / 5.6f;
    }
    return _powf(10.0f, ((x - 0.598206f) / 0.241514f)) - 0.00873f;
}

__DEVICE__ f32 hgb_sony_slog3_encode(f32 x) {
    if (x >= 0.01125000f) {
        return (420.0f + _log10f((x + 0.01f) / (0.18f + 0.01f)) * 261.5f) / 1023.0f;
    }
    return (x * (171.2102946929f - 95.0f) / 0.01125000f + 95.0f) / 1023.0f;
}

__DEVICE__ f32 hgb_sony_slog3_decode(f32 x) {
    if (x >= (171.2102946929f / 1023.0f)) {
        return _powf(10.0f, (x * 1023.0f - 420.0f) / 261.5f) * (0.18f + 0.01f) - 0.01f;
    }
    return (x * 1023.0f - 95.0f) * 0.01125000f / (171.2102946929f - 95.0f);
}

typedef struct Primaries_Whitepoint Primaries_Whitepoint;
struct Primaries_Whitepoint {
    Vec2 r;
    Vec2 g;
    Vec2 b;
    Vec2 w;
};

__DEVICE__ Mat3 hgb_npm(Primaries_Whitepoint pw) {
    f32 y = 1.0f;
    f32 x = pw.w.x * y / pw.w.y;
    f32 z = (1.0f - pw.w.x - pw.w.y) * y / pw.w.y;

    f32 d = pw.r.x * (pw.b.y - pw.g.y)
        + pw.b.x * (pw.g.y - pw.r.y)
        + pw.g.x * (pw.r.y - pw.b.y);

    f32 sr = (x * (pw.b.y - pw.g.y)
        - pw.g.x * (y * (pw.b.y - 1.0f) + pw.b.y * (x + z))
        + pw.b.x * (y * (pw.g.y - 1.0f) + pw.g.y * (x + z)))
        / d;
    f32 sg = (x * (pw.r.y - pw.b.y)
        + pw.r.x * (y * (pw.b.y - 1.0f) + pw.b.y * (x + z))
        - pw.b.x * (y * (pw.r.y - 1.0f) + pw.r.y * (x + z)))
        / d;
    f32 sb = (x * (pw.g.y - pw.r.y)
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

__CONSTANT__ Primaries_Whitepoint HGB_BT_709 = {
    {0.640, 0.330},
    {0.300, 0.600},
    {0.150, 0.060},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_BT_2020 = {
    {0.708, 0.292},
    {0.170, 0.797},
    {0.131, 0.046},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_DCI_P3 = {
    {0.680, 0.320},
    {0.265, 0.690},
    {0.150, 0.060},
    {0.314, 0.351}
};

__CONSTANT__ Primaries_Whitepoint HGB_DISPLAY_P3 = {
    {0.680, 0.320},
    {0.265, 0.690},
    {0.150, 0.060},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_ACES_AP0 = {
    {0.73470, 0.26530},
    {0.00000, 1.00000},
    {0.00010, -0.07700},
    {0.32168, 0.33767}
};

__CONSTANT__ Primaries_Whitepoint HGB_ACES_AP1 = {
    {0.71300, 0.29300},
    {0.16500, 0.83000},
    {0.12800, 0.04400},
    {0.32168, 0.33767}
};

__CONSTANT__ Primaries_Whitepoint HGB_ADOBE_RGB = {
    {0.6400, 0.3300},
    {0.2100, 0.7100},
    {0.1500, 0.0600},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_ADOBE_WIDE_GAMUT_RGB = {
    {0.7347, 0.2653},
    {0.1152, 0.8264},
    {0.1566, 0.0177},
    {0.3457, 0.3585}
};

__CONSTANT__ Primaries_Whitepoint HGB_ARRI_WIDE_GAMUT_3 = {
    {0.6840, 0.3130},
    {0.2210, 0.8480},
    {0.0861, -0.1020},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_ARRI_WIDE_GAMUT_4 = {
    {0.7347, 0.2653},
    {0.1424, 0.8576},
    {0.0991, -0.0308},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_CANON_CINEMA_GAMUT = {
    {0.7400, 0.2700},
    {0.1700, 1.1400},
    {0.0800, -0.1000},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_DJI_D_GAMUT = {
    {0.7100, 0.3100},
    {0.2100, 0.8800},
    {0.0900, -0.0800},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_E_GAMUT = {
    {0.8000, 0.3177},
    {0.1800, 0.9000},
    {0.0650, -0.0805},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_PANASONIC_V_GAMUT = {
    {0.7300, 0.2800},
    {0.1650, 0.8400},
    {0.1000, -0.0300},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_PROPHOTO = {
    {0.734699, 0.265301},
    {0.159597, 0.840403},
    {0.036598, 0.000105},
    {0.345704, 0.358540}
};

__CONSTANT__ Primaries_Whitepoint HGB_RED_WIDE_GAMUT_RGB = {
    {0.780308, 0.304253},
    {0.121595, 1.493994},
    {0.095612, -0.084589},
    {0.3127, 0.3290}
};

__CONSTANT__ Primaries_Whitepoint HGB_S_GAMUT = {
    {0.7300, 0.2800},
    {0.1400, 0.8550},
    {0.1000, -0.0500},
    {0.3127, 0.3290},
};

__CONSTANT__ Primaries_Whitepoint HGB_S_GAMUT3_CINE = {
    {0.7660, 0.2750},
    {0.2250, 0.8000},
    {0.0890, -0.0870},
    {0.3127, 0.3290},
};
