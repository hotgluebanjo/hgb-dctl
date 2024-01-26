// HGB DCTL header 0.1.0
// ---------------------
// Attempting to be compatible with all DCTL targets.
// - https://github.com/hotgluebanjo
//
// References:
//
// - https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
// - https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
//

// Some, uh, assumptions. stdint is not included.
typedef unsigned char   u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long  u64;

typedef signed char   s8;
typedef signed short s16;
typedef signed int   s32;
typedef signed long  s64;

typedef size_t    uint;
typedef ptrdiff_t sint;

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
#define for_range(i, min, max) for (uint i = min; i < max; i += 1)
#define cast(T) (T)
#define transmute(x, T) (*((T)(&x)))
#endif

#define HGB_PI 3.14159265358979323846264338327950288f
#define HGB_TAU 6.28318530717958647692528676655900576f

#define HGB_E 2.71828182845904523536f

#define HGB_SQRT_TWO 1.41421356237309504880168872420969808f
#define HGB_SQRT_THREE 1.73205080756887729352744634150587236f

// Computes x raised to the power of y
__DEVICE__ inline float hgb_pow(float x, float y) { return _powf(x, y); }

// Computes the value of the natural logarithm of x
__DEVICE__ inline float hgb_log(float x) { return _logf(x); }

// Computes the value of the logarithm of x to base 2
__DEVICE__ inline float hgb_log2(float x) { return _log2f(x); }

// Computes the value of the logarithm of x to base 10
__DEVICE__ inline float hgb_log10(float x) { return _log10f(x); }

// Computes e**x, the base-e exponential of x
__DEVICE__ inline float hgb_exp(float x) { return _expf(x); }

// Computes 2**x, the base-2 exponential of x
__DEVICE__ inline float hgb_exp2(float x) { return _exp2f(x); }

// Computes 10**x, the base-10 exponential of x
__DEVICE__ inline float hgb_exp10(float x) { return _exp10f(x); }

// Clamps x to be within the interval [min, max]
__DEVICE__ inline float hgb_clamp(float x, float min, float max) { return _clampf(x, min, max); }

// Clamps x to be within the interval [0.0f, 1.0f]
__DEVICE__ inline float hgb_saturate(float x) { return _saturatef(x); }

// Clamps x to be within the interval [0.0f, 1.0f]
__DEVICE__ inline float hgb_clamp01(float x) { return _saturatef(x); }

// Computes the non-negative square root of x
__DEVICE__ inline float hgb_sqrt(float x) { return _sqrtf(x); }

// Computes the cube root of x
__DEVICE__ inline float hgb_cbrt(float x) { return _cbrtf(x); }

// Returns the smallest integral value greater than or equal to x
__DEVICE__ inline float hgb_ceil(float x) { return _ceil(x); }

// Returns the largest integral value less than or equal to x
__DEVICE__ inline float hgb_floor(float x) { return _floor(x); }

// Returns the integral value nearest to but no larger in magnitude than x
__DEVICE__ inline float hgb_trunc(float x) { return _truncf(x); }

// Returns the integral value nearest to x rounding, with half-way cases rounded away from zero
__DEVICE__ inline float hgb_round(float x) { return _round(x); }

// Computes the floating-point remainder of x/y
__DEVICE__ inline float hgb_mod(float x, float y) { return _fmod(x, y); }

// Computes the value r such that r = x - n*y, where n is the integer nearest the exact value of x/y
__DEVICE__ inline float hgb_remainder(float x, float y) { return _fremainder(x, y); }

// Computes the square root of the sum of squares of x and y
__DEVICE__ inline float hgb_hypot(float x, float y) { return _hypotf(x, y); }

// Computes the cosine of x (measured in radians)
__DEVICE__ inline float hgb_cos(float x) { return _cosf(x); }

// Computes the sine of x (measured in radians)
__DEVICE__ inline float hgb_sin(float x) { return _sinf(x); }

// Computes the tangent of x (measured in radians)
__DEVICE__ inline float hgb_tan(float x) { return _tanf(x); }

// Computes the principle value of the arc cosine of x
__DEVICE__ inline float hgb_acos(float x) { return _acosf(x); }

// Computes the principle value of the arc sine of x
__DEVICE__ inline float hgb_asin(float x) { return _asinf(x); }

// Computes the principal value of the arc tangent of y/x, using the signs of both arguments to determine the quadrant of the return value
__DEVICE__ inline float hgb_atan2(float y, float x) { return _atan2f(y, x); }

// Computes the principle value of the inverse hyperbolic cosine of x
__DEVICE__ inline float hgb_acosh(float x) { return _acoshf(x); }

// Computes the principle value of the inverse hyperbolic sine of x
__DEVICE__ inline float hgb_asinh(float x) { return _asinhf(x); }

// Computes the inverse hyperbolic tangent of x
__DEVICE__ inline float hgb_atanh(float x) { return _atanhf(x); }

// Computes the hyperbolic cosine of x
__DEVICE__ inline float hgb_cosh(float x) { return _coshf(x); }

// Computes the hyperbolic sine of x
__DEVICE__ inline float hgb_sinh(float x) { return _sinhf(x); }

// Computes the hyperbolic tangent of x
__DEVICE__ inline float hgb_tanh(float x) { return _tanhf(x); }

// Returns the positive difference between x and y:  x - y if x > y, +0 if x is less than or equal to y
__DEVICE__ inline float hgb_fdim(float x, float y) { return _fdimf(x, y); }

// Computes (x * y) + z as a single operation
__DEVICE__ inline float hgb_fma(float x, float y, float z) { return _fmaf(x, y, z); }

// Computes the natural logorithm of the absolute value of the gamma function of x
__DEVICE__ inline float hgb_lgamma(float x) { return _lgammaf(x); }

// Computes the gamma function of x
__DEVICE__ inline float hgb_tgamma(float x) { return _tgammaf(x); }

// Computes the reciprocal of square root of x
__DEVICE__ inline float hgb_rsqrt(float x) { return _rsqrtf(x); }

// Returns a non-zero value if and only if x is an infinite value
__DEVICE__ inline int hgb_isinf(float x) { return isinf(x); }

// Returns a non-zero value if and only if x is a NaN value
__DEVICE__ inline int hgb_isnan(float x) { return isnan(x); }

// Returns a non-zero value if and only if sign bit of x is set
__DEVICE__ inline int hgb_signbit(float x) { return signbit(x); }

// Returns x with its sign changed to y's
__DEVICE__ inline float hgb_copysign(float x, float y) { return _copysignf(x, y); }

// Extract mantissa and exponent from x. The mantissa m returned is a float with magnitude in the interval [1/2, 1) or 0, and exp is updated with integer exponent value, whereas x = m * 2^exp
__DEVICE__ inline float hgb_frexp(float x, int exp) { return _frexp(x, exp); }

// Returns (x * 2^exp)
__DEVICE__ inline float hgb_ldexp(float x, int exp) { return _ldexp(x, exp); }

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

__DEVICE__ inline f32 hgb_spow_preserve(f32 x, f32 p) {
    if (x < 0.0f) {
        return x;
    }
    return hgb_pow(x, p);
}

__DEVICE__ inline f32 hgb_spow_mirror(f32 x, f32 p) {
    return hgb_sign(x) * hgb_pow(hgb_abs(x), p);
}

#define HGB_ERROR make_float3(1.0f, 0.0f, 0.0f)
#define HGB_WARNING make_float3(1.0f, 1.0f, 0.0f)

__DEVICE__ inline f32 hgb_lerp(f32 a, f32 b, f32 t) {
    return (1.0f - t) * a + t * b;
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

__DEVICE__ inline f32 hgb_basis_falloff(f32 x, f32 size, f32 knee = 1.0f) {
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

typedef struct Stack Stack;
struct Stack {
    __PRIVATE__ byte *data; // ?
    uint size;
    uint offset;
};

__DEVICE__ void hgb_stack_init(__PRIVATE__ Stack *s, __PRIVATE__ byte *backing, uint size) {
    s->data = backing;
    s->size = size;
    s->offset = 0;
}

__DEVICE__ void *hgb_stack_alloc(__PRIVATE__ Stack *s, uint amount) {
    void *ptr = s->data + s->offset;
    s->offset += amount;
    return ptr;
}

__DEVICE__ void hgb_stack_free_all(__PRIVATE__ Stack *s) {
    s->offset = 0;
}

__DEVICE__ void hgb_stack_free(__PRIVATE__ Stack *s, __PRIVATE__ void *ptr) {
    s->offset = uint(ptr) - uint(s->data);
}

typedef struct Arena Arena;
struct Arena {
    __PRIVATE__ byte *data; // TODO
    uint size;
    uint offset;
};

__DEVICE__ void arena_init(__PRIVATE__ Arena *a, __PRIVATE__ byte *backing, uint size) {
    a->data = backing;
    a->size = size;
    a->offset = 0;
}

__DEVICE__ void *arena_alloc(__PRIVATE__ Arena *a, uint amount) {
    uint total = a->offset + amount;
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
    uint steps;
    uint current;
    bool done;
};

__DEVICE__ Linspace linspace_create(f32 start, f32 end, uint steps) {
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
    uint i = 0;
    while (!it.done) {
        array[i] = linspace_next(&it);
        i += 1;
    }
    return array;
}
