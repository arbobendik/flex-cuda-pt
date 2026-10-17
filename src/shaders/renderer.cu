#include "renderer.h"
#include <optix.h>
#include <cuda_runtime.h>
#include "flexkernels/pathtracer.cuh"

constexpr int STACK = 64;
constexpr float PI = 3.14159265f;
constexpr float INV_PI = 0.3183098861837907f;
constexpr float BIAS = 0.0000152587890625f;
// Columns
struct Matrix { float3 x, y, z; };
__device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ float3 operator*(float s, float3 a) { return a * s; }
__device__ float3 operator/(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
__device__ float3 xzy(float3 a) { return make_float3(a.x, a.z, a.y); }
__device__ float3 mix(float3 a, float3 b, float t) { return a * (1 - t) + b * t; }
__device__ float4 mix(float4 a, float4 b, float t) {
    return make_float4(a.x * (1 - t) + b.x * t, a.y * (1 - t) + b.y * t,
        a.z * (1 - t) + b.z * t, a.w * (1 - t) + b.w * t);
}
// -1, 0 or 1
__device__ float sign(float x) { return (x > 0) - (x < 0); }
__device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 reflect(float3 i, float3 n) { return i - n * (2 * dot(n, i)); }
// Zero under total internal reflection
__device__ float3 refract(float3 i, float3 n, float eta) {
    const float k = 1 - eta * eta * (1 - dot(n, i) * dot(n, i));
    return k < 0 ? make_float3(0, 0, 0) : i * eta - n * (eta * dot(n, i) + sqrtf(k));
}
__device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float3 normalize(float3 a) { return a * rnorm3df(a.x, a.y, a.z); }
__device__ float length(float3 a) { return sqrtf(dot(a, a)); }
__device__ bool operator==(float3 a, float3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
// Repeats into [0, 1)
__device__ float wrap(float coordinate) { return fminf(coordinate - floorf(coordinate), 0x1.fffffep-1f); }
__device__ float random(unsigned& state) { return ((state = hash(state)) >> 8) * 0x1p-24f; }

// Build TBN matrix for tangent-space normal mapping. Uses triangle edges and UV deltas to compute
// tangent (U direction) and bitangent (V direction), with handedness fix.
__device__ Matrix normalMapTBN(float3 p0, float3 p1, float3 p2, float2 uv0, float2 uv1, float2 uv2, float3 n) {
    const float3 e1 = p1 - p0;
    const float3 e2 = p2 - p0;
    const float2 duv1 = uv1 - uv0;
    const float2 duv2 = uv2 - uv0;
    // Compute inverse determinant for UV to world space transformation.
    const float det = duv1.x * duv2.y - duv1.y * duv2.x;
    const float inv_det = 1.0f / det;
    float3 t = (e1 * duv2.y - e2 * duv1.y) * inv_det;
    const float3 b_geom = (e2 * duv1.x - e1 * duv2.x) * inv_det;

    // Gram-Schmidt: make tangent perpendicular to interpolated normal.
    t = normalize(t - n * dot(n, t));
    // Handedness: bitangent along the geometric b, mirrored with the UVs, as Blender's tangent space.
    const float3 b = cross(n, t) * sign(dot(cross(n, t), b_geom));

    return Matrix{t, b, n};
}

__device__ float3 tangentToWorldNormalMap(float3 v, Matrix tbn) {
    return tbn.x * v.x + tbn.y * v.y + tbn.z * v.z;
}

constexpr unsigned LOBE_DIFFUSE = 0u;
constexpr unsigned LOBE_REFRACT = 1u;

// Of a to the base b, zero where undefined as Blender's
__device__ float logarithm(float a, float b) { return a > 0 && b > 0 && b != 1 ? logf(a) / logf(b) : 0; }
__device__ float3 divide(float3 a, float3 b) {
    return make_float3(b.x != 0 ? a.x / b.x : 0, b.y != 0 ? a.y / b.y : 0, b.z != 0 ? a.z / b.z : 0);
}
__device__ float4 texel(const Scene& scene, const Op& op, int x, int y) {
    return scene.texels[op.texel + ((y + op.height) % op.height) * op.width + (x + op.width) % op.width];
}
// Surface a program resolves to at a UV, facing along the normal with the tangent space tbn
__device__ Surface shade(const Scene& scene, unsigned program, float3 normal, Matrix tbn, float u, float v) {
    float3 stack[STACK], *top = stack;
    for (const Op* op = scene.ops + program; op->code; ++op) switch (op->code) {
        case PUSH: *top++ = op->value; break;
        case UV: *top++ = make_float3(u, v, 0); break;
        case NORMAL: *top++ = normal; break;
        case IMAGE: case ALPHA: case IMAGE_LINEAR: case ALPHA_LINEAR: { // Repeating
            float x = wrap(top[-1].x) * op->width, y = wrap(top[-1].y) * op->height;
            float4 color = texel(scene, *op, int(x), int(y));
            if (op->code == IMAGE_LINEAR || op->code == ALPHA_LINEAR) {
                x -= 0.5f; y -= 0.5f;
                const int i = int(floorf(x)), j = int(floorf(y));
                color = mix(mix(texel(scene, *op, i, j), texel(scene, *op, i + 1, j), x - i),
                    mix(texel(scene, *op, i, j + 1), texel(scene, *op, i + 1, j + 1), x - i), y - j);
            }
            top[-1] = op->code == IMAGE || op->code == IMAGE_LINEAR ? make_float3(color.x, color.y, color.z)
                : make_float3(1, 1, 1) * color.w;
        } break;
        case NORMAL_MAP: top[-1] = normalize(tangentToWorldNormalMap(top[-1] * 2 - make_float3(1, 1, 1), tbn)); break;
        case ADD: --top; top[-1] = top[-1] + *top; break;
        case SUBTRACT: --top; top[-1] = top[-1] - *top; break;
        case MULTIPLY: --top; top[-1] = top[-1] * *top; break;
        case DOT_PRODUCT: --top; top[-1] = make_float3(1, 1, 1) * dot(top[-1], *top); break;
        case MAXIMUM: --top;
            top[-1] = make_float3(fmaxf(top[-1].x, top->x), fmaxf(top[-1].y, top->y), fmaxf(top[-1].z, top->z));
            break;
        case LOGARITHM: --top;
            top[-1] = make_float3(logarithm(top[-1].x, top->x), logarithm(top[-1].y, top->y),
                logarithm(top[-1].z, top->z));
            break;
        case MULTIPLY_ADD: top -= 2; top[-1] = top[-1] * top[0] + top[1]; break;
        case DIVIDE: --top; top[-1] = divide(top[-1], *top); break;
        case SNAP: { --top;
            const float3 q = divide(top[-1], *top);
            top[-1] = make_float3(floorf(q.x) * top->x, floorf(q.y) * top->y, floorf(q.z) * top->z);
        } break;
    }
    return {normalize(stack[0]), stack[1], stack[2].x, stack[3].x, stack[4], stack[5].x, stack[6].x, stack[7],
        stack[8].x, stack[9].x, stack[10], stack[11].x};
}
// Of a triangle at its barycentrics
__device__ Surface shade(const Scene& scene, unsigned triangle, float s, float t, float3 normal) {
    const float2* uv = scene.uvs + 3 * triangle;
    const float3* vertex = scene.vertices + 3 * triangle;
    return shade(scene, scene.materials[triangle].program, normal,
        normalMapTBN(vertex[0], vertex[1], vertex[2], uv[0], uv[1], uv[2], normal),
        uv[0].x + (uv[1].x - uv[0].x) * s + (uv[2].x - uv[0].x) * t,
        uv[0].y + (uv[1].y - uv[0].y) * s + (uv[2].y - uv[0].y) * t);
}
// Beer's law for absorption over the distance through a medium: T = exp(-sigma_a * d), sigma_a = density * (1 - color)
// as Cycles' Volume Absorption
__device__ float3 beerLambert(float distance, const Surface& material) {
    const float depth = distance * material.density;
    return make_float3(expf(depth * (material.absorption.x - 1)), expf(depth * (material.absorption.y - 1)),
        expf(depth * (material.absorption.z - 1)));
}
__device__ bool neeDecision(const Surface& material, float n_dot_v, float eta_i, float eta_o) {
    // Calculate fresnel term
    const float3 F_n = mix(tinted(material, fabsf(n_dot_v), eta_i, eta_o), material.albedo,
        material.metallic);
    const float F_n_greyscale = rgb_to_greyscale(F_n);
    const float diffuse_factor_estimate = fmaxf((1.0f - F_n_greyscale) * (1.0f - material.metallic)
        * (1.0f - material.transparency), 0.0f);
    return diffuse_factor_estimate > 0.04f || material.roughness > 0.2f;
}

// ReSTIR PT Enhanced (Lin, Kettunen and Wyman 2026) on top of the path tracing of walk below.
// Paths live in primary sample space, vertex b draws its random numbers from stream(seed, b).

// Single-vertex roughness threshold (Sec. 4.2)
constexpr float RESTIR_ROUGHNESS = 0.2f;
// Dual ray footprint constant c / 100 * 4 PI of Eq. 5 with c = 0.02
constexpr float RESTIR_FOOTPRINT = 0.0025132741228718345f;
// Temporal confidence cap and its duplication-driven reduction (Sec. 5)
constexpr float RESTIR_CAP = 20.0f;
constexpr float RESTIR_CAP_MIN = 1.0f;
constexpr float RESTIR_DUPLICATION_ALPHA = 0.1f;

// How a path ends: next event estimation, emissive surface or environment
constexpr unsigned END_NEE = 0u;
constexpr unsigned END_EMIT = 1u;
constexpr unsigned END_ENV = 2u;

// Reconnection vertex: none (pure random replay), triangle hit, NEE light vertex or environment direction
constexpr unsigned RC_NONE = 0u;
constexpr unsigned RC_SURFACE = 1u;
constexpr unsigned RC_LIGHT = 2u;
constexpr unsigned RC_ENV = 3u;

// Bit offsets of the packed path description
constexpr unsigned PATH_END = 0u;
constexpr unsigned PATH_K = 5u;
constexpr unsigned PATH_RC = 10u;
constexpr unsigned PATH_TYPE = 12u;
constexpr unsigned PATH_LOBE_PREV = 15u;
constexpr unsigned PATH_LOBE = 17u;
constexpr unsigned PATH_NEE = 19u;

// Random streams besides the per vertex ones
constexpr unsigned STREAM_NEE = 64u;
constexpr unsigned STREAM_RR = 128u;
constexpr unsigned STREAM_JITTER = 192u;
constexpr unsigned STREAM_SELECT = 193u;
constexpr unsigned STREAM_COVERAGE = 256u;

// Camera and primary hit of a pixel
struct Domain {
    float3 x0;
    float3 x1;
    Hit hit;
};

// Sampling record of the previous path vertex for the reconnection criteria (Sec. 4)
struct Vertex {
    bool rough;
    unsigned lobe;
    float pdf;
    float cos;
};

struct Selection {
    Reservoir r;
    float w_sum;
    float3 color;
    unsigned random_state;
};

__device__ unsigned extractBits(unsigned path, unsigned offset, unsigned count) {
    return path >> offset & ((1u << count) - 1u);
}

// Nearest hit of a ray within far and its distance, NONE on a miss
__device__ Hit traverse(const Scene& scene, float3 origin, float3 direction, float far, float& distance) {
    // Traverse only; hit data is read from the hit object below.
    optixTraverse(scene.handle, origin, direction, 1e-4f, far, 0.0f, 255, OPTIX_RAY_FLAG_NONE, 0, 1, 0);
    if (!optixHitObjectIsHit()) return {NONE};
    distance = optixHitObjectGetRayTmax();
    // Barycentrics from the first two attributes; optixHitObjectGetTriangleBarycentrics returns zeros here.
    return {optixHitObjectGetPrimitiveIndex(), make_float2(__uint_as_float(optixHitObjectGetAttribute_0()),
        __uint_as_float(optixHitObjectGetAttribute_1()))};
}

__device__ float3 hitPosition(const Scene& scene, Hit hit) {
    const float3* vertices = scene.vertices + 3 * hit.triangle;
    return vertices[0] + (vertices[1] - vertices[0]) * hit.uv.x + (vertices[2] - vertices[0]) * hit.uv.y;
}

__device__ float3 faceNormal(const Scene& scene, unsigned triangle) {
    const float3* vertices = scene.vertices + 3 * triangle;
    return normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
}

__device__ float3 smoothNormal(const Scene& scene, Hit hit) {
    const float3* normals = scene.normals + 3 * hit.triangle;
    return normalize(normals[0] * (1 - hit.uv.x - hit.uv.y) + normals[1] * hit.uv.x + normals[2] * hit.uv.y);
}

// Distance along the smooth normal rays leave a hit at position from, to prevent unnatural hard shadow / reflection
// borders due to the difference between the smooth normal and geometry: the corners' distances by the tangents of their
// normals' angles to the face, limited to PI / 4, interpolated and kept within an eighth of the shortest edge
__device__ float geometryOffset(const Scene& scene, Hit hit, float3 face, float3 position) {
    const float3* vertex = scene.vertices + 3 * hit.triangle;
    const float3* normal = scene.normals + 3 * hit.triangle;
    const float weight[3] = {1 - hit.uv.x - hit.uv.y, hit.uv.x, hit.uv.y};
    float offset = 0, edge = INFINITY;
    for (int k = 0; k < 3; ++k) {
        const float3 side = vertex[(k + 1) % 3] - vertex[k];
        const float3 diff = position - vertex[k];
        const float cosine = fabsf(dot(face, normalize(normal[k])));
        edge = fminf(edge, sqrtf(dot(side, side)));
        offset += weight[k] * sqrtf(dot(diff, diff)) * fminf(sqrtf(fmaxf(1 - cosine * cosine, 0)) / cosine, PI * 0.25f);
    }
    return fminf(fmaxf(offset, 0), edge * 0.125f);
}

// Whether a direction leaves a face to the other side than reflecting or refracting the incoming direction does
__device__ bool wrongSide(float3 in_dir, bool refracted, float3 out_dir, float3 face) {
    return (dot(in_dir, face) < 0 != refracted) != (dot(out_dir, face) > 0);
}

// Camera and primary hit of the ray through a pixel at the jitter its sample's seed draws, which makes the pixel's area
// part of the sample's primary sample space; hidden surfaces let camera rays through. On a miss x1 is one along the ray.
__device__ Domain pixelDomain(const Scene& scene, const Camera& camera, int pixel, uint3 size, unsigned seed) {
    unsigned state = stream(seed, STREAM_JITTER);
    const float x = pixel % size.x + random(state);
    const float y = pixel / size.x + random(state);
    const float3 direction = normalize(camera.x * (2 * x / size.x - 1) + camera.y * (1 - 2 * y / size.y) + camera.z);
    float3 origin = camera.origin;
    Hit hit;
    do {
        float distance = 0;
        hit = traverse(scene, origin, direction, 1e16f, distance);
        origin = origin + direction * distance;
    } while (hit.triangle != NONE && scene.materials[hit.triangle].hidden);
    return {camera.origin, hit.triangle == NONE ? camera.origin + direction : hitPosition(scene, hit), hit};
}

// World: equirectangular texture along the direction
__device__ float3 environment(const Scene& scene, float3 direction) {
    return shade(scene, scene.materials[scene.count].program, direction, Matrix{},
        0.5f - atan2f(direction.y, direction.x) / (2 * PI),
        0.5f + atan2f(direction.z, sqrtf(direction.x * direction.x + direction.y * direction.y)) / PI).emissive;
}

// Of max(0, a - b) for angles a, b in [0, PI] by their sines and cosines
__device__ float cosSubClamped(float sin_a, float cos_a, float sin_b, float cos_b) {
    return cos_a > cos_b ? 1 : cos_a * cos_b + sin_a * sin_b;
}
__device__ float sinSubClamped(float sin_a, float cos_a, float sin_b, float cos_b) {
    return cos_a > cos_b ? 0 : sin_a * cos_b - cos_a * sin_b;
}
// Of an angle in [0, PI] by its cosine
__device__ float sine(float c) { return sqrtf(fmaxf(1 - c * c, 0)); }

// Importance of a light BVH node to a point: power times bounds of the cosines at its emitters, two sided and diffuse,
// and at the point, for light in front of the normal only as light sampling counts it, over the squared distance
// (pbrt-v4's LightBounds::Importance)
__device__ float importance(const LightNode& node, float3 p, float3 normal) {
    const float3 center = (node.low + node.high) * 0.5f, diagonal = node.high - node.low, wi = normalize(p - center);
    const float d2 = dot(p - center, p - center), r2 = dot(diagonal, diagonal) / 4;
    const float cos_theta_w = fabsf(dot(node.w, wi)), sin_theta_o = sine(node.cos_theta_o);
    // Directions of the bounding sphere of the bounds, all from inside it
    const float cos_theta_b = d2 < r2 ? -1 : sqrtf(fmaxf(1 - r2 / d2, 0)), sin_theta_b = sine(cos_theta_b);
    const float cos_theta_x = cosSubClamped(sine(cos_theta_w), cos_theta_w, sin_theta_o, node.cos_theta_o);
    const float sin_theta_x = sinSubClamped(sine(cos_theta_w), cos_theta_w, sin_theta_o, node.cos_theta_o);
    const float cos_theta_p = cosSubClamped(sin_theta_x, cos_theta_x, sin_theta_b, cos_theta_b);
    // Diffuse emitters spread a quarter turn: cos theta_e = 0
    if (cos_theta_p <= 0) return 0;
    const float cos_theta_i = - dot(wi, normal);
    return fmaxf(node.phi * cos_theta_p / fmaxf(d2, length(diagonal) / 2)
        * cosSubClamped(sine(cos_theta_i), cos_theta_i, sin_theta_b, cos_theta_b), 0);
}

// Light sampling: an emitting triangle picked by the light BVH, a uniform point on it, counted if visible from in front
// of the normal, from offset along it. Returns emission * the light's cosine / distance^2 / pdf arriving along
// direction, pdf = the triangle's probability / area.
__device__ float3 sampleLight(const Scene& scene, float3 origin, float3 normal, float offset, unsigned state,
    float3& direction) {
    // From the root down to either child by its share of importance, the pick rescaled to choose again (pbrt-v4's
    // BVHLightSampler::Sample)
    float pick = random(state), probability = 1;
    unsigned node = 0;
    while (!scene.lights[node].leaf) {
        const unsigned children[2] = {node + 1, scene.lights[node].child};
        const float shares[2] = {importance(scene.lights[children[0]], origin, normal),
            importance(scene.lights[children[1]], origin, normal)};
        if (shares[0] == 0 && shares[1] == 0)
            return make_float3(0, 0, 0);
        const float total = shares[0] + shares[1];
        const unsigned child = shares[1] > 0 && pick * total >= shares[0];
        pick = fminf((pick * total - child * shares[0]) / shares[child], 0x1.fffffep-1f);
        probability *= shares[child] / total;
        node = children[child];
    }
    if (node == 0 && !(importance(scene.lights[0], origin, normal) > 0))
        return make_float3(0, 0, 0);
    const unsigned light = scene.lights[node].child;
    const float3 la = scene.vertices[3 * light];
    const float3 lb = scene.vertices[3 * light + 1];
    const float3 lc = scene.vertices[3 * light + 2];
    const float3 lightNormal = cross(lb - la, lc - la);
    const float side = sqrtf(random(state));
    const float along = random(state);
    const float3 toLight = la * (1 - side) + lb * (side * (1 - along)) + lc * (side * along) - origin;
    const float distance = length(toLight);
    // Visibility along the segment from the offset start to the sampled point
    const float3 shadow = toLight - normal * offset, toward = normalize(shadow);
    float occluder, left = length(shadow) * (1 - 1e-4f);
    float3 from = origin + normal * offset, transmitted = make_float3(1, 1, 1);
    direction = toLight * (1 / distance);
    if (!(dot(normal, toLight) > 0))
        return make_float3(0, 0, 0);
    // Through occluders passing rays straight through, where uncovered or tinted by their transparency
    for (Hit hit; (hit = traverse(scene, from, toward, left, occluder)).triangle != NONE; left -= occluder) {
        const Surface surface = shade(scene, hit.triangle, hit.uv.x, hit.uv.y, smoothNormal(scene, hit));
        const float3 through = surface.ior == 1 ? surface.albedo * surface.transparency : make_float3(0, 0, 0);
        transmitted = transmitted * (through * surface.coverage + make_float3(1, 1, 1) * (1 - surface.coverage));
        if (rgb_to_greyscale(transmitted) <= 0)
            return make_float3(0, 0, 0);
        from = from + toward * occluder;
    }
    return transmitted * shade(scene, light, side * (1 - along), side * along, lightNormal).emissive
        * (fabsf(dot(lightNormal, toLight)) / (2 * probability * distance * distance * distance));
}

// Weighted reservoir sampling step, weight is the vector-valued resampling weight (Sec. 6.3)
__device__ void offer(Selection& sel, const Reservoir& candidate, float3 weight) {
    const float w = rgb_to_greyscale(weight);
    // Also rejects NaN and infinite weights of degenerate samples
    if (!(w > 0.0f && w < 3.0e38f)) {
        return;
    }
    sel.w_sum += w;
    sel.color = sel.color + weight;
    if (random(sel.random_state) * sel.w_sum <= w) {
        sel.r = candidate;
    }
}

// Resample a shifted sample with MIS weight mis, keeping its Jacobian for later shifts
__device__ void offerShifted(Selection& sel, const Reservoir& r, Shift shifted, float mis) {
    Reservoir candidate = r;
    candidate.f = shifted.f / shifted.jac;
    candidate.jac = r.jac * shifted.jac;
    offer(sel, candidate, mis * shifted.f * r.w);
}

__device__ Reservoir finalize(const Selection& sel, float c) {
    Reservoir r = sel.r;
    const float p_hat = rgb_to_greyscale(r.f);
    r.w = p_hat > 0.0f ? sel.w_sum / p_hat : 0.0f;
    r.c = c;
    return r;
}

__device__ Reservoir withEnd(const Reservoir& r, unsigned end_type, unsigned end, float3 f) {
    Reservoir candidate = r;
    candidate.path = r.path | end << PATH_END | end_type << PATH_TYPE;
    candidate.f = f;
    return candidate;
}

// Walk a path from dom. Generating, every path of the tree is resampled into sel with its reconnection
// vertex (Sec. 6.1), Russian roulette only modifies their source density (Sec. 6.2.4). Replaying, the hybrid shift maps r into
// dom (Sec. 2.3): random replay up to x_k-1 and reconnection to x_k. It returns the shifted integrand times the Jacobian
// determinant of Eq. 2, zero if the path is not in the image of the shift.
__device__ Shift walk(const Scene& scene, const Domain& dom, const Reservoir& r, bool replay, float scale,
    Selection& sel) {
    const Shift fail = Shift{make_float3(0.0f, 0.0f, 0.0f), 0.0f};
    const unsigned end = extractBits(r.path, PATH_END, 5u);
    const unsigned k = extractBits(r.path, PATH_K, 5u);
    const unsigned rc = extractBits(r.path, PATH_RC, 2u);
    const unsigned end_type = extractBits(r.path, PATH_TYPE, 3u);
    // Replaying stops sampling at x_k-1 for reconnections and at the path end otherwise
    const unsigned stop = rc == RC_NONE ? end + (end_type == END_ENV ? 1u : 0u) : k - 1u;
    float3 origin = dom.x1;
    float3 direction = normalize(dom.x1 - dom.x0);
    Hit hit = dom.hit;
    float distance = length(dom.x1 - dom.x0);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    // Replaying the densities of the reconnection, generating the product of Russian roulette survival probabilities
    float density = 1.0f;
    bool direct_light_emission = true;
    float threshold = 0.0f;
    float3 reconnection = origin;
    Vertex prev = Vertex{false, 0u, 0.0f, 0.0f};
    // Template for generated paths reconnecting at the first vertex meeting both footprint criteria
    Reservoir chain = {};
    chain.seed = r.seed;
    float3 chain_throughput = make_float3(0.0f, 0.0f, 0.0f);
    for (unsigned b = 0u; ; b++) {
        const float3 face = faceNormal(scene, hit.triangle);
        Surface surface = shade(scene, hit.triangle, hit.uv.x, hit.uv.y, smoothNormal(scene, hit));
        // Rays pass straight through the uncovered share of surfaces, which keep their emission and medium
        if (pcg(stream(r.seed, b + STREAM_COVERAGE)).value >= surface.coverage)
            surface = Surface{surface.normal, make_float3(1, 1, 1), 0, 0, surface.emissive, 1, 1,
                make_float3(1, 1, 1), 1, 1, surface.absorption, surface.density};
        // Surfaces transmitting nothing face the ray, as in Cycles
        const float facing = surface.transparency > 0 || dot(face, direction) < 0 ? 1.0f : -1.0f;
        const float3 geometry_n = face * facing;
        surface.normal = surface.normal * facing;
        const float offset = geometryOffset(scene, hit, geometry_n, origin);
        // Rays reaching the back of a face come through its medium
        if (dot(direction, geometry_n) > 0) throughput = throughput * beerLambert(distance, surface);
        const float geometry_density = fabsf(dot(geometry_n, direction)) / (distance * distance);
        if (b == 0u) {
            // Primary ray footprint (Eq. 5)
            threshold = RESTIR_FOOTPRINT / geometry_density;
        }
        // Ray footprint criterion for reconnecting from the previous vertex to this one (Eq. 5)
        const bool footprint = prev.rough && prev.pdf * geometry_density * threshold <= 1.0f;
        const float n_dot_v = dot(surface.normal, - direction);
        const bool is_entering = n_dot_v < 0.0f;
        // Incident side is air, outgoing side is material (entering)
        const float eta_i = is_entering ? surface.ior : 1.0f;
        // Incident side is material, outgoing side is air (exiting)
        const float eta_o = is_entering ? 1.0f : surface.ior;
        const bool nee = neeDecision(surface, n_dot_v, eta_i, eta_o);
        const bool chain_found = extractBits(chain.path, PATH_RC, 2u) == RC_SURFACE;
        // Reconnection at this vertex for paths ending here
        Reservoir here = chain;
        here.path = b << PATH_K | RC_SURFACE << PATH_RC | prev.lobe << PATH_LOBE_PREV;
        here.jac = prev.pdf * geometry_density;
        here.rc = hit;
        const bool arrived = replay && rc == RC_SURFACE && b == k;
        if (arrived) {
            // The stored path takes over at the reconnection vertex
            if (hit.triangle != r.rc.triangle || !footprint) {
                return fail;
            }
            throughput = throughput * geometry_density;
            density *= geometry_density;
            if (k == end && end_type == END_EMIT) {
                if (!direct_light_emission) {
                    return fail;
                }
                return Shift{throughput * surface.emissive / r.jac, density / r.jac};
            }
            if (k == end && end_type == END_NEE) {
                if (!nee) {
                    return fail;
                }
                return Shift{throughput * BSDF(direction, r.dir, surface.normal, geometry_n, surface, eta_i, eta_o,
                    stream(r.seed, b)) * r.radiance / r.jac, density / r.jac};
            }
            if (nee != (extractBits(r.path, PATH_NEE, 1u) == 1u)) {
                return fail;
            }
        } else if (replay && b == stop && rc != RC_SURFACE && rc != RC_ENV) {
            if (rc == RC_LIGHT && nee && !footprint) {
                // Forced reconnection to the light vertex of next event estimation (Sec. 6.2.3), the point its replayed
                // random numbers pick again (unit Jacobian)
                float3 light_dir;
                const float3 radiance = sampleLight(scene, origin, surface.normal, offset,
                    stream(r.seed, b + STREAM_NEE), light_dir);
                return rgb_to_greyscale(radiance) > 0.0f ? Shift{throughput * BSDF(direction, light_dir, surface.normal,
                    geometry_n, surface, eta_i, eta_o, stream(r.seed, b)) * radiance, 1.0f} : fail;
            }
            // Emission seen right after a vertex that skipped next event estimation
            if (rc == RC_NONE && end_type == END_EMIT && direct_light_emission && !footprint) {
                return Shift{throughput * surface.emissive, 1.0f};
            }
            return fail;
        } else if (!replay) {
            // Emission seen directly carries no randomness to resample, shading adds it
            if (direct_light_emission && b > 0u) {
                Reservoir candidate = chain;
                if (chain_found) {
                    candidate.radiance = safeDivide(throughput * surface.emissive, chain_throughput);
                } else if (footprint) {
                    candidate = here;
                }
                offer(sel, withEnd(candidate, END_EMIT, b, throughput * surface.emissive),
                    throughput * surface.emissive * scale / density);
            }
            if (nee) {
                float3 light_dir;
                const float3 radiance = sampleLight(scene, origin, surface.normal, offset,
                    stream(r.seed, b + STREAM_NEE), light_dir);
                const float3 f = rgb_to_greyscale(radiance) > 0.0f ? throughput * BSDF(direction, light_dir,
                    surface.normal, geometry_n, surface, eta_i, eta_o, stream(r.seed, b)) * radiance : fail.f;
                Reservoir candidate = chain;
                if (chain_found) {
                    candidate.radiance = safeDivide(f, chain_throughput);
                } else if (footprint) {
                    candidate = here;
                    candidate.path |= 1u << PATH_NEE;
                    candidate.dir = light_dir;
                    candidate.radiance = radiance;
                } else {
                    // Forced reconnection to the light vertex (Sec. 6.2.3)
                    candidate.path = (b + 1u) << PATH_K | RC_LIGHT << PATH_RC;
                }
                offer(sel, withEnd(candidate, END_NEE, b, f), f * scale / density);
            }
        }
        // Replaying, x_k-1 connects to the reconnection vertex and x_k leaves along the stored direction with the stored lobes
        const bool connect = replay && b == stop;
        unsigned lobe_index = extractBits(r.path, arrived ? PATH_LOBE : PATH_LOBE_PREV, 2u);
        float3 dir = r.dir;
        // Sample leaving this vertex
        SampleBSDF lobe = SampleBSDF{};
        if (connect) {
            if (rc == RC_SURFACE) {
                reconnection = hitPosition(scene, r.rc);
                dir = normalize(reconnection - (origin
                    + surface.normal * (offset * sign(dot(reconnection - origin, geometry_n)))));
            }
            // The path tracer would resample such a direction against the geometry normal
            if (wrongSide(direction, lobe_index == LOBE_REFRACT, dir, geometry_n)) {
                return fail;
            }
        } else if (!arrived) {
            SampleBSDF bsdf_sampled = sampleBSDF(direction, surface.normal, surface, eta_i, eta_o,
                stream(r.seed, b), make_float2(0, 0));
            // Continue sampling with the face normal if the smooth normal sent the ray to the wrong side of the face,
            // pretending the self reflection faces along it
            const bool fallback = wrongSide(direction, bsdf_sampled.refracted, bsdf_sampled.unit_direction, geometry_n);
            if (fallback) {
                const SampleBSDF geometry_bsdf_sampled = sampleBSDF(bsdf_sampled.unit_direction, geometry_n, surface,
                    eta_i, eta_o, bsdf_sampled.random_state, make_float2(0, 0));
                bsdf_sampled.unit_direction = geometry_bsdf_sampled.unit_direction;
                bsdf_sampled.throughput = geometry_bsdf_sampled.throughput;
            }
            lobe_index = bsdf_sampled.lobe;
            dir = bsdf_sampled.unit_direction;
            lobe = bsdf_sampled;
            // Density of the direction is unknown after the fallback
            lobe.pdf = fallback ? 0.0f : bsdf_sampled.pdf;
        }
        // Rays passing straight through stay where light sampling, which passes them too, last was or was not
        direct_light_emission = !nee && (direct_light_emission || lobe_index != LOBE_REFRACT || surface.ior != 1);
        // Connections evaluate the given direction
        if (connect || arrived) {
            lobe = connectBSDF(direction, dir, surface.normal, surface, eta_i, eta_o, lobe_index, stream(r.seed, b));
        }
        // Inverse ray footprint criterion for the density of the direction leaving this vertex (Eq. 5)
        const bool inverse_footprint = lobe.pdf > 0.0f
            && (lobe_index == LOBE_DIFFUSE || lobe.pdf * prev.cos / (distance * distance) * threshold <= 1.0f);
        const bool rough = lobe.pdf > 0.0f && (lobe_index == LOBE_DIFFUSE || surface.roughness >= RESTIR_ROUGHNESS);
        if (arrived) {
            if (!inverse_footprint) {
                return fail;
            }
            return Shift{throughput * (lobe.throughput * lobe.pdf) * r.radiance / r.jac, density * lobe.pdf / r.jac};
        }
        // A replayed path meeting the criteria before x_k is not in the image of the shift
        if ((replay && footprint && inverse_footprint) || (connect && !rough)) {
            return fail;
        }
        if (connect) {
            throughput = throughput * (lobe.throughput * lobe.pdf);
            density = lobe.pdf;
        } else {
            throughput = throughput * make_float3(fmaxf(lobe.throughput.x, 0.0f), fmaxf(lobe.throughput.y, 0.0f),
                fmaxf(lobe.throughput.z, 0.0f));
        }
        if (!replay && !chain_found && footprint && inverse_footprint) {
            chain = here;
            chain.path |= lobe_index << PATH_LOBE | (nee ? 1u : 0u) << PATH_NEE;
            chain.jac *= lobe.pdf;
            chain.dir = dir;
            chain_throughput = throughput;
        }
        prev = Vertex{rough, lobe_index, lobe.pdf, fabsf(dot(geometry_n, dir))};
        direction = dir;
        // Off the side it leaves through
        origin = origin + surface.normal * (offset * sign(dot(direction, geometry_n)));
        if (!replay) {
            const float survival_probability = fminf(fmaxf(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z))
                / density, 0.0f), 1.0f);
            if (survival_probability < pcg(stream(r.seed, b + STREAM_RR)).value || b >= scene.settings.bounces) {
                break;
            }
            density *= survival_probability;
        }
        // Only occluders in front of the reconnection vertex matter
        const bool reconnecting = replay && b == stop && rc == RC_SURFACE;
        hit = traverse(scene, origin, direction, reconnecting ? length(reconnection - origin) * 1.001f : 1e16f,
            distance);
        // Reaching the reconnection vertex, it is shaded exactly where the stored path has it
        if (reconnecting && hit.triangle == r.rc.triangle) {
            hit.uv = r.rc.uv;
            distance = length(reconnection - origin);
        }
        // The connection to the environment direction has to escape the scene
        if (replay && b == stop && rc == RC_ENV && hit.triangle != NONE) {
            return fail;
        }
        if (hit.triangle == NONE) {
            const float3 f = throughput * environment(scene, direction);
            if (replay) {
                if (b == stop && rc == RC_ENV) {
                    return Shift{f / r.jac, density / r.jac};
                }
                if (rc == RC_NONE && end_type == END_ENV && b == end && !prev.rough) {
                    return Shift{f, 1.0f};
                }
                return fail;
            }
            Reservoir candidate = chain;
            if (extractBits(chain.path, PATH_RC, 2u) == RC_SURFACE) {
                candidate.radiance = safeDivide(f, chain_throughput);
            } else if (prev.rough) {
                // Reconnection to the environment direction, guarded by the roughness threshold only (Sec. 4.2)
                candidate.path = (b + 1u) << PATH_K | RC_ENV << PATH_RC | prev.lobe << PATH_LOBE_PREV;
                candidate.jac = prev.pdf;
                candidate.dir = direction;
            }
            offer(sel, withEnd(candidate, END_ENV, b, f), f * scale / density);
            break;
        }
        // Project ray origin to hit point
        origin = origin + direction * distance;
    }
    return fail;
}

// The sample of r shifted into the pixel the camera sees, zero if its primary ray misses there
__device__ Shift shift(const Scene& scene, const Camera& camera, int pixel, uint3 size, const Reservoir& r) {
    Selection sel;
    if (r.w <= 0.0f) {
        return Shift{make_float3(0.0f, 0.0f, 0.0f), 0.0f};
    }
    const Domain dom = pixelDomain(scene, camera, pixel, size, r.seed);
    if (dom.hit.triangle == NONE) {
        return Shift{make_float3(0.0f, 0.0f, 0.0f), 0.0f};
    }
    return walk(scene, dom, r, true, 0.0f, sel);
}

// Pixel coordinates a camera sees a position at
__device__ float2 project(const Camera& camera, float3 position, uint3 size) {
    const float3 d = position - camera.origin;
    // d = (x * u + y * v + z) * depth
    const float depth = dot(cross(camera.x, camera.y), d);
    return make_float2((1 + dot(cross(d, camera.y), camera.z) / depth) * 0.5f * size.x,
        (1 - dot(cross(camera.x, d), camera.z) / depth) * 0.5f * size.y);
}

// Index of the pixel at q if the previous frame left it a sample, -1 otherwise
__device__ int previousValid(const Scene& scene, float2 q, uint3 size) {
    const int x = int(floorf(q.x));
    const int y = int(floorf(q.y));
    const bool valid = x >= 0 && y >= 0 && x < int(size.x) && y < int(size.y)
        && scene.previous[y * size.x + x].c > 0.0f;
    return valid ? y * int(size.x) + x : -1;
}

// Backprojection of the camera motion, or the dual motion vector of the surface previously seen there (Sec. 6.4)
__device__ int temporalNeighbor(const Scene& scene, uint3 p, uint3 size, const Domain& dom) {
    const float2 q = project(scene.previousCamera, dom.x1, size);
    const int index = previousValid(scene, q, size);
    if (index < 0) {
        return -1;
    }
    const float3 previous = pixelDomain(scene, scene.previousCamera, index, size, scene.previous[index].seed).x1;
    if (length(previous - dom.x1) <= length(dom.x1 - dom.x0) * 8.0f / size.x) {
        return index;
    }
    const float2 moved = project(scene.camera, previous, size);
    return previousValid(scene, make_float2(p.x + floorf(q.x) + 1.0f - moved.x, p.y + floorf(q.y) + 1.0f - moved.y),
        size);
}

// Index of the partner of p in reuse texture t, randomly flipped, transposed and offset every frame (Sec. 3.2),
// -1 off the image
__device__ int partner(const Scene& scene, uint3 p, uint3 size, unsigned t) {
    const int sizes[] = {PAIRING_SIZES[0], PAIRING_SIZES[1], PAIRING_SIZES[2]};
    const int texture = sizes[t];
    const unsigned random = hash(scene.frame * 3u + t);
    const int flip_x = (random & 1u) != 0u ? -1 : 1;
    const int flip_y = (random & 2u) != 0u ? -1 : 1;
    const bool transpose = (random & 4u) != 0u;
    const int x = ((int(transpose ? p.y : p.x) * flip_x + int((random >> 8u) & 255u)) % texture + texture) % texture;
    const int y = ((int(transpose ? p.x : p.y) * flip_y + int((random >> 16u) & 255u)) % texture + texture) % texture;
    const char2 delta = scene.pairing[t][y * texture + x];
    const int qx = int(p.x) + (transpose ? delta.y * flip_y : delta.x * flip_x);
    const int qy = int(p.y) + (transpose ? delta.x * flip_x : delta.y * flip_y);
    const bool valid = qx >= 0 && qy >= 0 && qx < int(size.x) && qy < int(size.y);
    return valid ? qy * int(size.x) + qx : -1;
}

// Initial candidates: the path trees of the pixel's primary hits, each at a jitter of its own; it adds the radiance the
// camera sees right there, of an emitter or the world, as it carries no randomness to resample. Without ReSTIR it adds
// the path trees' estimate as well.
extern "C" __global__ void __raygen__initial() {
    const uint3 p = optixGetLaunchIndex();
    const uint3 size = optixGetLaunchDimensions();
    const unsigned index = p.y * size.x + p.x;
    const Scene& scene = *(Scene*)optixGetSbtDataPointer();
    const unsigned pixel_seed = stream(hash(scene.frame), index);
    Selection initial = {};
    initial.random_state = stream(pixel_seed, STREAM_SELECT);
    // The pixel's own seed even without a sample, its domain for later passes
    initial.r.seed = stream(pixel_seed, 0u);
    float3 seen = make_float3(0.0f, 0.0f, 0.0f);
    for (unsigned i = 0u; i < scene.settings.paths; i++) {
        Reservoir r = {};
        r.seed = stream(pixel_seed, i);
        const Domain dom = pixelDomain(scene, scene.camera, index, size, r.seed);
        if (dom.hit.triangle != NONE) {
            walk(scene, dom, r, false, 1.0f / scene.settings.paths, initial);
        }
        seen = seen + (dom.hit.triangle == NONE ? environment(scene, normalize(dom.x1 - dom.x0))
            : shade(scene, dom.hit.triangle, dom.hit.uv.x, dom.hit.uv.y, smoothNormal(scene, dom.hit)).emissive)
            / scene.settings.paths;
    }
    scene.initial[index] = finalize(initial, 1.0f);
    const float3 radiance = scene.settings.restir ? seen : seen + initial.color;
    float4& sum = scene.accumulation[index];
    sum = make_float4(sum.x + radiance.x, sum.y + radiance.y, sum.z + radiance.z, sum.w + 1);
}

// Temporal reuse: the initial sample against the one the previous frame left the pixel's temporal neighbor, which is
// the pixel itself under a still camera, where both shifts are the identity
extern "C" __global__ void __raygen__temporal() {
    const uint3 p = optixGetLaunchIndex();
    const uint3 size = optixGetLaunchDimensions();
    const unsigned index = p.y * size.x + p.x;
    const Scene& scene = *(Scene*)optixGetSbtDataPointer();
    Reservoir r = scene.initial[index];
    const bool still = scene.camera.origin == scene.previousCamera.origin && scene.camera.x == scene.previousCamera.x
        && scene.camera.y == scene.previousCamera.y && scene.camera.z == scene.previousCamera.z;
    int q = scene.previous[index].c > 0.0f ? int(index) : -1;
    if (!still) {
        const Domain dom = pixelDomain(scene, scene.camera, index, size, r.seed);
        q = dom.hit.triangle == NONE ? -1 : temporalNeighbor(scene, p, size, dom);
    }
    if (q >= 0) {
        const Reservoir temporal = scene.previous[q];
        // Optionally lower the confidence cap where the previous frame duplicated samples (Sec. 5), this biases the estimate
        const float duplicated = scene.settings.decorrelation ? scene.duplication[q] : 0.0f;
        const float cap = RESTIR_CAP + (RESTIR_CAP_MIN - RESTIR_CAP) * powf(duplicated, RESTIR_DUPLICATION_ALPHA);
        const float c_t = fminf(temporal.c, cap);
        const Shift to_current = still ? Shift{temporal.f, 1.0f} : shift(scene, scene.camera, index, size, temporal);
        const Shift to_previous = still ? Shift{r.f, 1.0f} : shift(scene, scene.previousCamera, q, size, r);
        const float p_c = rgb_to_greyscale(r.f);
        const float p_t = rgb_to_greyscale(temporal.f);
        Selection sel = {};
        sel.random_state = stream(r.seed, STREAM_SELECT + 1u);
        offerShifted(sel, r, Shift{r.f, 1.0f}, r.c * p_c / (r.c * p_c + c_t * rgb_to_greyscale(to_previous.f)));
        offerShifted(sel, temporal, to_current, c_t * p_t / (c_t * p_t + r.c * rgb_to_greyscale(to_current.f)));
        r = finalize(sel, r.c + c_t);
    }
    scene.temporal[index] = r;
}

// Paired shifts (Sec. 3): the pixel's sample into the domain of its partner in the reuse texture of the launch's layer
extern "C" __global__ void __raygen__shift() {
    const uint3 p = optixGetLaunchIndex();
    const uint3 size = optixGetLaunchDimensions();
    const unsigned index = p.y * size.x + p.x;
    const Scene& scene = *(Scene*)optixGetSbtDataPointer();
    const int q = partner(scene, p, size, p.z);
    scene.shifts[3 * index + p.z] = q < 0 ? Shift{make_float3(0.0f, 0.0f, 0.0f), 0.0f}
        : shift(scene, scene.camera, q, size, scene.temporal[index]);
}

// Paired spatial reuse with pairwise MIS (Sec. 3) and shading with vector-valued weights (Sec. 6.3)
extern "C" __global__ void __raygen__spatial() {
    const uint3 p = optixGetLaunchIndex();
    const uint3 size = optixGetLaunchDimensions();
    const unsigned index = p.y * size.x + p.x;
    const Scene& scene = *(Scene*)optixGetSbtDataPointer();
    const Reservoir r = scene.temporal[index];
    float count = 0.0f;
    float c_sum = r.c;
    for (unsigned t = 0u; t < scene.settings.neighbors; t++) {
        const int q = partner(scene, p, size, t);
        if (q >= 0) {
            count += 1.0f;
            c_sum += scene.temporal[q].c;
        }
    }
    // Confidence of the current pixel is split evenly across the pairs
    const float c_pair = r.c / fmaxf(count, 1.0f);
    const float p_c = rgb_to_greyscale(r.f);
    float canonical_mis = count == 0.0f ? 1.0f : 0.0f;
    Selection sel = {};
    sel.random_state = stream(scene.initial[index].seed, STREAM_SELECT + 2u);
    for (unsigned t = 0u; t < scene.settings.neighbors; t++) {
        const int q = partner(scene, p, size, t);
        if (q >= 0) {
            const Reservoir neighbor = scene.temporal[q];
            const Shift to_here = scene.shifts[3 * q + t];
            const Shift to_there = scene.shifts[3 * index + t];
            const float share = (neighbor.c + c_pair) / c_sum;
            const float p_n = rgb_to_greyscale(neighbor.f);
            if (p_c > 0.0f) {
                canonical_mis += share * c_pair * p_c / (c_pair * p_c + neighbor.c * rgb_to_greyscale(to_there.f));
            }
            offerShifted(sel, neighbor, to_here,
                share * neighbor.c * p_n / (neighbor.c * p_n + c_pair * rgb_to_greyscale(to_here.f)));
        }
    }
    offerShifted(sel, r, Shift{r.f, 1.0f}, canonical_mis);
    scene.previous[index] = finalize(sel, c_sum);
    float4& sum = scene.accumulation[index];
    sum = make_float4(sum.x + sel.color.x, sum.y + sel.color.y, sum.z + sel.color.z, sum.w);
}

// Fraction of the 17 x 17 neighborhood sharing each pixel's sample (Sec. 5)
extern "C" __global__ void __raygen__duplicates() {
    const uint3 p = optixGetLaunchIndex();
    const uint3 size = optixGetLaunchDimensions();
    const unsigned index = p.y * size.x + p.x;
    const Scene& scene = *(Scene*)optixGetSbtDataPointer();
    const Reservoir& own = scene.previous[index];
    unsigned count = 0u;
    for (int y = int(p.y) - 8; y <= int(p.y) + 8; y++) {
        for (int x = int(p.x) - 8; x <= int(p.x) + 8; x++) {
            if (x >= 0 && y >= 0 && x < int(size.x) && y < int(size.y)) {
                const Reservoir& other = scene.previous[y * size.x + x];
                count += other.w > 0.0f && other.seed == own.seed ? 1u : 0u;
            }
        }
    }
    scene.duplication[index] = own.w > 0.0f ? (count - 1u) / 288.0f : 0.0f;
}
