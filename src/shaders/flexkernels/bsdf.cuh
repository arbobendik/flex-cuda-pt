#pragma once
#define Matrix SlangMatrix
#include "slang-cuda-prelude.h"
#undef Matrix

static __device__ float clamp_0(float x_0, float minBound_0, float maxBound_0)
{
    return (F32_min(((F32_max((x_0), (minBound_0)))), (maxBound_0)));
}

static __device__ float dot_0(float3  x_1, float3  y_0)
{
    return x_1.x * y_0.x + x_1.y * y_0.y + x_1.z * y_0.z;
}

static __device__ float3  cross_0(float3  left_0, float3  right_0)
{
    float _S1 = left_0.y;
    float _S2 = right_0.z;
    float _S3 = left_0.z;
    float _S4 = right_0.y;
    float _S5 = right_0.x;
    float _S6 = left_0.x;
    return make_float3 (_S1 * _S2 - _S3 * _S4, _S3 * _S5 - _S6 * _S2, _S6 * _S4 - _S1 * _S5);
}

__device__ float trowbridgeReitz(float alpha_0, float n_dot_h_0)
{
    float _S7 = alpha_0 * alpha_0;
    float _S8 = n_dot_h_0 * n_dot_h_0 * (_S7 - 1.0f) + 1.0f;
    return _S7 / (F32_max((3.14159274101257324f * _S8 * _S8), (0.0000152587890625f)));
}

__device__ float smithG1(float alpha_1, float n_dot_x_0)
{
    float _S9 = alpha_1 * alpha_1;
    return 2.0f * n_dot_x_0 / (n_dot_x_0 + (F32_sqrt((_S9 + (1.0f - _S9) * n_dot_x_0 * n_dot_x_0))));
}

__device__ float fresnel(float cos_theta_0, float eta_i_0, float eta_o_0)
{
    float _S10 = (F32_sqrt(((F32_max((0.0f), (1.0f - cos_theta_0 * cos_theta_0))))));
    float _S11 = eta_i_0 / eta_o_0 * _S10;
    bool _S12 = _S11 < 1.0f;
    float kr_0;
    if(_S12)
    {
        float _S13 = (F32_sqrt(((F32_max((0.0f), (1.0f - _S11 * _S11))))));
        float _S14 = eta_o_0 * cos_theta_0;
        float _S15 = eta_i_0 * _S13;
        float _S16 = (_S14 - _S15) / (_S14 + _S15);
        float _S17 = eta_i_0 * cos_theta_0;
        float _S18 = eta_o_0 * _S13;
        float _S19 = (_S17 - _S18) / (_S17 + _S18);
        float _S20 = (_S16 * _S16 + _S19 * _S19) / 2.0f;
        kr_0 = _S20;
    }
    else
    {
        kr_0 = 1.0f;
    }
    return kr_0;
}

static __device__ float unshared_0(float x_2)
{
    return x_2;
}

__device__ float3  sampleGGXVNDF(float3  Ve_0, float alpha_2, float U1_0, float U2_0)
{
    float3  _S21 = make_float3 (alpha_2 * Ve_0.x, alpha_2 * Ve_0.y, Ve_0.z);
    float3  _S22 = ((((_S21)) * make_float3(rnorm3df(((_S21)).x, ((_S21)).y, ((_S21)).z))));
    float _S23 = _S22.x;
    float _S24 = _S22.y;
    float _S25 = _S23 * _S23 + _S24 * _S24;
    float3  _S26 = make_float3 (1.0f, 0.0f, 0.0f);
    float3  _S27 = make_float3 (- _S24, _S23, 0.0f);
    float _S28 = (rsqrtf((_S25)));
    float3  _S29 = _S27 * make_float3 (_S28);
    bool _S30 = _S25 > 0.0f;
    float3  _S31 = _slang_select(_S30, _S29,_S26);
    float3  _S32 = cross_0(_S22, _S31);
    float _S33 = (F32_sqrt((U1_0)));
    float _S34 = 6.28318548202514648f * U2_0;
    float _S35 = _S33 * (F32_cos((_S34)));
    float t2_0 = _S33 * (F32_sin((_S34)));
    float _S36 = 0.5f * (1.0f + _S22.z);
    float t2_1 = (1.0f - _S36) * (F32_sqrt(((F32_max((0.0f), (1.0f - _S35 * _S35)))))) + _S36 * t2_0;
    float3  _S37 = make_float3 (_S35) * _S31 + make_float3 (t2_1) * _S32;
    float _S38 = unshared_0(_S35);
    float _S39 = unshared_0(_S35);
    float3  _S40 = _S37 + make_float3 ((F32_sqrt(((F32_max((0.0f), (1.0f - _S38 * _S39 - t2_1 * t2_1))))))) * _S22;
    float3  _S41 = make_float3 (alpha_2 * _S40.x, alpha_2 * _S40.y, (F32_max((0.0f), (_S40.z))));
    float3  _S42 = ((((_S41)) * make_float3(rnorm3df(((_S41)).x, ((_S41)).y, ((_S41)).z))));
    return _S42;
}

__device__ float3  sampleCosWeightedHemisphere(float random_1_0, float random_2_0)
{
    float _S43 = (F32_sqrt((random_1_0)));
    float _S44 = 6.28318548202514648f * random_2_0;
    float _S45 = _S43 * (F32_cos((_S44)));
    float _S46 = _S43 * (F32_sin((_S44)));
    float _S47 = (F32_sqrt(((F32_max((0.0f), (1.0f - _S45 * _S45 - _S46 * _S46))))));
    return make_float3 (_S45, _S47, _S46);
}

__device__ float3  tangentToWorld(float3  v_0, float3  n_0)
{
    float3  a_0 = make_float3 (0.0f, 1.0f, 0.0f);
    bool _S48 = (F32_abs((dot_0(n_0, a_0)))) > 0.9999847412109375f;
    float3  a_1;
    if(_S48)
    {
        float3  _S49 = make_float3 (1.0f, 0.0f, 0.0f);
        a_1 = _S49;
    }
    else
    {
        a_1 = a_0;
    }
    float3  _S50 = cross_0(n_0, a_1);
    float3  _S51 = ((((_S50)) * make_float3(rnorm3df(((_S50)).x, ((_S50)).y, ((_S50)).z))));
    float3  _S52 = cross_0(n_0, _S51);
    return make_float3 (v_0.x) * _S51 + make_float3 (v_0.y) * n_0 + make_float3 (v_0.z) * _S52;
}

__device__ float3  worldToTangent(float3  v_1, float3  n_1)
{
    float3  a_2 = make_float3 (0.0f, 1.0f, 0.0f);
    bool _S53 = (F32_abs((dot_0(n_1, a_2)))) > 0.9999847412109375f;
    float3  a_3;
    if(_S53)
    {
        float3  _S54 = make_float3 (1.0f, 0.0f, 0.0f);
        a_3 = _S54;
    }
    else
    {
        a_3 = a_2;
    }
    float3  _S55 = cross_0(n_1, a_3);
    float3  _S56 = ((((_S55)) * make_float3(rnorm3df(((_S55)).x, ((_S55)).y, ((_S55)).z))));
    float3  _S57 = cross_0(n_1, _S56);
    return make_float3 (dot_0(v_1, _S56), dot_0(v_1, n_1), dot_0(v_1, _S57));
}

struct Random
{
    uint state;
    float value;
};

static __device__ Random Random_x24init_0(uint state_0, float value_0)
{
    Random _S58;
    (&_S58)->state = state_0;
    (&_S58)->value = value_0;
    return _S58;
}

__device__ uint hash(uint x_3)
{
    uint _S59 = x_3 * 747796405U + 2891336453U;
    uint _S60 = ((_S59 >> ((_S59 >> 28U) + 4U)) ^ _S59) * 277803737U;
    return (_S60 >> 22U) ^ _S60;
}

static __device__ float random_0(uint * state_1)
{
    uint _S61 = hash(*state_1);
    *state_1 = _S61;
    return float((*state_1) >> int(8)) * 5.9604644775390625e-08f;
}

__device__ Random pcg(uint state_2)
{
    uint _S62 = state_2;
    float value_1 = random_0(&_S62);
    return Random_x24init_0(_S62, value_1);
}

__device__ float3  sampleHalfVector(float3  v_2, float3  n_i_0, float alpha_3, uint * random_state_0)
{
    Random _S63 = pcg(*random_state_0);
    Random _S64 = pcg(_S63.state);
    *random_state_0 = _S64.state;
    float3  _S65 = worldToTangent(v_2, n_i_0);
    float3  _S66 = sampleGGXVNDF(float3 {_S65.x, _S65.z, _S65.y}, alpha_3, _S63.value, _S64.value);
    float3  _S67 = float3 {_S66.x, _S66.z, _S66.y};
    float3  _S68 = tangentToWorld(_S67, n_i_0);
    return _S68;
}

__device__ float3  safeDivide(float3  a_4, float3  b_0)
{
    float3  _S69 = a_4 / b_0;
    float3  _S70 = make_float3 (0.0f);
    bool3  _S71 = b_0 == _S70;
    float3  _S72 = _slang_select(_S71, _S70,_S69);
    return _S72;
}

__device__ float rgb_to_greyscale(float3  rgb_0)
{
    return dot_0(rgb_0, make_float3 (0.29899999499320984f, 0.58700001239776611f, 0.11400000005960464f));
}

__device__ Random normal_distribution(uint state_3)
{
    Random _S73 = pcg(state_3);
    Random _S74 = pcg(_S73.state);
    float _S75 = 6.28318548202514648f * _S73.value;
    float _S76 = (F32_sqrt((-2.0f * clamp_0((F32_log((_S74.value))), -8.388607e+06f, 0.0f))));
    return Random_x24init_0(_S74.state, _S76 * (F32_cos((_S75))));
}

struct RandomSphere
{
    uint state;
    float3  value;
};

static __device__ RandomSphere RandomSphere_x24init_0(uint state_4, float3  value_2)
{
    RandomSphere _S77;
    (&_S77)->state = state_4;
    (&_S77)->value = value_2;
    return _S77;
}

__device__ RandomSphere random_sphere(uint state_5)
{
    Random _S78 = normal_distribution(state_5);
    Random _S79 = normal_distribution(_S78.state);
    Random _S80 = normal_distribution(_S79.state);
    float3  _S81 = make_float3 (_S78.value, _S79.value, _S80.value);
    float3  _S82 = ((((_S81)) * make_float3(rnorm3df(((_S81)).x, ((_S81)).y, ((_S81)).z))));
    return RandomSphere_x24init_0(_S80.state, _S82);
}

struct RandomHemisphere
{
    uint state;
    float3  value;
};

static __device__ RandomHemisphere RandomHemisphere_x24init_0(uint state_6, float3  value_3)
{
    RandomHemisphere _S83;
    (&_S83)->state = state_6;
    (&_S83)->value = value_3;
    return _S83;
}

__device__ RandomHemisphere random_hemisphere(uint state_7, float3  normal_0)
{
    RandomSphere _S84 = random_sphere(state_7);
    bool _S85 = (dot_0(_S84.value, normal_0)) > 0.0f;
    if(_S85)
    {
        return RandomHemisphere_x24init_0(_S84.state, _S84.value);
    }
    else
    {
        return RandomHemisphere_x24init_0(_S84.state, - _S84.value);
    }
}

__device__ uint stream(uint seed_0, uint index_0)
{
    uint _S86 = hash(index_0);
    uint _S87 = hash(seed_0 ^ _S86);
    return _S87;
}

