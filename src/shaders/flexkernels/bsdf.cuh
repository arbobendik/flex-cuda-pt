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
    float3  _S26;
    if(_S25 > 0.0f)
    {
        float3  _S27 = make_float3 (- _S24, _S23, 0.0f);
        float _S28 = (rsqrtf((_S25)));
        _S26 = _S27 * make_float3 (_S28);
    }
    else
    {
        _S26 = make_float3 (1.0f, 0.0f, 0.0f);
    }
    float3  _S29 = cross_0(_S22, _S26);
    float _S30 = (F32_sqrt((U1_0)));
    float _S31 = 6.28318548202514648f * U2_0;
    float _S32 = _S30 * (F32_cos((_S31)));
    float t2_0 = _S30 * (F32_sin((_S31)));
    float _S33 = 0.5f * (1.0f + _S22.z);
    float t2_1 = (1.0f - _S33) * (F32_sqrt(((F32_max((0.0f), (1.0f - _S32 * _S32)))))) + _S33 * t2_0;
    float3  _S34 = make_float3 (_S32) * _S26 + make_float3 (t2_1) * _S29;
    float _S35 = unshared_0(_S32);
    float _S36 = unshared_0(_S32);
    float3  _S37 = _S34 + make_float3 ((F32_sqrt(((F32_max((0.0f), (1.0f - _S35 * _S36 - t2_1 * t2_1))))))) * _S22;
    float3  _S38 = make_float3 (alpha_2 * _S37.x, alpha_2 * _S37.y, (F32_max((0.0f), (_S37.z))));
    float3  _S39 = ((((_S38)) * make_float3(rnorm3df(((_S38)).x, ((_S38)).y, ((_S38)).z))));
    return _S39;
}

__device__ float3  sampleCosWeightedHemisphere(float random_1_0, float random_2_0)
{
    float _S40 = (F32_sqrt((random_1_0)));
    float _S41 = 6.28318548202514648f * random_2_0;
    float _S42 = _S40 * (F32_cos((_S41)));
    float _S43 = _S40 * (F32_sin((_S41)));
    float _S44 = (F32_sqrt(((F32_max((0.0f), (1.0f - _S42 * _S42 - _S43 * _S43))))));
    return make_float3 (_S42, _S44, _S43);
}

__device__ float3  tangentToWorld(float3  v_0, float3  n_0)
{
    float3  a_0 = make_float3 (0.0f, 1.0f, 0.0f);
    bool _S45 = (F32_abs((dot_0(n_0, a_0)))) > 0.9999847412109375f;
    float3  a_1;
    if(_S45)
    {
        float3  _S46 = make_float3 (1.0f, 0.0f, 0.0f);
        a_1 = _S46;
    }
    else
    {
        a_1 = a_0;
    }
    float3  _S47 = cross_0(n_0, a_1);
    float3  _S48 = ((((_S47)) * make_float3(rnorm3df(((_S47)).x, ((_S47)).y, ((_S47)).z))));
    float3  _S49 = cross_0(n_0, _S48);
    return make_float3 (v_0.x) * _S48 + make_float3 (v_0.y) * n_0 + make_float3 (v_0.z) * _S49;
}

__device__ float3  worldToTangent(float3  v_1, float3  n_1)
{
    float3  a_2 = make_float3 (0.0f, 1.0f, 0.0f);
    bool _S50 = (F32_abs((dot_0(n_1, a_2)))) > 0.9999847412109375f;
    float3  a_3;
    if(_S50)
    {
        float3  _S51 = make_float3 (1.0f, 0.0f, 0.0f);
        a_3 = _S51;
    }
    else
    {
        a_3 = a_2;
    }
    float3  _S52 = cross_0(n_1, a_3);
    float3  _S53 = ((((_S52)) * make_float3(rnorm3df(((_S52)).x, ((_S52)).y, ((_S52)).z))));
    float3  _S54 = cross_0(n_1, _S53);
    return make_float3 (dot_0(v_1, _S53), dot_0(v_1, n_1), dot_0(v_1, _S54));
}

struct Random
{
    uint state;
    float value;
};

static __device__ Random Random_x24init_0(uint state_0, float value_0)
{
    Random _S55;
    (&_S55)->state = state_0;
    (&_S55)->value = value_0;
    return _S55;
}

__device__ uint hash(uint x_3)
{
    uint _S56 = x_3 * 747796405U + 2891336453U;
    uint _S57 = ((_S56 >> ((_S56 >> 28U) + 4U)) ^ _S56) * 277803737U;
    return (_S57 >> 22U) ^ _S57;
}

static __device__ float random_0(uint * state_1)
{
    uint _S58 = hash(*state_1);
    *state_1 = _S58;
    return float((*state_1) >> int(8)) * 5.9604644775390625e-08f;
}

__device__ Random pcg(uint state_2)
{
    uint _S59 = state_2;
    float value_1 = random_0(&_S59);
    return Random_x24init_0(_S59, value_1);
}

__device__ float3  sampleHalfVector(float3  v_2, float3  n_i_0, float alpha_3, uint * random_state_0)
{
    Random _S60 = pcg(*random_state_0);
    Random _S61 = pcg(_S60.state);
    *random_state_0 = _S61.state;
    float3  _S62 = worldToTangent(v_2, n_i_0);
    float3  _S63 = sampleGGXVNDF(float3 {_S62.x, _S62.z, _S62.y}, alpha_3, _S60.value, _S61.value);
    float3  _S64 = float3 {_S63.x, _S63.z, _S63.y};
    float3  _S65 = tangentToWorld(_S64, n_i_0);
    return _S65;
}

__device__ float3  safeDivide(float3  a_4, float3  b_0)
{
    float3  _S66 = a_4 / b_0;
    float3  _S67 = make_float3 (0.0f);
    bool3  _S68 = b_0 == _S67;
    float3  _S69 = _slang_select(_S68, _S67,_S66);
    return _S69;
}

__device__ float rgb_to_greyscale(float3  rgb_0)
{
    return dot_0(rgb_0, make_float3 (0.29899999499320984f, 0.58700001239776611f, 0.11400000005960464f));
}

__device__ Random normal_distribution(uint state_3)
{
    Random _S70 = pcg(state_3);
    Random _S71 = pcg(_S70.state);
    float _S72 = 6.28318548202514648f * _S70.value;
    float _S73 = (F32_sqrt((-2.0f * clamp_0((F32_log((_S71.value))), -8.388607e+06f, 0.0f))));
    return Random_x24init_0(_S71.state, _S73 * (F32_cos((_S72))));
}

struct RandomSphere
{
    uint state;
    float3  value;
};

static __device__ RandomSphere RandomSphere_x24init_0(uint state_4, float3  value_2)
{
    RandomSphere _S74;
    (&_S74)->state = state_4;
    (&_S74)->value = value_2;
    return _S74;
}

__device__ RandomSphere random_sphere(uint state_5)
{
    Random _S75 = normal_distribution(state_5);
    Random _S76 = normal_distribution(_S75.state);
    Random _S77 = normal_distribution(_S76.state);
    float3  _S78 = make_float3 (_S75.value, _S76.value, _S77.value);
    float3  _S79 = ((((_S78)) * make_float3(rnorm3df(((_S78)).x, ((_S78)).y, ((_S78)).z))));
    return RandomSphere_x24init_0(_S77.state, _S79);
}

struct RandomHemisphere
{
    uint state;
    float3  value;
};

static __device__ RandomHemisphere RandomHemisphere_x24init_0(uint state_6, float3  value_3)
{
    RandomHemisphere _S80;
    (&_S80)->state = state_6;
    (&_S80)->value = value_3;
    return _S80;
}

__device__ RandomHemisphere random_hemisphere(uint state_7, float3  normal_0)
{
    RandomSphere _S81 = random_sphere(state_7);
    bool _S82 = (dot_0(_S81.value, normal_0)) > 0.0f;
    if(_S82)
    {
        return RandomHemisphere_x24init_0(_S81.state, _S81.value);
    }
    else
    {
        return RandomHemisphere_x24init_0(_S81.state, - _S81.value);
    }
}

__device__ uint stream(uint seed_0, uint index_0)
{
    uint _S83 = hash(index_0);
    uint _S84 = hash(seed_0 ^ _S83);
    return _S84;
}

