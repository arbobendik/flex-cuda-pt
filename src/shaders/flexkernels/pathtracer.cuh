#pragma once
#define Matrix SlangMatrix
#include "slang-cuda-prelude.h"
#undef Matrix

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

static __device__ float clamp_0(float x_0, float minBound_0, float maxBound_0)
{
    return (F32_min(((F32_max((x_0), (minBound_0)))), (maxBound_0)));
}

static __device__ float dot_0(float3  x_1, float3  y_0)
{
    return x_1.x * y_0.x + x_1.y * y_0.y + x_1.z * y_0.z;
}

__device__ float rgb_to_greyscale(float3  rgb_0)
{
    return dot_0(rgb_0, make_float3 (0.29899999499320984f, 0.58700001239776611f, 0.11400000005960464f));
}

__device__ uint hash(uint x_2)
{
    uint _S7 = x_2 * 747796405U + 2891336453U;
    uint _S8 = ((_S7 >> ((_S7 >> 28U) + 4U)) ^ _S7) * 277803737U;
    return (_S8 >> 22U) ^ _S8;
}

struct Random
{
    uint state;
    float value;
};

static __device__ Random Random_x24init_0(uint state_0, float value_0)
{
    Random _S9;
    (&_S9)->state = state_0;
    (&_S9)->value = value_0;
    return _S9;
}

static __device__ float random_0(uint * state_1)
{
    uint _S10 = hash(*state_1);
    *state_1 = _S10;
    return float((*state_1) >> int(8)) * 5.9604644775390625e-08f;
}

__device__ Random pcg(uint state_2)
{
    uint _S11 = state_2;
    float value_1 = random_0(&_S11);
    return Random_x24init_0(_S11, value_1);
}

__device__ Random normal_distribution(uint state_3)
{
    Random _S12 = pcg(state_3);
    Random _S13 = pcg(_S12.state);
    float _S14 = 6.28318548202514648f * _S12.value;
    float _S15 = (F32_sqrt((-2.0f * clamp_0((F32_log((_S13.value))), -8.388607e+06f, 0.0f))));
    return Random_x24init_0(_S13.state, _S15 * (F32_cos((_S14))));
}

struct RandomSphere
{
    uint state;
    float3  value;
};

static __device__ RandomSphere RandomSphere_x24init_0(uint state_4, float3  value_2)
{
    RandomSphere _S16;
    (&_S16)->state = state_4;
    (&_S16)->value = value_2;
    return _S16;
}

__device__ RandomSphere random_sphere(uint state_5)
{
    Random _S17 = normal_distribution(state_5);
    Random _S18 = normal_distribution(_S17.state);
    Random _S19 = normal_distribution(_S18.state);
    float3  _S20 = make_float3 (_S17.value, _S18.value, _S19.value);
    float3  _S21 = ((((_S20)) * make_float3(rnorm3df(((_S20)).x, ((_S20)).y, ((_S20)).z))));
    return RandomSphere_x24init_0(_S19.state, _S21);
}

struct RandomHemisphere
{
    uint state;
    float3  value;
};

static __device__ RandomHemisphere RandomHemisphere_x24init_0(uint state_6, float3  value_3)
{
    RandomHemisphere _S22;
    (&_S22)->state = state_6;
    (&_S22)->value = value_3;
    return _S22;
}

__device__ RandomHemisphere random_hemisphere(uint state_7, float3  normal_0)
{
    RandomSphere _S23 = random_sphere(state_7);
    bool _S24 = (dot_0(_S23.value, normal_0)) > 0.0f;
    if(_S24)
    {
        return RandomHemisphere_x24init_0(_S23.state, _S23.value);
    }
    else
    {
        return RandomHemisphere_x24init_0(_S23.state, - _S23.value);
    }
}

__device__ uint stream(uint seed_0, uint index_0)
{
    uint _S25 = hash(index_0);
    uint _S26 = hash(seed_0 ^ _S25);
    return _S26;
}

__device__ float trowbridgeReitz(float alpha_0, float n_dot_h_0)
{
    float _S27 = alpha_0 * alpha_0;
    float _S28 = n_dot_h_0 * n_dot_h_0 * (_S27 - 1.0f) + 1.0f;
    return _S27 / (F32_max((3.14159274101257324f * _S28 * _S28), (0.0000152587890625f)));
}

__device__ float smithG1(float alpha_1, float n_dot_x_0)
{
    float _S29 = alpha_1 * alpha_1;
    return 2.0f * n_dot_x_0 / (n_dot_x_0 + (F32_sqrt((_S29 + (1.0f - _S29) * n_dot_x_0 * n_dot_x_0))));
}

__device__ float fresnel(float cos_theta_0, float eta_i_0, float eta_o_0)
{
    float _S30 = (F32_sqrt(((F32_max((0.0f), (1.0f - cos_theta_0 * cos_theta_0))))));
    float _S31 = eta_i_0 / eta_o_0 * _S30;
    bool _S32 = _S31 < 1.0f;
    float kr_0;
    if(_S32)
    {
        float _S33 = (F32_sqrt(((F32_max((0.0f), (1.0f - _S31 * _S31))))));
        float _S34 = eta_o_0 * cos_theta_0;
        float _S35 = eta_i_0 * _S33;
        float _S36 = (_S34 - _S35) / (_S34 + _S35);
        float _S37 = eta_i_0 * cos_theta_0;
        float _S38 = eta_o_0 * _S33;
        float _S39 = (_S37 - _S38) / (_S37 + _S38);
        float _S40 = (_S36 * _S36 + _S39 * _S39) / 2.0f;
        kr_0 = _S40;
    }
    else
    {
        kr_0 = 1.0f;
    }
    return kr_0;
}

static __device__ float unshared_0(float x_3)
{
    return x_3;
}

__device__ float3  sampleGGXVNDF(float3  Ve_0, float alpha_2, float U1_0, float U2_0)
{
    float3  _S41 = make_float3 (alpha_2 * Ve_0.x, alpha_2 * Ve_0.y, Ve_0.z);
    float3  _S42 = ((((_S41)) * make_float3(rnorm3df(((_S41)).x, ((_S41)).y, ((_S41)).z))));
    float _S43 = _S42.x;
    float _S44 = _S42.y;
    float _S45 = _S43 * _S43 + _S44 * _S44;
    float3  _S46;
    if(_S45 > 0.0f)
    {
        float3  _S47 = make_float3 (- _S44, _S43, 0.0f);
        float _S48 = (rsqrtf((_S45)));
        _S46 = _S47 * make_float3 (_S48);
    }
    else
    {
        _S46 = make_float3 (1.0f, 0.0f, 0.0f);
    }
    float3  _S49 = cross_0(_S42, _S46);
    float _S50 = (F32_sqrt((U1_0)));
    float _S51 = 6.28318548202514648f * U2_0;
    float _S52 = _S50 * (F32_cos((_S51)));
    float t2_0 = _S50 * (F32_sin((_S51)));
    float _S53 = 0.5f * (1.0f + _S42.z);
    float t2_1 = (1.0f - _S53) * (F32_sqrt(((F32_max((0.0f), (1.0f - _S52 * _S52)))))) + _S53 * t2_0;
    float3  _S54 = make_float3 (_S52) * _S46 + make_float3 (t2_1) * _S49;
    float _S55 = unshared_0(_S52);
    float _S56 = unshared_0(_S52);
    float3  _S57 = _S54 + make_float3 ((F32_sqrt(((F32_max((0.0f), (1.0f - _S55 * _S56 - t2_1 * t2_1))))))) * _S42;
    float3  _S58 = make_float3 (alpha_2 * _S57.x, alpha_2 * _S57.y, (F32_max((0.0f), (_S57.z))));
    float3  _S59 = ((((_S58)) * make_float3(rnorm3df(((_S58)).x, ((_S58)).y, ((_S58)).z))));
    return _S59;
}

__device__ float3  sampleCosWeightedHemisphere(float random_1_0, float random_2_0)
{
    float _S60 = (F32_sqrt((random_1_0)));
    float _S61 = 6.28318548202514648f * random_2_0;
    float _S62 = _S60 * (F32_cos((_S61)));
    float _S63 = _S60 * (F32_sin((_S61)));
    float _S64 = (F32_sqrt(((F32_max((0.0f), (1.0f - _S62 * _S62 - _S63 * _S63))))));
    return make_float3 (_S62, _S64, _S63);
}

__device__ float3  tangentToWorld(float3  v_0, float3  n_0)
{
    float3  a_0 = make_float3 (0.0f, 1.0f, 0.0f);
    bool _S65 = (F32_abs((dot_0(n_0, a_0)))) > 0.9999847412109375f;
    float3  a_1;
    if(_S65)
    {
        float3  _S66 = make_float3 (1.0f, 0.0f, 0.0f);
        a_1 = _S66;
    }
    else
    {
        a_1 = a_0;
    }
    float3  _S67 = cross_0(n_0, a_1);
    float3  _S68 = ((((_S67)) * make_float3(rnorm3df(((_S67)).x, ((_S67)).y, ((_S67)).z))));
    float3  _S69 = cross_0(n_0, _S68);
    return make_float3 (v_0.x) * _S68 + make_float3 (v_0.y) * n_0 + make_float3 (v_0.z) * _S69;
}

__device__ float3  worldToTangent(float3  v_1, float3  n_1)
{
    float3  a_2 = make_float3 (0.0f, 1.0f, 0.0f);
    bool _S70 = (F32_abs((dot_0(n_1, a_2)))) > 0.9999847412109375f;
    float3  a_3;
    if(_S70)
    {
        float3  _S71 = make_float3 (1.0f, 0.0f, 0.0f);
        a_3 = _S71;
    }
    else
    {
        a_3 = a_2;
    }
    float3  _S72 = cross_0(n_1, a_3);
    float3  _S73 = ((((_S72)) * make_float3(rnorm3df(((_S72)).x, ((_S72)).y, ((_S72)).z))));
    float3  _S74 = cross_0(n_1, _S73);
    return make_float3 (dot_0(v_1, _S73), dot_0(v_1, n_1), dot_0(v_1, _S74));
}

__device__ float3  sampleHalfVector(float3  v_2, float3  n_i_0, float alpha_3, uint * random_state_0)
{
    Random _S75 = pcg(*random_state_0);
    Random _S76 = pcg(_S75.state);
    *random_state_0 = _S76.state;
    float3  _S77 = worldToTangent(v_2, n_i_0);
    float3  _S78 = sampleGGXVNDF(float3 {_S77.x, _S77.z, _S77.y}, alpha_3, _S75.value, _S76.value);
    float3  _S79 = float3 {_S78.x, _S78.z, _S78.y};
    float3  _S80 = tangentToWorld(_S79, n_i_0);
    return _S80;
}

__device__ float3  safeDivide(float3  a_4, float3  b_0)
{
    float3  _S81 = a_4 / b_0;
    float3  _S82 = make_float3 (0.0f);
    bool3  _S83 = b_0 == _S82;
    float3  _S84 = _slang_select(_S83, _S82,_S81);
    return _S84;
}

