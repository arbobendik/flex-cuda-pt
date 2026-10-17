#pragma once
#define Matrix SlangMatrix
#include "slang-cuda-prelude.h"
#undef Matrix

static __device__ float dot_0(float3  x_0, float3  y_0)
{
    return x_0.x * y_0.x + x_0.y * y_0.y + x_0.z * y_0.z;
}

static __device__ float3  reflect_0(float3  i_0, float3  n_0)
{
    return i_0 - make_float3 (2.0f * dot_0(n_0, i_0)) * n_0;
}

static __device__ float3  refract_0(float3  i_1, float3  n_1, float eta_0)
{
    float _S1 = dot_0(n_1, i_1);
    float _S2 = 1.0f - eta_0 * eta_0 * (1.0f - _S1 * _S1);
    if(_S2 < 0.0f)
    {
        return make_float3 (0.0f);
    }
    return make_float3 (eta_0) * i_1 - make_float3 (eta_0 * _S1 + (F32_sqrt((_S2)))) * n_1;
}

static __device__ float3  cross_0(float3  left_0, float3  right_0)
{
    float _S3 = left_0.y;
    float _S4 = right_0.z;
    float _S5 = left_0.z;
    float _S6 = right_0.y;
    float _S7 = right_0.x;
    float _S8 = left_0.x;
    return make_float3 (_S3 * _S4 - _S5 * _S6, _S5 * _S7 - _S8 * _S4, _S8 * _S6 - _S3 * _S7);
}

static __device__ float clamp_0(float x_1, float minBound_0, float maxBound_0)
{
    return (F32_min(((F32_max((x_1), (minBound_0)))), (maxBound_0)));
}

__device__ float rgb_to_greyscale(float3  rgb_0)
{
    return dot_0(rgb_0, make_float3 (0.29899999499320984f, 0.58700001239776611f, 0.11400000005960464f));
}

__device__ uint hash(uint x_2)
{
    uint _S9 = x_2 * 747796405U + 2891336453U;
    uint _S10 = ((_S9 >> ((_S9 >> 28U) + 4U)) ^ _S9) * 277803737U;
    return (_S10 >> 22U) ^ _S10;
}

struct Random
{
    uint state;
    float value;
};

static __device__ Random Random_x24init_0(uint state_0, float value_0)
{
    Random _S11;
    (&_S11)->state = state_0;
    (&_S11)->value = value_0;
    return _S11;
}

static __device__ float random_0(uint * state_1)
{
    uint _S12 = hash(*state_1);
    *state_1 = _S12;
    return float((*state_1) >> int(8)) * 5.9604644775390625e-08f;
}

__device__ Random pcg(uint state_2)
{
    uint _S13 = state_2;
    float value_1 = random_0(&_S13);
    return Random_x24init_0(_S13, value_1);
}

__device__ Random normal_distribution(uint state_3)
{
    Random _S14 = pcg(state_3);
    Random _S15 = pcg(_S14.state);
    float _S16 = 6.28318548202514648f * _S14.value;
    float _S17 = (F32_sqrt((-2.0f * clamp_0((F32_log((_S15.value))), -8.388607e+06f, 0.0f))));
    return Random_x24init_0(_S15.state, _S17 * (F32_cos((_S16))));
}

struct RandomSphere
{
    uint state;
    float3  value;
};

static __device__ RandomSphere RandomSphere_x24init_0(uint state_4, float3  value_2)
{
    RandomSphere _S18;
    (&_S18)->state = state_4;
    (&_S18)->value = value_2;
    return _S18;
}

__device__ RandomSphere random_sphere(uint state_5)
{
    Random _S19 = normal_distribution(state_5);
    Random _S20 = normal_distribution(_S19.state);
    Random _S21 = normal_distribution(_S20.state);
    float3  _S22 = make_float3 (_S19.value, _S20.value, _S21.value);
    float3  _S23 = ((((_S22)) * make_float3(rnorm3df(((_S22)).x, ((_S22)).y, ((_S22)).z))));
    return RandomSphere_x24init_0(_S21.state, _S23);
}

struct RandomHemisphere
{
    uint state;
    float3  value;
};

static __device__ RandomHemisphere RandomHemisphere_x24init_0(uint state_6, float3  value_3)
{
    RandomHemisphere _S24;
    (&_S24)->state = state_6;
    (&_S24)->value = value_3;
    return _S24;
}

__device__ RandomHemisphere random_hemisphere(uint state_7, float3  normal_0)
{
    RandomSphere _S25 = random_sphere(state_7);
    bool _S26 = (dot_0(_S25.value, normal_0)) > 0.0f;
    if(_S26)
    {
        return RandomHemisphere_x24init_0(_S25.state, _S25.value);
    }
    else
    {
        return RandomHemisphere_x24init_0(_S25.state, - _S25.value);
    }
}

__device__ uint stream(uint seed_0, uint index_0)
{
    uint _S27 = hash(index_0);
    uint _S28 = hash(seed_0 ^ _S27);
    return _S28;
}

__device__ float trowbridgeReitz(float alpha_0, float n_dot_h_0)
{
    float _S29 = alpha_0 * alpha_0;
    float _S30 = n_dot_h_0 * n_dot_h_0 * (_S29 - 1.0f) + 1.0f;
    return _S29 / (F32_max((3.14159274101257324f * _S30 * _S30), (0.0000152587890625f)));
}

__device__ float smithG1(float alpha_1, float n_dot_x_0)
{
    float _S31 = alpha_1 * alpha_1;
    return 2.0f * n_dot_x_0 / (n_dot_x_0 + (F32_sqrt((_S31 + (1.0f - _S31) * n_dot_x_0 * n_dot_x_0))));
}

__device__ float fresnel(float cos_theta_0, float eta_i_0, float eta_o_0)
{
    float _S32 = (F32_sqrt(((F32_max((0.0f), (1.0f - cos_theta_0 * cos_theta_0))))));
    float _S33 = eta_i_0 / eta_o_0 * _S32;
    bool _S34 = _S33 < 1.0f;
    float kr_0;
    if(_S34)
    {
        float _S35 = (F32_sqrt(((F32_max((0.0f), (1.0f - _S33 * _S33))))));
        float _S36 = eta_o_0 * cos_theta_0;
        float _S37 = eta_i_0 * _S35;
        float _S38 = (_S36 - _S37) / (_S36 + _S37);
        float _S39 = eta_i_0 * cos_theta_0;
        float _S40 = eta_o_0 * _S35;
        float _S41 = (_S39 - _S40) / (_S39 + _S40);
        float _S42 = (_S38 * _S38 + _S41 * _S41) / 2.0f;
        kr_0 = _S42;
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
    float3  _S43 = make_float3 (alpha_2 * Ve_0.x, alpha_2 * Ve_0.y, Ve_0.z);
    float3  _S44 = ((((_S43)) * make_float3(rnorm3df(((_S43)).x, ((_S43)).y, ((_S43)).z))));
    float _S45 = _S44.x;
    float _S46 = _S44.y;
    float _S47 = _S45 * _S45 + _S46 * _S46;
    float3  _S48;
    if(_S47 > 0.0f)
    {
        float3  _S49 = make_float3 (- _S46, _S45, 0.0f);
        float _S50 = (rsqrtf((_S47)));
        _S48 = _S49 * make_float3 (_S50);
    }
    else
    {
        _S48 = make_float3 (1.0f, 0.0f, 0.0f);
    }
    float3  _S51 = cross_0(_S44, _S48);
    float _S52 = (F32_sqrt((U1_0)));
    float _S53 = 6.28318548202514648f * U2_0;
    float _S54 = _S52 * (F32_cos((_S53)));
    float t2_0 = _S52 * (F32_sin((_S53)));
    float _S55 = 0.5f * (1.0f + _S44.z);
    float t2_1 = (1.0f - _S55) * (F32_sqrt(((F32_max((0.0f), (1.0f - _S54 * _S54)))))) + _S55 * t2_0;
    float3  _S56 = make_float3 (_S54) * _S48 + make_float3 (t2_1) * _S51;
    float _S57 = unshared_0(_S54);
    float _S58 = unshared_0(_S54);
    float3  _S59 = _S56 + make_float3 ((F32_sqrt(((F32_max((0.0f), (1.0f - _S57 * _S58 - t2_1 * t2_1))))))) * _S44;
    float3  _S60 = make_float3 (alpha_2 * _S59.x, alpha_2 * _S59.y, (F32_max((0.0f), (_S59.z))));
    float3  _S61 = ((((_S60)) * make_float3(rnorm3df(((_S60)).x, ((_S60)).y, ((_S60)).z))));
    return _S61;
}

__device__ float3  sampleCosWeightedHemisphere(float random_1_0, float random_2_0)
{
    float _S62 = (F32_sqrt((random_1_0)));
    float _S63 = 6.28318548202514648f * random_2_0;
    float _S64 = _S62 * (F32_cos((_S63)));
    float _S65 = _S62 * (F32_sin((_S63)));
    float _S66 = (F32_sqrt(((F32_max((0.0f), (1.0f - _S64 * _S64 - _S65 * _S65))))));
    return make_float3 (_S64, _S66, _S65);
}

__device__ float3  tangentToWorld(float3  v_0, float3  n_2)
{
    float3  a_0 = make_float3 (0.0f, 1.0f, 0.0f);
    bool _S67 = (F32_abs((dot_0(n_2, a_0)))) > 0.9999847412109375f;
    float3  a_1;
    if(_S67)
    {
        float3  _S68 = make_float3 (1.0f, 0.0f, 0.0f);
        a_1 = _S68;
    }
    else
    {
        a_1 = a_0;
    }
    float3  _S69 = cross_0(n_2, a_1);
    float3  _S70 = ((((_S69)) * make_float3(rnorm3df(((_S69)).x, ((_S69)).y, ((_S69)).z))));
    float3  _S71 = cross_0(n_2, _S70);
    return make_float3 (v_0.x) * _S70 + make_float3 (v_0.y) * n_2 + make_float3 (v_0.z) * _S71;
}

__device__ float3  worldToTangent(float3  v_1, float3  n_3)
{
    float3  a_2 = make_float3 (0.0f, 1.0f, 0.0f);
    bool _S72 = (F32_abs((dot_0(n_3, a_2)))) > 0.9999847412109375f;
    float3  a_3;
    if(_S72)
    {
        float3  _S73 = make_float3 (1.0f, 0.0f, 0.0f);
        a_3 = _S73;
    }
    else
    {
        a_3 = a_2;
    }
    float3  _S74 = cross_0(n_3, a_3);
    float3  _S75 = ((((_S74)) * make_float3(rnorm3df(((_S74)).x, ((_S74)).y, ((_S74)).z))));
    float3  _S76 = cross_0(n_3, _S75);
    return make_float3 (dot_0(v_1, _S75), dot_0(v_1, n_3), dot_0(v_1, _S76));
}

__device__ float3  sampleHalfVector(float3  v_2, float3  n_i_0, float alpha_3, uint * random_state_0)
{
    Random _S77 = pcg(*random_state_0);
    Random _S78 = pcg(_S77.state);
    *random_state_0 = _S78.state;
    float3  _S79 = worldToTangent(v_2, n_i_0);
    float3  _S80 = sampleGGXVNDF(float3 {_S79.x, _S79.z, _S79.y}, alpha_3, _S77.value, _S78.value);
    float3  _S81 = float3 {_S80.x, _S80.z, _S80.y};
    float3  _S82 = tangentToWorld(_S81, n_i_0);
    return _S82;
}

__device__ float3  safeDivide(float3  a_4, float3  b_0)
{
    float3  _S83 = a_4 / b_0;
    float3  _S84 = make_float3 (0.0f);
    bool3  _S85 = b_0 == _S84;
    float3  _S86 = _slang_select(_S85, _S84,_S83);
    return _S86;
}

__device__ float3  tinted(Surface material_0, float cos_theta_1, float eta_i_1, float eta_o_1)
{
    float root_0 = (eta_i_1 - eta_o_1) / (eta_i_1 + eta_o_1) * (F32_sqrt((material_0.level)));
    float F_0 = fresnel(cos_theta_1, 1.0f + root_0, 1.0f - root_0);
    float F0_0 = root_0 * root_0;
    return make_float3 (F_0, F_0, F_0) - (make_float3 (1.0f, 1.0f, 1.0f) - material_0.tint) * make_float3 (F0_0 * (1.0f - F_0) / (F32_max((1.0f - F0_0), (0.0000152587890625f))));
}

static __device__ float3  mix_0(float3  a_5, float3  b_1, float t_0)
{
    return a_5 * make_float3 (1.0f - t_0) + b_1 * make_float3 (t_0);
}

static __device__ float sign_0(float x_4)
{
    return float(x_4 > 0.0f) - float(x_4 < 0.0f);
}

struct BSDFLobes
{
    float v_dot_h;
    float n_dot_h;
    float3  refracted;
    float3  F;
    float F_greyscale;
    float reflect_weight;
    float diffuse_weight;
    float refract_weight;
    float p_diffuse;
    float p_reflect;
    float p_refract;
};

static __device__ BSDFLobes BSDFLobes_x24init_0(float v_dot_h_0, float n_dot_h_1, float3  refracted_0, float3  F_1, float F_greyscale_0, float reflect_weight_0, float diffuse_weight_0, float refract_weight_0, float p_diffuse_0, float p_reflect_0, float p_refract_0)
{
    BSDFLobes _S87;
    (&_S87)->v_dot_h = v_dot_h_0;
    (&_S87)->n_dot_h = n_dot_h_1;
    (&_S87)->refracted = refracted_0;
    (&_S87)->F = F_1;
    (&_S87)->F_greyscale = F_greyscale_0;
    (&_S87)->reflect_weight = reflect_weight_0;
    (&_S87)->diffuse_weight = diffuse_weight_0;
    (&_S87)->refract_weight = refract_weight_0;
    (&_S87)->p_diffuse = p_diffuse_0;
    (&_S87)->p_reflect = p_reflect_0;
    (&_S87)->p_refract = p_refract_0;
    return _S87;
}

__device__ BSDFLobes bsdfLobes(float3  in_dir_0, float3  v_3, float3  n_i_1, float3  ggx_n_0, Surface material_1, float eta_i_2, float eta_o_2)
{
    float v_dot_h_1 = dot_0(ggx_n_0, v_3);
    float n_dot_h_2 = dot_0(n_i_1, ggx_n_0);
    float eta_1 = eta_i_2 / eta_o_2;
    float3  _S88 = refract_0(in_dir_0, ggx_n_0, eta_1);
    float3  refracted_1 = ((((_S88)) * make_float3(rnorm3df(((_S88)).x, ((_S88)).y, ((_S88)).z))));
    float3  _S89 = tinted(material_1, (F32_abs((v_dot_h_1))), eta_i_2, eta_o_2);
    float3  F_2 = mix_0(_S89, material_1.albedo, material_1.metallic);
    float F_greyscale_1 = rgb_to_greyscale(F_2);
    float diffuse_weight_1 = (1.0f - material_1.transparency) * (1.0f - material_1.metallic);
    float _S90 = (F32_max((F_greyscale_1), (0.0f)));
    float _S91 = 1.0f - F_greyscale_1;
    float _S92 = (F32_max((diffuse_weight_1 * _S91), (0.0f)));
    float _S93 = material_1.transparency * _S91;
    float _S94 = sign_0((F32_sqrt((dot_0(refracted_1, refracted_1)))));
    float _S95 = (F32_max((_S93 * _S94), (0.0f)));
    float total_component_0 = _S90 + _S92 + _S95;
    float total_component_inv_0 = 1.0f / (F32_max((total_component_0), (0.0000152587890625f)));
    float p_diffuse_1 = _S92 * total_component_inv_0;
    float p_reflect_1 = _S90 * total_component_inv_0;
    float p_refract_1 = _S95 * total_component_inv_0;
    return BSDFLobes_x24init_0(v_dot_h_1, n_dot_h_2, refracted_1, F_2, F_greyscale_1, 1.0f, diffuse_weight_1, material_1.transparency, p_diffuse_1, p_reflect_1, p_refract_1);
}

static __device__ float brightest_0(float3  rgb_1)
{
    return (F32_max((rgb_1.x), ((F32_max((rgb_1.y), (rgb_1.z))))));
}

static __device__ float G1_0(float alpha_4, float n_dot_x_1)
{
    float _S96 = smithG1(alpha_4, n_dot_x_1);
    return _S96;
}

__device__ float3  lobeThroughput(float3  v_4, float3  l_0, float3  n_i_2, float alpha_5, Surface material_2, BSDFLobes lobes_0, float eta_i_3, float eta_o_3, uint lobe_0)
{
    bool _S97 = lobe_0 == 0U;
    if(_S97)
    {
        float3  _S98 = make_float3 (lobes_0.diffuse_weight) * material_2.albedo / make_float3 (lobes_0.p_diffuse);
        float3  _S99 = v_4 + l_0;
        float3  _S100 = ((((_S99)) * make_float3(rnorm3df(((_S99)).x, ((_S99)).y, ((_S99)).z))));
        float3  _S101 = tinted(material_2, (F32_abs((dot_0(v_4, _S100)))), eta_i_3, eta_o_3);
        float3  _S102 = mix_0(_S101, material_2.albedo, material_2.metallic);
        float _S103 = brightest_0(_S102);
        return _S98 * make_float3 (1.0f - _S103);
    }
    bool _S104 = lobe_0 == 1U;
    if(_S104)
    {
        float m_n_i_dot_l_0 = - dot_0(n_i_2, l_0);
        float G1_l_0 = G1_0(alpha_5, (F32_max((m_n_i_dot_l_0), (0.0f))));
        return material_2.albedo * make_float3 (lobes_0.refract_weight * G1_l_0 * (1.0f - lobes_0.F_greyscale) / lobes_0.p_refract);
    }
    float n_i_dot_l_0 = dot_0(n_i_2, l_0);
    float G1_l_1 = G1_0(alpha_5, (F32_max((n_i_dot_l_0), (0.0f))));
    return make_float3 (lobes_0.reflect_weight * G1_l_1) * lobes_0.F / make_float3 (lobes_0.p_reflect);
}

__device__ float lobeDensity(float n_i_dot_l_1, float l_dot_h_0, float n_i_dot_v_0, float alpha_6, BSDFLobes lobes_1, float eta_i_4, float eta_o_4, uint lobe_1)
{
    bool _S105 = lobe_1 == 0U;
    float pdf_h_0;
    if(_S105)
    {
        if(n_i_dot_v_0 > 0.0f)
        {
            pdf_h_0 = lobes_1.p_diffuse * n_i_dot_l_1 * 0.31830987334251404f;
        }
        else
        {
            pdf_h_0 = 0.0f;
        }
        return pdf_h_0;
    }
    float _S106 = lobes_1.n_dot_h;
    float d_term_0 = _S106 * _S106 * (alpha_6 * alpha_6 - 1.0f) + 1.0f;
    bool _S107 = alpha_6 > 0.0f;
    bool _S108;
    if(_S107)
    {
        _S108 = (lobes_1.v_dot_h) > 0.0f;
    }
    else
    {
        _S108 = false;
    }
    if(_S108)
    {
        _S108 = (lobes_1.n_dot_h) > 0.0f;
    }
    else
    {
        _S108 = false;
    }
    if(_S108)
    {
        _S108 = n_i_dot_v_0 > 0.0f;
    }
    else
    {
        _S108 = false;
    }
    if(_S108)
    {
        float _S109 = smithG1(alpha_6, n_i_dot_v_0);
        pdf_h_0 = _S109 * lobes_1.v_dot_h * alpha_6 * alpha_6 / (3.14159274101257324f * d_term_0 * d_term_0 * n_i_dot_v_0);
    }
    else
    {
        pdf_h_0 = 0.0f;
    }
    bool _S110 = lobe_1 == 1U;
    if(_S110)
    {
        float denom_0 = eta_i_4 * lobes_1.v_dot_h + eta_o_4 * l_dot_h_0;
        bool _S111 = eta_i_4 != eta_o_4;
        if(_S111)
        {
            _S108 = l_dot_h_0 < 0.0f;
        }
        else
        {
            _S108 = false;
        }
        if(_S108)
        {
            pdf_h_0 = lobes_1.p_refract * pdf_h_0 * eta_o_4 * eta_o_4 * (F32_abs((l_dot_h_0))) / (denom_0 * denom_0);
        }
        else
        {
            pdf_h_0 = 0.0f;
        }
        return pdf_h_0;
    }
    if(pdf_h_0 > 0.0f)
    {
        pdf_h_0 = lobes_1.p_reflect * pdf_h_0 / (4.0f * lobes_1.v_dot_h);
    }
    else
    {
        pdf_h_0 = 0.0f;
    }
    return pdf_h_0;
}

struct SampleBSDF
{
    float3  unit_direction;
    float3  throughput;
    uint random_state;
    bool refracted;
    uint lobe;
    float pdf;
};

static __device__ SampleBSDF SampleBSDF_x24init_0(float3  unit_direction_0, float3  throughput_0, uint random_state_1, bool refracted_2, uint lobe_2, float pdf_0)
{
    SampleBSDF _S112;
    (&_S112)->unit_direction = unit_direction_0;
    (&_S112)->throughput = throughput_0;
    (&_S112)->random_state = random_state_1;
    (&_S112)->refracted = refracted_2;
    (&_S112)->lobe = lobe_2;
    (&_S112)->pdf = pdf_0;
    return _S112;
}

__device__ SampleBSDF sampleBSDF(float3  in_dir_1, float3  n_4, Surface material_3, float eta_i_5, float eta_o_5, uint random_init_0, float2  screen_space_0)
{
    uint random_state_2 = random_init_0;
    float3  v_5 = - in_dir_1;
    float n_dot_v_0 = dot_0(n_4, v_5);
    float _S113 = sign_0(n_dot_v_0);
    float3  n_i_3 = n_4 * make_float3 (_S113);
    float n_i_dot_v_1 = (F32_abs((n_dot_v_0)));
    float _S114 = material_3.roughness;
    float alpha_7 = _S114 * _S114;
    float3  ggx_n_1 = sampleHalfVector(v_5, n_i_3, alpha_7, &random_state_2);
    BSDFLobes lobes_2 = bsdfLobes(in_dir_1, v_5, n_i_3, ggx_n_1, material_3, eta_i_5, eta_o_5);
    float3  _S115 = make_float3 (1.0f, 1.0f, 1.0f);
    SampleBSDF _S116 = SampleBSDF_x24init_0(_S115, _S115, 0U, false, 0U, 0.0f);
    SampleBSDF sample_0 = _S116;
    Random random_p_0 = pcg(random_state_2);
    random_state_2 = random_p_0.state;
    bool _S117 = (random_p_0.value) < (lobes_2.p_diffuse);
    if(_S117)
    {
        Random random_d_1_0 = pcg(random_state_2);
        Random random_d_2_0 = pcg(random_d_1_0.state);
        (&sample_0)->random_state = random_d_2_0.state;
        float3  cosine_hemisphere_0 = sampleCosWeightedHemisphere(random_d_1_0.value, random_d_2_0.value);
        float3  _S118 = tangentToWorld(cosine_hemisphere_0, n_i_3);
        (&sample_0)->unit_direction = _S118;
        float3  _S119 = lobeThroughput(v_5, (&sample_0)->unit_direction, n_i_3, alpha_7, material_3, lobes_2, eta_i_5, eta_o_5, 0U);
        (&sample_0)->throughput = _S119;
        float _S120 = lobeDensity(cosine_hemisphere_0.y, 0.0f, n_i_dot_v_1, alpha_7, lobes_2, eta_i_5, eta_o_5, 0U);
        (&sample_0)->pdf = _S120;
        return sample_0;
    }
    bool _S121 = (random_p_0.value) < (lobes_2.p_diffuse + lobes_2.p_refract);
    if(_S121)
    {
        (&sample_0)->unit_direction = lobes_2.refracted;
        float _S122 = lobeDensity(0.0f, dot_0((&sample_0)->unit_direction, ggx_n_1), n_i_dot_v_1, alpha_7, lobes_2, eta_i_5, eta_o_5, 1U);
        (&sample_0)->pdf = _S122;
        float3  _S123 = lobeThroughput(v_5, (&sample_0)->unit_direction, n_i_3, alpha_7, material_3, lobes_2, eta_i_5, eta_o_5, 1U);
        (&sample_0)->throughput = _S123;
        (&sample_0)->random_state = random_state_2;
        (&sample_0)->refracted = true;
        (&sample_0)->lobe = 1U;
        return sample_0;
    }
    float3  _S124 = reflect_0(in_dir_1, ggx_n_1);
    float3  _S125 = ((((_S124)) * make_float3(rnorm3df(((_S124)).x, ((_S124)).y, ((_S124)).z))));
    (&sample_0)->unit_direction = _S125;
    float3  _S126 = lobeThroughput(v_5, (&sample_0)->unit_direction, n_i_3, alpha_7, material_3, lobes_2, eta_i_5, eta_o_5, 2U);
    (&sample_0)->throughput = _S126;
    (&sample_0)->random_state = random_state_2;
    (&sample_0)->lobe = 2U;
    float _S127 = lobeDensity(0.0f, 0.0f, n_i_dot_v_1, alpha_7, lobes_2, eta_i_5, eta_o_5, 2U);
    (&sample_0)->pdf = _S127;
    return sample_0;
}

__device__ SampleBSDF connectBSDF(float3  in_dir_2, float3  out_dir_0, float3  n_5, Surface material_4, float eta_i_6, float eta_o_6, uint lobe_3, uint random_init_1)
{
    uint random_state_3 = random_init_1;
    float3  v_6 = - in_dir_2;
    float n_dot_v_1 = dot_0(n_5, v_6);
    float _S128 = sign_0(n_dot_v_1);
    float3  n_i_4 = n_5 * make_float3 (_S128);
    float _S129 = material_4.roughness;
    float alpha_8 = _S129 * _S129;
    float3  _S130 = v_6 + out_dir_0;
    float3  _S131 = ((((_S130)) * make_float3(rnorm3df(((_S130)).x, ((_S130)).y, ((_S130)).z))));
    bool _S132 = lobe_3 == 0U;
    float3  ggx_n_2;
    if(_S132)
    {
        float3  _S133 = sampleHalfVector(v_6, n_i_4, alpha_8, &random_state_3);
        ggx_n_2 = _S133;
    }
    else
    {
        bool _S134 = lobe_3 == 1U;
        if(_S134)
        {
            float3  h_t_0 = - (make_float3 (eta_i_6) * v_6 + make_float3 (eta_o_6) * out_dir_0);
            float3  _S135 = ((((h_t_0)) * make_float3(rnorm3df(((h_t_0)).x, ((h_t_0)).y, ((h_t_0)).z))));
            float _S136 = sign_0(dot_0(h_t_0, n_i_4));
            float3  _S137 = _S135 * make_float3 (_S136);
            ggx_n_2 = _S137;
        }
        else
        {
            ggx_n_2 = _S131;
        }
    }
    BSDFLobes lobes_3 = bsdfLobes(in_dir_2, v_6, n_i_4, ggx_n_2, material_4, eta_i_6, eta_o_6);
    float pdf_1 = lobeDensity(dot_0(n_i_4, out_dir_0), dot_0(out_dir_0, ggx_n_2), (F32_abs((n_dot_v_1))), alpha_8, lobes_3, eta_i_6, eta_o_6, lobe_3);
    bool _S138 = !(pdf_1 > 0.0f);
    if(_S138)
    {
        float3  _S139 = make_float3 (0.0f);
        return SampleBSDF_x24init_0(_S139, _S139, 0U, false, 0U, 0.0f);
    }
    float3  _S140 = lobeThroughput(v_6, out_dir_0, n_i_4, alpha_8, material_4, lobes_3, eta_i_6, eta_o_6, lobe_3);
    return SampleBSDF_x24init_0(out_dir_0, _S140, random_state_3, lobe_3 == 1U, lobe_3, pdf_1);
}

__device__ float3  BSDF(float3  in_dir_3, float3  out_dir_1, float3  n_6, float3  g_n_0, Surface material_5, float eta_i_7, float eta_o_7, uint random_state_4)
{
    bool _S141 = (dot_0(g_n_0, - in_dir_3) * dot_0(g_n_0, out_dir_1)) > 0.0f;
    if(_S141)
    {
        SampleBSDF diffuse_0 = connectBSDF(in_dir_3, out_dir_1, n_6, material_5, eta_i_7, eta_o_7, 0U, random_state_4);
        SampleBSDF reflection_0 = connectBSDF(in_dir_3, out_dir_1, n_6, material_5, eta_i_7, eta_o_7, 2U, random_state_4);
        return diffuse_0.throughput * make_float3 (diffuse_0.pdf) + reflection_0.throughput * make_float3 (reflection_0.pdf);
    }
    SampleBSDF refraction_0 = connectBSDF(in_dir_3, out_dir_1, n_6, material_5, eta_i_7, eta_o_7, 1U, random_state_4);
    return refraction_0.throughput * make_float3 (refraction_0.pdf);
}

__device__ float3  beerLambert(float distance_0, Surface material_6)
{
    float depth_0 = distance_0 * material_6.density;
    return make_float3 ((F32_exp((depth_0 * (material_6.absorption.x - 1.0f)))), (F32_exp((depth_0 * (material_6.absorption.y - 1.0f)))), (F32_exp((depth_0 * (material_6.absorption.z - 1.0f)))));
}

__device__ bool neeDecision(Surface material_7, float n_dot_v_2, float eta_i_8, float eta_o_8)
{
    float3  _S142 = tinted(material_7, (F32_abs((n_dot_v_2))), eta_i_8, eta_o_8);
    float3  F_n_0 = mix_0(_S142, material_7.albedo, material_7.metallic);
    float F_n_greyscale_0 = rgb_to_greyscale(F_n_0);
    float _S143 = (F32_max(((1.0f - F_n_greyscale_0) * (1.0f - material_7.metallic) * (1.0f - material_7.transparency)), (0.0f)));
    bool _S144 = _S143 > 0.03999999910593033f;
    bool _S145;
    if(_S144)
    {
        _S145 = true;
    }
    else
    {
        _S145 = (material_7.roughness) > 0.20000000298023224f;
    }
    return _S145;
}

struct Matrix
{
    float3  x;
    float3  y;
    float3  z;
};

static __device__ Matrix Matrix_x24init_0(float3  x_5, float3  y_1, float3  z_0)
{
    Matrix _S146;
    (&_S146)->x = x_5;
    (&_S146)->y = y_1;
    (&_S146)->z = z_0;
    return _S146;
}

__device__ Matrix normalMapTBN(float3  p0_0, float3  p1_0, float3  p2_0, float2  uv0_0, float2  uv1_0, float2  uv2_0, float3  n_7)
{
    float3  e1_0 = p1_0 - p0_0;
    float3  e2_0 = p2_0 - p0_0;
    float2  duv1_0 = uv1_0 - uv0_0;
    float2  duv2_0 = uv2_0 - uv0_0;
    float _S147 = duv1_0.x;
    float _S148 = duv2_0.y;
    float _S149 = duv1_0.y;
    float _S150 = duv2_0.x;
    float det_0 = _S147 * _S148 - _S149 * _S150;
    float inv_det_0 = 1.0f / det_0;
    float3  t_1 = (e1_0 * make_float3 (_S148) - e2_0 * make_float3 (_S149)) * make_float3 (inv_det_0);
    float3  b_geom_0 = (e2_0 * make_float3 (_S147) - e1_0 * make_float3 (_S150)) * make_float3 (inv_det_0);
    float3  _S151 = t_1 - n_7 * make_float3 (dot_0(n_7, t_1));
    float3  t_2 = ((((_S151)) * make_float3(rnorm3df(((_S151)).x, ((_S151)).y, ((_S151)).z))));
    float3  _S152 = cross_0(n_7, t_2);
    float _S153 = sign_0(dot_0(_S152, b_geom_0));
    float3  b_2 = _S152 * make_float3 (_S153);
    return Matrix_x24init_0(t_2, b_2, n_7);
}

__device__ float3  tangentToWorldNormalMap(float3  v_7, Matrix tbn_0)
{
    return tbn_0.x * make_float3 (v_7.x) + tbn_0.y * make_float3 (v_7.y) + tbn_0.z * make_float3 (v_7.z);
}

struct Selection
{
    Reservoir r;
    float w_sum;
    float3  color;
    uint random_state;
};

__device__ void offer(Selection * sel_0, Reservoir candidate_0, float3  weight_0)
{
    float w_0 = rgb_to_greyscale(weight_0);
    bool _S154 = w_0 > 0.0f;
    bool _S155;
    if(_S154)
    {
        _S155 = w_0 < 3.00000000549775576e+38f;
    }
    else
    {
        _S155 = false;
    }
    bool _S156 = !_S155;
    if(_S156)
    {
        return;
    }
    float _S157 = sel_0->w_sum + w_0;
    sel_0->w_sum = _S157;
    float3  _S158 = sel_0->color + weight_0;
    sel_0->color = _S158;
    Random random_value_0 = pcg(sel_0->random_state);
    sel_0->random_state = random_value_0.state;
    bool _S159 = (random_value_0.value * sel_0->w_sum) <= w_0;
    if(_S159)
    {
        sel_0->r = candidate_0;
    }
    return;
}

__device__ void offerShifted(Selection * sel_1, Reservoir r_0, Shift shifted_0, float mis_0)
{
    Reservoir candidate_1 = r_0;
    float3  _S160 = shifted_0.f / make_float3 (shifted_0.jac);
    (&candidate_1)->f = _S160;
    float _S161 = r_0.jac * shifted_0.jac;
    (&candidate_1)->jac = _S161;
    offer(sel_1, candidate_1, make_float3 (mis_0) * shifted_0.f * make_float3 (r_0.w));
    return;
}

__device__ Reservoir finalize(Selection sel_2, float c_0)
{
    Reservoir r_1 = sel_2.r;
    float p_hat_0 = rgb_to_greyscale((&r_1)->f);
    float _S162;
    if(p_hat_0 > 0.0f)
    {
        _S162 = sel_2.w_sum / p_hat_0;
    }
    else
    {
        _S162 = 0.0f;
    }
    (&r_1)->w = _S162;
    (&r_1)->c = c_0;
    return r_1;
}

__device__ Reservoir withEnd(Reservoir r_2, uint end_type_0, uint end_0, float3  f_0)
{
    Reservoir candidate_2 = r_2;
    uint _S163 = ((r_2.path) | (end_0 << 0U)) | (end_type_0 << 12U);
    (&candidate_2)->path = _S163;
    (&candidate_2)->f = f_0;
    return candidate_2;
}

