#ifndef PARALLAX_PS_INCLUDED
#define PARALLAX_PS_INCLUDED

cbuffer FScb : register(b0)
{
    float3 lightDir;
    float3 camPos;
    float2 center;
    float2 HMres;
    float2 HMres_r;
    float  scale;
    float  heightMapHeight;
    float  relax;
    uint   steps;
    uint   refine_steps;
    float  oneOverSteps;
    float  lightIntensity;
    bool   discardFragments;
    bool   displayNonConverged;
    int    const_isolate;
};

Texture2D gTexture;
Texture2D gAlbedoTexture;
SamplerState gSampler;

struct FsIn
{
    float4 posH : SV_POSITION;
    float2 texC : TEXCOORD;
    float3 posW : POSITION;
    float3 norm : NORMAL;
};


float getH_texture(float2 uv)
{
    return gTexture.Sample(gSampler, uv).r;
}


float getH(float2 uv)
{
#ifndef HEIGHT_SOURCE
    errorf("HEIGHT_SOURCE is not defined");
    return 1;
#elif HEIGHT_SOURCE == 0
    return getH_texture(uv);
#elif HEIGHT_SOURCE == 1
    return getH_Hermite(uv);
#else
    errorf("HEIGHT_SOURCE has an unused value: %w", HEIGHT_SOURCE);
    return 1;
#endif
}
float2 getHC_texture(float2 uv)
{
    return gTexture.Sample(gSampler, uv).rg;
}
#ifndef NORMAL_CHANNELS
#define NORMAL_CHANNELS ba
#endif
float2 getN_texture(float2 uv)
{
    return gTexture.Sample(gSampler, uv).NORMAL_CHANNELS;
}

float3 getNormalTBN_finiteDiff(float2 uv)
{
    const float mutliplier = 1;
    float2 delta = HMres_r * mutliplier;
    float2 one_over_delta = HMres / mutliplier;
    
    float2 du = float2(delta.x, 0);
    float2 dv = float2(0, delta.y);
    float dhdu = 0.5 * one_over_delta.x * (getH(uv + du) - getH(uv - du));
    float dhdv = 0.5 * one_over_delta.y * (getH(uv + dv) - getH(uv - dv));
    return normalize(float3(-dhdu, -dhdv, 1));
}
float3 decodeNormal(float2 xy)
{
    float2 nxy = 2 * xy - 1;
    return float3(nxy, sqrt(1 - dot(nxy, nxy)));

}
float3 getNormalTBN_texture(float2 uv)
{
    return decodeNormal(getN_texture(uv));

}

float3 getNormalTBN(float2 uv)
{
#ifndef NORMAL_SOURCE
    errorf("NORMAL_SOURCE is not defined");
    return float3(0, 0, 1);
#elif NORMAL_SOURCE == 0
    return getNormalTBN_finiteDiff(uv);
#elif NORMAL_SOURCE == 1
    return getNormalTBN_texture(uv);
#elif NORMAL_SOURCE == 2
    return getNormalTBN_Hermite(uv);
#else
    errorf("NORMAL_SOURCE has an unused value: %w", NORMAL_SOURCE);
    return float3(0, 0, 1);
#endif
}

float2 getDiffFromNormalTBN(float3 normal)
{
    return -normal.xy / normal.z;
}

#include "Hermite.slang"

struct BilinearWeights
{
    float2 weights;
    float2 center_uv;
    int2 texel_index;
};
BilinearWeights getBilinearWeights(float2 uv, float2 texSize, float2 oneOverTexSize)
{
    BilinearWeights ret;
    float2 texelCoordF = uv * texSize - 0.5;
    float2 topLeft = floor(texelCoordF);
    ret.weights = texelCoordF - topLeft;
    ret.center_uv = (topLeft + 1.0) * oneOverTexSize;
    ret.texel_index = (int2) topLeft;
    return ret;
}
float getH_Hermite(float2 uv)
{
    BilinearWeights ww = getBilinearWeights(uv, HMres, HMres_r);
    float2 r = ww.weights;
    float2 c = ww.center_uv;
    float4 t00, t01, t11, t10;

#if HERMITE_GATHER_USE_CONST_ISOLATE
    if (const_isolate)
    {
        float4 gR = gTexture.GatherRed(gSampler, c, int2(0, 0));
        t00.x = gR.w;
        t01.x = gR.x;
        t11.x = gR.y;
        t10.x = gR.z;
    }
    if (const_isolate)
    {
        float4 gG = gTexture.GatherGreen(gSampler, c, int2(0, 0));
        t00.y = gG.w;
        t01.y = gG.x;
        t11.y = gG.y;
        t10.y = gG.z;
    }
    if (const_isolate)
    {
        float4 gB = gTexture.GatherBlue(gSampler, c, int2(0, 0));
        t00.z = gB.w;
        t01.z = gB.x;
        t11.z = gB.y;
        t10.z = gB.z;
    }
    if (const_isolate)
    {
        float4 gA = gTexture.GatherAlpha(gSampler, c, int2(0, 0));
        t00.w = gA.w;
        t01.w = gA.x;
        t11.w = gA.y;
        t10.w = gA.z;
    }
#else
    float4 gR = gTexture.GatherRed(gSampler, c, int2(0, 0));
    float4 gG = gTexture.GatherGreen(gSampler, c, int2(0, 0));
    float4 gB = gTexture.GatherBlue(gSampler, c, int2(0, 0));
    float4 gA = gTexture.GatherAlpha(gSampler, c, int2(0, 0));
    t00 = float4(gR.w, gG.w, gB.w, gA.w);
    t01 = float4(gR.x, gG.x, gB.x, gA.x);
    t11 = float4(gR.y, gG.y, gB.y, gA.y);
    t10 = float4(gR.z, gG.z, gB.z, gA.z);
#endif
    
    t00.yz = getDiffFromNormalTBN(decodeNormal(t00.NORMAL_CHANNELS));
    t01.yz = getDiffFromNormalTBN(decodeNormal(t01.NORMAL_CHANNELS));
    t11.yz = getDiffFromNormalTBN(decodeNormal(t11.NORMAL_CHANNELS));
    t10.yz = getDiffFromNormalTBN(decodeNormal(t10.NORMAL_CHANNELS));
    
    return hermite3_2d(r, HMres_r, t01.xyz, t11.xyz, t10.xyz, t00.xyz);
}
float3 getNormalTBN_Hermite(float2 uv)
{
    BilinearWeights ww = getBilinearWeights(uv, HMres, HMres_r);
    float2 r = ww.weights;
    float2 c = ww.center_uv;
    float4 t00, t01, t11, t10;

#if HERMITE_GATHER_USE_CONST_ISOLATE
    if (const_isolate)
    {
        float4 gRaw = gTexture.GatherRed(gSampler, c, int2(0, 0));
        t00.x = gRaw.w;
        t01.x = gRaw.x;
        t11.x = gRaw.y;
        t10.x = gRaw.z;
    }
    if (const_isolate)
    {
        float4 gRaw = gTexture.GatherGreen(gSampler, c, int2(0, 0));
        t00.g = gRaw.w;
        t01.g = gRaw.x;
        t11.g = gRaw.y;
        t10.g = gRaw.z;
    }
    if (const_isolate)
    {
        float4 gRaw = gTexture.GatherBlue(gSampler, c, int2(0, 0));
        t00.b = gRaw.w;
        t01.b = gRaw.x;
        t11.b = gRaw.y;
        t10.b = gRaw.z;
    }
    if (const_isolate)
    {
        float4 gRaw = gTexture.GatherAlpha(gSampler, c, int2(0, 0));
        t00.a = gRaw.w;
        t01.a = gRaw.x;
        t11.a = gRaw.y;
        t10.a = gRaw.z;
    }    
#else
    float4 gR = gTexture.GatherRed(gSampler, c, int2(0, 0));
    float4 gG = gTexture.GatherGreen(gSampler, c, int2(0, 0));
    float4 gB = gTexture.GatherBlue(gSampler, c, int2(0, 0));
    float4 gA = gTexture.GatherAlpha(gSampler, c, int2(0, 0));
    
    t00 = float4(gR.w, gG.w, gB.w, gA.w);
    t01 = float4(gR.x, gG.x, gB.x, gA.x);
    t11 = float4(gR.y, gG.y, gB.y, gA.y);
    t10 = float4(gR.z, gG.z, gB.z, gA.z);
#endif
    
    t00.yz = getDiffFromNormalTBN(decodeNormal(t00.NORMAL_CHANNELS));
    t01.yz = getDiffFromNormalTBN(decodeNormal(t01.NORMAL_CHANNELS));
    t11.yz = getDiffFromNormalTBN(decodeNormal(t11.NORMAL_CHANNELS));
    t10.yz = getDiffFromNormalTBN(decodeNormal(t10.NORMAL_CHANNELS));
    
    float2 grad = hermite3_grad_2d(r, HMres_r, t01.xyz, t11.xyz, t10.xyz, t00.xyz);
    return normalize(float3(-grad * HMres * HMres, 1));
}

struct HMapIntersection
{
    float2 uv;
    float t;    // uv = (1-t)*u + t*u2
    float last_t;
    bool wasHit;
};

static const HMapIntersection INIT_INTERSECTION = { 0.0.xx, 0.0, 0.0, false };

#include "FindIntersection.slang"
#include "Refinement.slang"

// invert a 3x3 matrix
// input: the 3 columns of the matrix
float3x3 invMat(float3 a, float3 b, float3 c)
{
    float3 r1 = cross(b, c);
    float3 r2 = cross(c, a);
    float3 r3 = cross(a, b);
    float invDet = 1 / dot(r3, c);
    return invDet * float3x3(r1, r2, r3);
}

float4 main(FsIn fs) : SV_TARGET
{
    float2 u = fs.texC; // original texcoords
    float3 p = fs.posW; // fragment world pos
    float3 col = float3(.5);

    // calc tangent space
    float2 dxu = ddx_fine(u);
    float2 dyu = ddy_fine(u);
    float3 dxp = ddx_fine(p);
    float3 dyp = ddy_fine(p);
    
    float2x3 P = float2x3(dxp, dyp);
    float2x2 U = float2x2(dxu, dyu);
    float2x2 U_inv = float2x2(U[1][1], -U[0][1], -U[1][0], U[0][0]) / determinant(U);
    float2x3 A = mul(U_inv, P);
    float3 T = A[0];
    float3 B = A[1];
    float3 N = normalize(fs.norm) * heightMapHeight;

    float3x3 TBN = float3x3(T, B, N);
    float3x3 TBN_inv = transpose(invMat(T, B, N));

    float3 viewDirW = p - camPos;
    float3 viewDirT = mul(viewDirW, TBN_inv);

    // texcoords where the camera ray intersects the bottom plate
    float2 u2 = u - viewDirT.xy / viewDirT.z;

    // find the intersection with the height map
    HMapIntersection I = findIntersection(u, u2);
    float2 u3 = refineIntersection(I, u, u2);
    
    // intersection is outside of the bottom plate
    if (any(u3 < 0) || any(u3 > 1))
    {
        if (discardFragments)
            discard;
        return float4(1, 0, 0, 1);
    }
    
    // get normal in TBN space
    float3 normT = getNormalTBN(u3);
    // transform normal from TBN to world coordinates
    float3 norm = normalize(mul(normT, transpose(TBN_inv)));
    

    // fetch the final albedo color
#if defined( USE_ALBEDO_TEXTURE ) && USE_ALBEDO_TEXTURE
    float3 albedo = gAlbedoTexture.Sample(gSampler, u3).rgb;
    col *= albedo;
#endif
    

    
    float diffuse = lightIntensity * saturate(dot(-normalize(lightDir), norm));
    col.rgb *= diffuse;

    if (displayNonConverged && !I.wasHit)
    {
        col.rgb = float3(1,0,1);
    }
    
    return float4(col, 1);
}
#endif
