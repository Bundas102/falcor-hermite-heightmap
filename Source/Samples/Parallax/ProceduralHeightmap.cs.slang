cbuffer CScb : register(b0)
{
    uint2 maxSize;
    int4 intParams;
    float4 floatParams;
};



#include "AutoDiff.slang"

float3 paramFun_sinc(float3 x, float3 y)
{
    // domain
    x = MUL(SUB(x, 0.5), 10);
    y = MUL(SUB(y, 0.5), 10);

    // sinc
    float3 ret = ADD(MUL(x, x), MUL(y, y));
    ret = DIV(SIN(ret), ret);
    ret = MUL(0.8, ret);
    ret = ADD(0.2, ret);

    return ret;
}
float3 paramFun_spheres(float3 x, float3 y)
{
    float r = floatParams.x;
    float2 n = float2(intParams.xx);
    // domain
    x = MUL(x, n.x);
    y = MUL(y, n.y);
    x.x = x.x % 1.0;
    y.x = y.x % 1.0;
    x = SUB(MUL(2.0, x), 1.0);
    y = SUB(MUL(2.0, y), 1.0);
    // dist^2 from circle center
    float3 len2 = ADD(MUL(x, x), MUL(y, y));
    if (len2.x >= r * r) // outside
        return float3(1 - r, 0, 0);
    else // inside
        return ADD(1 - r, SQRT(SUB(r * r, len2)));
}


// float3(H, du, dv)
float3 getHD(float2 t)
{
    // init
    float3 x = VAR1_2D(t.x);
    float3 y = VAR2_2D(t.y);

    // function
#ifndef HEIGHT_FUN
    return paramFun_sinc(x,y);
#elif HEIGHT_FUN == 0
    return paramFun_sinc(x,y);
#elif HEIGHT_FUN == 1
    return paramFun_spheres(x,y);
#else
    return paramFun_sinc(x,y);
#endif
}

float2 encodeNorm(float2 Nxy)
{
    return 0.5 * Nxy + 0.5;
}

float4 encodeToTexture(float3 val, float2 t)
{
    float4 ret = float4(0, 0, 0, 1);
    // x: height, yz: encoded normal
    // texture can be unorm
        // calc normal from partial derivatives
        val.yz = normalize(float3(-val.yz, 1)).xy;
    
        ret.x = val.x;
        ret.yz = encodeNorm(val.yz);
    return ret;
}

RWTexture2D<float4> tex2D_uav; // [Height, Norm.x, Norm.y, unused]

[numthreads(16, 16, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    if (any(threadId.xy >= maxSize))
        return;
    float2 t = ((float2) threadId.xy + 0.5) / maxSize; // texture coords
    float3 val = getHD(t);
    tex2D_uav[threadId.xy] = encodeToTexture(val, t);
}
