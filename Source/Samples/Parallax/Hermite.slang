#ifndef HERMITE_INCLUDED
#define HERMITE_INCLUDED

#define HERMITE_GATHER_USE_CONST_ISOLATE 1

// base functions (i = 0, 1; k = 0, 1)
// ai(i) = 1, ai(1-i) = 0, ai'(k) = 0
// bi'(i) = 1, bi'(1-i) = 0, bi(k) = 0
float a0(float x)
{
    return 1 - 3 * x * x + 2 * x * x * x;
}
float a1(float x)
{
    return 3 * x * x - 2 * x * x * x;
}
float b0(float x)
{
    return x - 2 * x * x + x * x * x;
}
float b1(float x)
{
    return -x * x + x * x * x;
}

// aij                                          // sample positions
// xy = bilin(r, aij..)							// query position
// d  = a11 - a00								// sample spacing
// tij = [f(aij), f'(aij).xy]	                // samples
// ret = f(xy)									// queried function value
float hermite3_2d(float2 r, float2 d, float3 t01, float3 t11, float3 t10, float3 t00)
{
    float rx = r.x, ry = r.y;
    float dx = d.x, dy = d.y;

    float a0x = a0(rx), a0y = a0(ry), a1x = a1(rx), a1y = a1(ry);
    float b0x = dx * b0(rx), b0y = dy * b0(ry), b1x = dx * b1(rx), b1y = dy * b1(ry);

    float a01 = a0x * a1y;
    float a11 = a1x * a1y;
    float a10 = a1x * a0y;
    float a00 = a0x * a0y;

    float2 b01 = float2(b0x * a1y, a0x * b1y);
    float2 b11 = float2(b1x * a1y, a1x * b1y);
    float2 b10 = float2(b1x * a0y, a1x * b0y);
    float2 b00 = float2(b0x * a0y, a0x * b0y);

    return
        t01.x * a01 +
        t11.x * a11 +
        t10.x * a10 +
        t00.x * a00 +
        dot(t01.yz, b01) +
        dot(t11.yz, b11) +
        dot(t10.yz, b10) +
        dot(t00.yz, b00);
}

// derivative of the base functions above
float a0d(float x)
{
    return - 6 * x + 6 * x * x;
}
float a1d(float x)
{
    return -a0d(x);
}
float b0d(float x)
{
    return 1 - 4 * x + 3 * x * x;
}
float b1d(float x)
{
    return -2 * x + 3 * x * x;
}

// parameters are as in `hermite3_2d`
float2 hermite3_grad_2d(float2 r, float2 d, float3 t01, float3 t11, float3 t10, float3 t00)
{
    float rx = r.x, ry = r.y;
    float dx = d.x, dy = d.y;

    float a0x = a0(rx), a0y = a0(ry), a1x = a1(rx), a1y = a1(ry);
    float a0xd = dx * a0d(rx), a0yd = dy * a0d(ry), a1xd = dx * a1d(rx), a1yd = dy * a1d(ry);
    float b0x = dx * b0(rx), b0y = dy * b0(ry), b1x = dx * b1(rx), b1y = dy * b1(ry);
    float b0xd = dx*dx * b0d(rx), b0yd = dy*dy * b0d(ry), b1xd = dx*dx * b1d(rx), b1yd = dy*dy * b1d(ry);

    float2 a01 = float2(a0xd * a1y, a0x * a1yd);
    float2 a11 = float2(a1xd * a1y, a1x * a1yd);
    float2 a10 = float2(a1xd * a0y, a1x * a0yd);
    float2 a00 = float2(a0xd * a0y, a0x * a0yd);

    float2 b1_01 = float2(b0xd * a1y, b0x * a1yd);
    float2 b1_11 = float2(b1xd * a1y, b1x * a1yd);
    float2 b1_10 = float2(b1xd * a0y, b1x * a0yd);
    float2 b1_00 = float2(b0xd * a0y, b0x * a0yd);
    
    float2 b2_01 = float2(a0xd * b1y, a0x * b1yd);
    float2 b2_11 = float2(a1xd * b1y, a1x * b1yd);
    float2 b2_10 = float2(a1xd * b0y, a1x * b0yd);
    float2 b2_00 = float2(a0xd * b0y, a0x * b0yd);
    
    return
        t01.x * a01 +
        t11.x * a11 +
        t10.x * a10 +
        t00.x * a00 +
        t01.y * b1_01 +
        t11.y * b1_11 +
        t10.y * b1_10 +
        t00.y * b1_00 +
        t01.z * b2_01 +
        t11.z * b2_11 +
        t10.z * b2_10 +
        t00.z * b2_00;
}


#endif
