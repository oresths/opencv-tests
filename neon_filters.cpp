#elif CV_NEON

struct SymmRowSmallVec_8u32s
{
    SymmRowSmallVec_8u32s() { smallValues = false; }
    SymmRowSmallVec_8u32s( const Mat& _kernel, int _symmetryType )
    {
        kernel = _kernel;
        symmetryType = _symmetryType;
        smallValues = true;
        int k, ksize = kernel.rows + kernel.cols - 1;
        for( k = 0; k < ksize; k++ )
        {
            int v = kernel.ptr<int>()[k];
            if( v < SHRT_MIN || v > SHRT_MAX )
            {
                smallValues = false;
                break;
            }
        }
    }

    int operator()(const uchar* src, uchar* _dst, int width, int cn) const
    {
        //Uncomment the two following lines when runtime support for neon is implemented.
        // if( !checkHardwareSupport(CV_CPU_NEON) )
        //     return 0;

        int i = 0, _ksize = kernel.rows + kernel.cols - 1;
        int* dst = (int*)_dst;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int* kx = kernel.ptr<int>() + _ksize/2;
        if( !smallValues )
            return 0;

        src += (_ksize/2)*cn;
        width *= cn;

        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                {
                    uint16x8_t zq = vdupq_n_u16(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1, x2;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src) );
                        x2 = vld1_u8( (uint8_t *) (src + cn) );

                        uint16x8_t y0, y1, y2;
                        y0 = vaddl_u8(x0, x2);
                        y1 = vshll_n_u8(x1, 1);
                        y2 = vaddq_u16(y0, y1);

                        uint16x8x2_t str;
                        str.val[0] = y2; str.val[1] = zq;
                        vst2q_u16( (uint16_t *) (dst + i), str );
                    }
                }
                else if( kx[0] == -2 && kx[1] == 1 )
                    return 0;
                else
                {
                    int32x4_t k32 = vdupq_n_s32(0);
                    k32 = vld1q_lane_s32(kx, k32, 0);
                    k32 = vld1q_lane_s32(kx + 1, k32, 1);

                    int16x4_t k = vqmovn_s32(k32);

                    uint8x8_t z = vdup_n_u8(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1, x2;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src) );
                        x2 = vld1_u8( (uint8_t *) (src + cn) );

                        int16x8_t y0, y1;
                        int32x4_t y2, y3, y4, y5, y6, y7;
                        y0 = vreinterpretq_s16_u16(vaddl_u8(x1, z));
                        y1 = vreinterpretq_s16_u16(vaddl_u8(x0, x2));
                        y2 = vmull_lane_s16(vget_low_s16(y0), k, 0);
                        y3 = vmull_lane_s16(vget_low_s16(y1), k, 1);
                        y4 = vmull_lane_s16(vget_high_s16(y0), k, 0);
                        y5 = vmull_lane_s16(vget_high_s16(y1), k, 1);
                        y6 = vaddq_s32(y2, y3);
                        y7 = vaddq_s32(y4, y5);

                        vst1q_s32((int32_t *)(dst + i), y6);
                        vst1q_s32((int32_t *)(dst + i + 4), y7);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                    return 0;
                else
                {
                    int32x4_t k32 = vdupq_n_s32(0);
                    k32 = vld1q_lane_s32(kx, k32, 0);
                    k32 = vld1q_lane_s32(kx + 1, k32, 1);
                    k32 = vld1q_lane_s32(kx + 2, k32, 2);

                    int16x4_t k = vqmovn_s32(k32);

                    uint8x8_t z = vdup_n_u8(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1, x2, x3, x4;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src) );
                        x2 = vld1_u8( (uint8_t *) (src + cn) );

                        int16x8_t y0, y1;
                        int32x4_t y2, y3, y4, y5, y6, y7;
                        y0 = vreinterpretq_s16_u16(vaddl_u8(x1, z));
                        y1 = vreinterpretq_s16_u16(vaddl_u8(x0, x2));
                        y2 = vmull_lane_s16(vget_low_s16(y0), k, 0);
                        y3 = vmull_lane_s16(vget_low_s16(y1), k, 1);
                        y4 = vmull_lane_s16(vget_high_s16(y0), k, 0);
                        y5 = vmull_lane_s16(vget_high_s16(y1), k, 1);
                        y6 = vaddq_s32(y2, y3);
                        y7 = vaddq_s32(y4, y5);

                        int16x8_t y8;
                        int32x4_t y9, y10, y11, y12;
                        x3 = vld1_u8( (uint8_t *) (src - cn*2) );
                        x4 = vld1_u8( (uint8_t *) (src + cn*2) );
                        y8 = vreinterpretq_s16_u16(vaddl_u8(x3, x4));
                        y9 = vmull_lane_s16(vget_low_s16(y8), k, 2);
                        y10 = vmull_lane_s16(vget_high_s16(y8), k, 2);
                        y11 = vaddq_s32(y9, y6);
                        y12 = vaddq_s32(y10, y7);

                        vst1q_s32((int32_t *)(dst + i), y11);
                        vst1q_s32((int32_t *)(dst + i + 4), y12);
                    }
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                {
                    uint8x8_t z = vdup_n_u8(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src + cn) );

                        int16x8_t y0;
                        y0 = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(x1, z)),
                                vreinterpretq_s16_u16(vaddl_u8(x0, z)));

                        vst1q_s32((int32_t *)(dst + i), vmovl_s16(vget_low_s16(y0)));
                        vst1q_s32((int32_t *)(dst + i + 4), vmovl_s16(vget_high_s16(y0)));
                    }
                }
                else
                {
                    int32x4_t k32 = vdupq_n_s32(0);
                    k32 = vld1q_lane_s32(kx + 1, k32, 1);

                    int16x4_t k = vqmovn_s32(k32);

                    uint8x8_t z = vdup_n_u8(0);

                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        uint8x8_t x0, x1;
                        x0 = vld1_u8( (uint8_t *) (src - cn) );
                        x1 = vld1_u8( (uint8_t *) (src + cn) );

                        int16x8_t y0;
                        int32x4_t y1, y2;
                        y0 = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(x1, z)),
                            vreinterpretq_s16_u16(vaddl_u8(x0, z)));
                        y1 = vmull_lane_s16(vget_low_s16(y0), k, 1);
                        y2 = vmull_lane_s16(vget_high_s16(y0), k, 1);

                        vst1q_s32((int32_t *)(dst + i), y1);
                        vst1q_s32((int32_t *)(dst + i + 4), y2);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                int32x4_t k32 = vdupq_n_s32(0);
                k32 = vld1q_lane_s32(kx + 1, k32, 1);
                k32 = vld1q_lane_s32(kx + 2, k32, 2);

                int16x4_t k = vqmovn_s32(k32);

                uint8x8_t z = vdup_n_u8(0);

                for( ; i <= width - 8; i += 8, src += 8 )
                {
                    uint8x8_t x0, x1;
                    x0 = vld1_u8( (uint8_t *) (src - cn) );
                    x1 = vld1_u8( (uint8_t *) (src + cn) );

                    int16x8_t y0;
                    int32x4_t y1, y2;
                    y0 = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(x1, z)),
                        vreinterpretq_s16_u16(vaddl_u8(x0, z)));
                    y1 = vmull_lane_s16(vget_low_s16(y0), k, 1);
                    y2 = vmull_lane_s16(vget_high_s16(y0), k, 1);

                    uint8x8_t x2, x3;
                    x2 = vld1_u8( (uint8_t *) (src - cn*2) );
                    x3 = vld1_u8( (uint8_t *) (src + cn*2) );

                    int16x8_t y3;
                    int32x4_t y4, y5;
                    y3 = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(x3, z)),
                        vreinterpretq_s16_u16(vaddl_u8(x2, z)));
                    y4 = vmull_lane_s16(vget_low_s16(y3), k, 2);
                    y5 = vmull_lane_s16(vget_high_s16(y3), k, 2);

                    int32x4_t y6, y7;
                    y6 = vaddq_s32(y1, y4);
                    y7 = vaddq_s32(y2, y5);

                    vst1q_s32((int32_t *)(dst + i), y6);
                    vst1q_s32((int32_t *)(dst + i + 4), y7);
                }
            }
        }

        return i;
    }

    Mat kernel;
    int symmetryType;
    bool smallValues;
};


struct SymmColumnSmallVec_32s16s
{
    SymmColumnSmallVec_32s16s() { symmetryType=0; }
    SymmColumnSmallVec_32s16s(const Mat& _kernel, int _symmetryType, int _bits, double _delta)
    {
        symmetryType = _symmetryType;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        //Uncomment the two following lines when runtime support for neon is implemented.
        // if( !checkHardwareSupport(CV_CPU_NEON) )
        //     return 0;

        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int** src = (const int**)_src;
        const int *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        short* dst = (short*)_dst;
        float32x4_t df4 = vdupq_n_f32(delta);
        int32x4_t d4 = vcvtq_s32_f32(df4);

        if( symmetrical )
        {
            if( ky[0] == 2 && ky[1] == 1 )
            {
                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S1 + i));
                    x2 = vld1q_s32((int32_t const *)(S2 + i));

                    int32x4_t y0, y1, y2, y3;
                    y0 = vaddq_s32(x0, x2);
                    y1 = vqshlq_n_s32(x1, 1);
                    y2 = vaddq_s32(y0, y1);
                    y3 = vaddq_s32(y2, d4);

                    int16x4_t t;
                    t = vqmovn_s32(y3);

                    vst1_s16((int16_t *)(dst + i), t);
                }
            }
            else if( ky[0] == -2 && ky[1] == 1 )
            {
                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S1 + i));
                    x2 = vld1q_s32((int32_t const *)(S2 + i));

                    int32x4_t y0, y1, y2, y3;
                    y0 = vaddq_s32(x0, x2);
                    y1 = vqshlq_n_s32(x1, 1);
                    y2 = vsubq_s32(y0, y1);
                    y3 = vaddq_s32(y2, d4);

                    int16x4_t t;
                    t = vqmovn_s32(y3);

                    vst1_s16((int16_t *)(dst + i), t);
                }
            }
            else if( ky[0] == 10 && ky[1] == 3 )
            {
                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2, x3;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S1 + i));
                    x2 = vld1q_s32((int32_t const *)(S2 + i));

                    x3 = vaddq_s32(x0, x2);

                    int32x4_t y0, y1, y2, y3;
                    y0 = vmulq_n_s32(x1, 10);
                    y1 = vmulq_n_s32(x3, 3);
                    y2 = vaddq_s32(y0, y1);
                    y3 = vaddq_s32(y2, d4);

                    int16x4_t t;
                    t = vqmovn_s32(y3);

                    vst1_s16((int16_t *)(dst + i), t);
                }
            }
            else
            {
                float32x2_t k32 = vdup_n_f32(0);
                k32 = vld1_lane_f32(ky, k32, 0);
                k32 = vld1_lane_f32(ky + 1, k32, 1);

                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2, x3, x4;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S1 + i));
                    x2 = vld1q_s32((int32_t const *)(S2 + i));

                    x3 = vaddq_s32(x0, x2);

                    float32x4_t s0, s1, s2, s3, s4, s5;
                    s0 = vcvtq_f32_s32(x1);
                    s1 = vcvtq_f32_s32(x3);
                    s2 = vmulq_lane_f32(s0, k32, 0);
                    s3 = vmulq_lane_f32(s1, k32, 1);
                    s4 = vaddq_f32(s2, s3);
                    s5 = vaddq_f32(s4, df4);

                    x4 = vcvtq_s32_f32(s5);

                    int16x4_t x5;
                    x5 = vqmovn_s32(x4);

                    vst1_s16((int16_t *)(dst + i), x5);
                }
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S2 + i));

                    int32x4_t y0, y1;
                    y0 = vsubq_s32(x1, x0);
                    y1 = vqaddq_s32(y0, d4);

                    int16x4_t t;
                    t = vqmovn_s32(y1);

                    vst1_s16((int16_t *)(dst + i), t);
                }
            }
            else
            {
                float32x2_t k32 = vdup_n_f32(0);
                k32 = vld1_lane_f32(ky + 1, k32, 1);

                for( ; i <= width - 4; i += 4 )
                {
                    int32x4_t x0, x1, x2, x3;
                    x0 = vld1q_s32((int32_t const *)(S0 + i));
                    x1 = vld1q_s32((int32_t const *)(S2 + i));

                    x2 = vsubq_s32(x0, x1);

                    float32x4_t s0, s1, s2;
                    s0 = vcvtq_f32_s32(x2);
                    s1 = vmulq_lane_f32(s0, k32, 1);
                    s2 = vaddq_f32(s1, df4);

                    x3 = vcvtq_s32_f32(s2);

                    int16x4_t x4;
                    x4 = vqmovn_s32(x3);

                    vst1_s16((int16_t *)(dst + i), x4);
                }
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


struct SymmRowSmallVec_32f
{
    SymmRowSmallVec_32f() {}
    SymmRowSmallVec_32f( const Mat& _kernel, int _symmetryType )
    {
        kernel = _kernel;
        symmetryType = _symmetryType;
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        //Uncomment the two following lines when runtime support for neon is implemented.
        // if( !checkHardwareSupport(CV_CPU_NEON) )
        //     return 0;

        int i = 0, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* src = (const float*)_src + (_ksize/2)*cn;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float* kx = kernel.ptr<float>() + _ksize/2;
        width *= cn;

        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                    return 0;
                else if( kx[0] == -2 && kx[1] == 1 )
                    return 0;
                else
                {
                    return 0;
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                    return 0;
                else
                {
                    float32x2_t k0, k1;
                    k0 = k1 = vdup_n_f32(0);
                    k0 = vld1_lane_f32(kx + 0, k0, 0);
                    k0 = vld1_lane_f32(kx + 1, k0, 1);
                    k1 = vld1_lane_f32(kx + 2, k1, 0);

                    for( ; i <= width - 4; i += 4, src += 4 )
                    {
                        float32x4_t x0, x1, x2, x3, x4;
                        x0 = vld1q_f32(src);
                        x1 = vld1q_f32(src - cn);
                        x2 = vld1q_f32(src + cn);
                        x3 = vld1q_f32(src - cn*2);
                        x4 = vld1q_f32(src + cn*2);

                        float32x4_t y0;
                        y0 = vmulq_lane_f32(x0, k0, 0);
                        y0 = vmlaq_lane_f32(y0, vaddq_f32(x1, x2), k0, 1);
                        y0 = vmlaq_lane_f32(y0, vaddq_f32(x3, x4), k1, 0);

                        vst1q_f32(dst + i, y0);
                    }
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    return 0;
                else
                {
                    return 0;
                }
            }
            else if( _ksize == 5 )
            {
                float32x2_t k;
                k = vdup_n_f32(0);
                k = vld1_lane_f32(kx + 1, k, 0);
                k = vld1_lane_f32(kx + 2, k, 1);

                for( ; i <= width - 4; i += 4, src += 4 )
                {
                    float32x4_t x0, x1, x2, x3;
                    x0 = vld1q_f32(src - cn);
                    x1 = vld1q_f32(src + cn);
                    x2 = vld1q_f32(src - cn*2);
                    x3 = vld1q_f32(src + cn*2);

                    float32x4_t y0;
                    y0 = vmulq_lane_f32(vsubq_f32(x1, x0), k, 0);
                    y0 = vmlaq_lane_f32(y0, vsubq_f32(x3, x2), k, 1);

                    vst1q_f32(dst + i, y0);
                }
            }
        }

        return i;
    }

    Mat kernel;
    int symmetryType;
};


struct SymmColumnVec_32f16s
{
    SymmColumnVec_32f16s() { symmetryType=0; }
    SymmColumnVec_32f16s(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
        //Uncomment the following line when runtime support for neon is implemented.
        // neon_supported = checkHardwareSupport(CV_CPU_NEON);
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        //Uncomment the two following lines when runtime support for neon is implemented.
        // if( !neon_supported )
        //     return 0;

        int _ksize = kernel.rows + kernel.cols - 1;
        int ksize2 = _ksize / 2;
        const float* ky = kernel.ptr<float>() + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        const float *S, *S2;
        short* dst = (short*)_dst;

        float32x4_t d4 = vdupq_n_f32(delta);

        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;

            float32x4_t accl, acch;
            accl = vdupq_n_f32(0);
            acch = vdupq_n_f32(0);

            float32x2_t k32;
            k32 = vdup_n_f32(0);
            k32 = vld1_lane_f32(ky, k32, 0);
            k32 = vld1_lane_f32(ky + 1, k32, 1);

            for( ; i <= width - 8; i += 8 )
            {
                float32x4_t x0l, x0h, x1l, x1h, x2l, x2h;

                S = src[0] + i;

                x0l = vld1q_f32(S);
                x0h = vld1q_f32(S + 4);

                S = src[1] + i;
                S2 = src[-1] + i;

                x1l = vld1q_f32(S);
                x1h = vld1q_f32(S + 4);
                x2l = vld1q_f32(S2);
                x2h = vld1q_f32(S2 + 4);

                accl = vmlaq_lane_f32(d4, x0l, k32, 0);
                acch = vmlaq_lane_f32(d4, x0h, k32, 0);
                accl = vmlaq_lane_f32(accl, vaddq_f32(x1l, x2l), k32, 1);
                acch = vmlaq_lane_f32(acch, vaddq_f32(x1h, x2h), k32, 1);

                for( k = 2; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;

                    float32x4_t x3l, x3h, x4l, x4h;
                    x3l = vld1q_f32(S);
                    x3h = vld1q_f32(S + 4);
                    x4l = vld1q_f32(S2);
                    x4h = vld1q_f32(S2 + 4);

                    accl = vmlaq_n_f32(accl, vaddq_f32(x3l, x4l), ky[k]);
                    acch = vmlaq_n_f32(acch, vaddq_f32(x3h, x4h), ky[k]);
                }

                int32x4_t s32l, s32h;
                s32l = vcvtq_s32_f32(accl);
                s32h = vcvtq_s32_f32(acch);

                int16x4_t s16l, s16h;
                s16l = vqmovn_s32(s32l);
                s16h = vqmovn_s32(s32h);

                vst1_s16((int16_t *)(dst + i), s16l);
                vst1_s16((int16_t *)(dst + i + 4), s16h);
            }
        }
        else
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f, s0 = d4, s1 = d4, s2 = d4, s3 = d4;
                __m128 x0, x1;
                S = src[0] + i;

                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_sub_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_sub_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_sub_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
                }

                __m128i s0i = _mm_cvtps_epi32(s0);
                __m128i s1i = _mm_cvtps_epi32(s1);
                __m128i s2i = _mm_cvtps_epi32(s2);
                __m128i s3i = _mm_cvtps_epi32(s3);

                _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0i, s1i));
                _mm_storeu_si128((__m128i*)(dst + i + 8), _mm_packs_epi32(s2i, s3i));
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
    bool neon_supported;
};


typedef RowNoVec RowVec_8u32s;
typedef RowNoVec RowVec_16s32f;
typedef RowNoVec RowVec_32f;
typedef ColumnNoVec SymmColumnVec_32s8u;
typedef ColumnNoVec SymmColumnVec_32f;
typedef SymmColumnSmallNoVec SymmColumnSmallVec_32f;
typedef FilterNoVec FilterVec_8u;
typedef FilterNoVec FilterVec_8u16s;
typedef FilterNoVec FilterVec_32f;