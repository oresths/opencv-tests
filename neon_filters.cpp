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

                    // int32x4_t accl = vdupq_n_s32(0), acch = vdupq_n_s32(0);
                    int32x4_t acc0, acc1, acc2, acc3;

                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        acc0 = acc1 = acc2 = acc3 = vdupq_n_s32(0);

                        uint8x16_t x0, x1, x2;
                        x0 = vld1q_u8( (uint8_t *) (src - cn) );
                        x1 = vld1q_u8( (uint8_t *) (src) );
                        x2 = vld1q_u8( (uint8_t *) (src + cn) );

                        int16x8_t y0, y1, y2, y3;
                        y0 = vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(x1), z));
                        y1 = vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(x0), vget_low_u8(x2)));
                        y2 = vreinterpretq_s16_u16(vaddl_u8(vget_high_u8(x1), z));
                        y3 = vreinterpretq_s16_u16(vaddl_u8(vget_high_u8(x0), vget_high_u8(x2)));
                        acc0 = vmlal_lane_s16(acc0, vget_low_s16(y0), k, 0);
                        acc0 = vmlal_lane_s16(acc0, vget_low_s16(y1), k, 1);
                        acc1 = vmlal_lane_s16(acc1, vget_high_s16(y0), k, 0);
                        acc1 = vmlal_lane_s16(acc1, vget_high_s16(y1), k, 1);
                        acc2 = vmlal_lane_s16(acc2, vget_low_s16(y2), k, 0);
                        acc2 = vmlal_lane_s16(acc2, vget_low_s16(y3), k, 1);
                        acc3 = vmlal_lane_s16(acc3, vget_high_s16(y2), k, 0);
                        acc3 = vmlal_lane_s16(acc3, vget_high_s16(y3), k, 1);

                        vst1q_s32((int32_t *)(dst + i), acc0);
                        vst1q_s32((int32_t *)(dst + i + 4), acc1);
                        vst1q_s32((int32_t *)(dst + i + 8), acc2);
                        vst1q_s32((int32_t *)(dst + i + 12), acc3);
                    }


                    // __m128i k0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[0]), 0),
                    // k1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[1]), 0);
                    // k0 = _mm_packs_epi32(k0, k0);
                    // k1 = _mm_packs_epi32(k1, k1);

                    // for( ; i <= width - 16; i += 16, src += 16 )
                    // {
                    //     __m128i x0, x1, x2, y0, y1, t0, t1, z0, z1, z2, z3;
                    //     x0 = _mm_loadu_si128((__m128i*)(src - cn));
                    //     x1 = _mm_loadu_si128((__m128i*)src);
                    //     x2 = _mm_loadu_si128((__m128i*)(src + cn));
                    //     y0 = _mm_add_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x2, z));
                    //     x0 = _mm_add_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x2, z));
                    //     y1 = _mm_unpackhi_epi8(x1, z);
                    //     x1 = _mm_unpacklo_epi8(x1, z);

                    //     t1 = _mm_mulhi_epi16(x1, k0);
                    //     t0 = _mm_mullo_epi16(x1, k0);
                    //     x2 = _mm_mulhi_epi16(x0, k1);
                    //     x0 = _mm_mullo_epi16(x0, k1);
                    //     z0 = _mm_unpacklo_epi16(t0, t1);
                    //     z1 = _mm_unpackhi_epi16(t0, t1);
                    //     z0 = _mm_add_epi32(z0, _mm_unpacklo_epi16(x0, x2));
                    //     z1 = _mm_add_epi32(z1, _mm_unpackhi_epi16(x0, x2));

                    //     t1 = _mm_mulhi_epi16(y1, k0);
                    //     t0 = _mm_mullo_epi16(y1, k0);
                    //     y1 = _mm_mulhi_epi16(y0, k1);
                    //     y0 = _mm_mullo_epi16(y0, k1);
                    //     z2 = _mm_unpacklo_epi16(t0, t1);
                    //     z3 = _mm_unpackhi_epi16(t0, t1);
                    //     z2 = _mm_add_epi32(z2, _mm_unpacklo_epi16(y0, y1));
                    //     z3 = _mm_add_epi32(z3, _mm_unpackhi_epi16(y0, y1));
                    //     _mm_store_si128((__m128i*)(dst + i), z0);
                    //     _mm_store_si128((__m128i*)(dst + i + 4), z1);
                    //     _mm_store_si128((__m128i*)(dst + i + 8), z2);
                    //     _mm_store_si128((__m128i*)(dst + i + 12), z3);
                    // }
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                    return 0;
                else
                {
                    return 0;
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
                    return 0;
                }
            }
            else if( _ksize == 5 )
            {
                return 0;
            }
        }

        // src -= (_ksize/2)*cn;
        // kx -= _ksize/2;
        // for( ; i <= width - 4; i += 4, src += 4 )
        // {
        //     __m128i f, s0 = z, x0, x1;

        //     for( k = j = 0; k < _ksize; k++, j += cn )
        //     {
        //         f = _mm_cvtsi32_si128(kx[k]);
        //         f = _mm_shuffle_epi32(f, 0);
        //         f = _mm_packs_epi32(f, f);

        //         x0 = _mm_cvtsi32_si128(*(const int*)(src + j));
        //         x0 = _mm_unpacklo_epi8(x0, z);
        //         x1 = _mm_mulhi_epi16(x0, f);
        //         x0 = _mm_mullo_epi16(x0, f);
        //         s0 = _mm_add_epi32(s0, _mm_unpacklo_epi16(x0, x1));
        //     }
        //     _mm_store_si128((__m128i*)(dst + i), s0);
        // }

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
        __m128 df4 = _mm_set1_ps(delta);
        __m128i d4 = _mm_cvtps_epi32(df4);

        if( symmetrical )
        {
            if( ky[0] == 2 && ky[1] == 1 )
            {
                for( ; i <= width - 8; i += 8 )
                {
                    __m128i s0, s1, s2, s3, s4, s5;
                    s0 = _mm_load_si128((__m128i*)(S0 + i));
                    s1 = _mm_load_si128((__m128i*)(S0 + i + 4));
                    s2 = _mm_load_si128((__m128i*)(S1 + i));
                    s3 = _mm_load_si128((__m128i*)(S1 + i + 4));
                    s4 = _mm_load_si128((__m128i*)(S2 + i));
                    s5 = _mm_load_si128((__m128i*)(S2 + i + 4));
                    s0 = _mm_add_epi32(s0, _mm_add_epi32(s4, _mm_add_epi32(s2, s2)));
                    s1 = _mm_add_epi32(s1, _mm_add_epi32(s5, _mm_add_epi32(s3, s3)));
                    s0 = _mm_add_epi32(s0, d4);
                    s1 = _mm_add_epi32(s1, d4);
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0, s1));
                }
            }
            else if( ky[0] == -2 && ky[1] == 1 )
            {
                return 0;
            }
            else
            {
                return 0;
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128i s0, s1, s2, s3;
                    s0 = _mm_load_si128((__m128i*)(S2 + i));
                    s1 = _mm_load_si128((__m128i*)(S2 + i + 4));
                    s2 = _mm_load_si128((__m128i*)(S0 + i));
                    s3 = _mm_load_si128((__m128i*)(S0 + i + 4));
                    s0 = _mm_add_epi32(_mm_sub_epi32(s0, s2), d4);
                    s1 = _mm_add_epi32(_mm_sub_epi32(s1, s3), d4);
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0, s1));
                }
            }
            else
            {
                return 0;
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};




typedef RowNoVec RowVec_8u32s;
typedef RowNoVec RowVec_16s32f;
typedef RowNoVec RowVec_32f;
//typedef SymmRowSmallNoVec SymmRowSmallVec_8u32s;
typedef SymmRowSmallNoVec SymmRowSmallVec_32f;
typedef ColumnNoVec SymmColumnVec_32s8u;
typedef ColumnNoVec SymmColumnVec_32f16s;
typedef ColumnNoVec SymmColumnVec_32f;
// typedef SymmColumnSmallNoVec SymmColumnSmallVec_32s16s;
typedef SymmColumnSmallNoVec SymmColumnSmallVec_32f;
typedef FilterNoVec FilterVec_8u;
typedef FilterNoVec FilterVec_8u16s;
typedef FilterNoVec FilterVec_32f;