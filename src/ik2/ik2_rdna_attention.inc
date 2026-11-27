// Adapted from Project Maya 53885cef8faf0d3b3f7e28c0b1537a056c1f3ecf, src/ik2/ik2_rdna.inc.
// MIT: third_party/project-maya/{LICENSE,NOTICE.md}. Only the 24-KiB prompt-attention slice.
#if defined(IK2_RDNA_WMMA)
__device__ __forceinline__ ik2::rdna::wmma_h16 rm_frag(const uint16_t* p) {
    union {
        uint4 u[2];
        ik2::rdna::wmma_h16 v;
    } f;
    f.u[0] = *(const uint4*) p, f.u[1] = *(const uint4*) (p + 8);
    return f.v;
}
#endif

constexpr int kRfNW = 4, kRfBN = 32, kRfLd = kFaD + 8, kRfVld = kRfBN + 8;
template <class KVT>
__global__ void __launch_bounds__(kRfNW * 32)
attn_flash_rdna_kernel(const uint16_t* __restrict__ q16, int q_ld, const KVT* __restrict__ kc,
                       const KVT* __restrict__ vc, int kv_cap, int H, int KV, int Tq, int qpos0, float qk_scale,
                       const float* __restrict__ bias, int extent, int window, int n_split, float* __restrict__ o_part,
                       float2* __restrict__ ml_part) {
#if defined(IK2_RDNA_WMMA)
    using namespace ik2::rdna;
    __shared__ __align__(16) uint16_t Ks[kRfBN * kRfLd];          // [key][dim]
    __shared__ __align__(16) uint16_t Vt[kFaD * kRfVld];          // [dim][key]
    __shared__ __align__(16) uint16_t Pss[kRfNW * 16 * kRfVld];   // each warp's P [16][key]
    constexpr bool F8 = sizeof(KVT) == 1;
    const int tid = threadIdx.x, warp = tid >> 5, hi = (tid & 31) >> 4, ln = tid & 15;
    uint16_t* Ps = Pss + warp * 16 * kRfVld;
    const int h = blockIdx.y, split = blockIdx.z, g = h / (H / KV), KD = KV * kFaD;
    constexpr int BM = 16 * kRfNW;
    // (the tiles are the CUDA kernel's 64-key ones cut in two: the same keys, so attn_flash_splits' counts hold)
    const int i0 = blockIdx.x * BM, i_last = min(Tq, i0 + BM) - 1;
    const int k_hi = qpos0 + i_last + 1, k_lo = max(0, qpos0 + i0 - window + 1);
    const int n_tiles = (k_hi - k_lo + kRfBN - 1) / kRfBN;
    const int per = (n_tiles + n_split - 1) / n_split;
    const int t0 = min(n_tiles, split * per), t1 = min(n_tiles, t0 + per);
    const int wi = i0 + warp * 16;
    wmma_h16 qf[kFaD / 16];   // (the lane's query row wi + ln, its 128 dims)
    {
        const int iq = wi + ln;
        const uint16_t* qr = q16 + (size_t) (iq < Tq ? iq : 0) * q_ld + (size_t) h * kFaD;
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) qf[kk] = iq < Tq ? rm_frag(qr + kk * 16) : wmma_h16{};
    }
    wmma_f8 o[kFaD / 16];
#pragma unroll
    for (int j = 0; j < kFaD / 16; ++j) o[j] = wmma_f8{};
    float mx[8], ls[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) mx[i] = -INFINITY, ls[i] = 0.0f;
    const float sc = qk_scale * kLog2e;
    // A tile's values: K by (key c / 16, 8 dims from 8 (c % 16)), V by a pair of keys (tid % 16) x 16 dims (tid / 16),
    // so that the transposed V goes in as words, a half-warp a row's 16 words.
    using Raw = typename std::conditional<F8, uint2, uint4>::type;
    Raw zr;
    std::memset(&zr, 0, sizeof zr);
    const int vp = tid & 15, vg = tid >> 4;
    auto h8 = [&](const Raw& r) -> uint4 {
        if constexpr (F8) return e4m3x8_h8(r);
        else return r;
    };
    for (int t = t0; t < t1; ++t) {
        const int kt = k_lo + t * kRfBN;
        __syncthreads();   // (the previous tile's reads done)
#pragma unroll
        for (int it = 0; it < (kRfBN * kFaD / 8) / (kRfNW * 32); ++it) {
            const int c = it * kRfNW * 32 + tid, row = c >> 4, col = (c & 15) * 8, kp = kt + row;
            const bool ok = kp < k_hi;
            const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
            *(uint4*) (Ks + row * kRfLd + col) = h8(ok ? *(const Raw*) (kc + (size_t) pr * KD + (size_t) g * kFaD + col) : zr);
        }
        {
            Raw vr[2][2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int kp = kt + 2 * vp + j;
                const bool ok = kp < k_hi;
                const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
                const KVT* src = vc + (size_t) pr * KD + (size_t) g * kFaD + 16 * vg;
                vr[j][0] = ok ? *(const Raw*) src : zr;
                vr[j][1] = ok ? *(const Raw*) (src + 8) : zr;
            }
            uint32_t* Vt32 = (uint32_t*) Vt;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint4 a = h8(vr[0][i]), b = h8(vr[1][i]);
                const uint32_t aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int d = 16 * vg + 8 * i + 2 * m;
                    Vt32[d * (kRfVld / 2) + vp] = __byte_perm(aw[m], bw[m], 0x5410);
                    Vt32[(d + 1) * (kRfVld / 2) + vp] = __byte_perm(aw[m], bw[m], 0x7632);
                }
            }
        }
        __syncthreads();
        // does this warp's rows see any key of the tile, and is any distance within the bias's extent?
        const bool any = !(kt > qpos0 + wi + 15 || kt + kRfBN - 1 < qpos0 + wi - window + 1) && wi < Tq;
        const bool near = qpos0 + wi - (kt + kRfBN - 1) < extent;
        if (!any) continue;
        wmma_f8 s[kRfBN / 16];
#pragma unroll
        for (int j = 0; j < kRfBN / 16; ++j) s[j] = wmma_f8{};
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk)
#pragma unroll
            for (int j = 0; j < kRfBN / 16; ++j)
                s[j] = wmma_f16(qf[kk], rm_frag(Ks + (16 * j + ln) * kRfLd + kk * 16), s[j]);
        // logits (log2 units): scale, bias (the table's value at the distance, 0 out of its extent), mask; the rows'
        // new maxima, the output's rescale
        float u[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = wmma_row(i, hi), ia = wi + m, qa = qpos0 + ia;
            const float* brow = bias + ((size_t) (ia < Tq ? ia : 0) * H + h) * extent;
            float mxi = -INFINITY;
#pragma unroll
            for (int j = 0; j < kRfBN / 16; ++j) {
                const int kp = kt + 16 * j + ln, dist = qa - kp;
                float v = s[j][i] * sc;
                if (near && ia < Tq && dist >= 0 && dist < extent) v += brow[dist] * kLog2e;
                v = (dist < 0 || kp < qa - window + 1 || ia >= Tq) ? -INFINITY : v;
                s[j][i] = v;
                mxi = fmaxf(mxi, v);
            }
#pragma unroll
            for (int sh = 1; sh < 16; sh <<= 1) mxi = fmaxf(mxi, __shfl_xor_sync(0xffffffffu, mxi, sh));
            const float mn = fmaxf(mx[i], mxi);
            const float cf = mn == -INFINITY ? 1.0f : exp2f(mx[i] - mn);
            u[i] = mn == -INFINITY ? 0.0f : mn;
            mx[i] = mn;
            ls[i] *= cf;
#pragma unroll
            for (int j = 0; j < kFaD / 16; ++j) o[j][i] *= cf;
        }
#pragma unroll
        for (int j = 0; j < kRfBN / 16; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = exp2f(s[j][i] - u[i]);
                ls[i] += p;
                Ps[wmma_row(i, hi) * kRfVld + 16 * j + ln] = f2h(p);
            }
        __syncwarp();
#pragma unroll
        for (int kk = 0; kk < kRfBN / 16; ++kk) {
            const wmma_h16 a = rm_frag(Ps + ln * kRfVld + kk * 16);
#pragma unroll
            for (int j = 0; j < kFaD / 16; ++j) o[j] = wmma_f16(a, rm_frag(Vt + (16 * j + ln) * kRfVld + kk * 16), o[j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int sh = 1; sh < 16; sh <<= 1) ls[i] += __shfl_xor_sync(0xffffffffu, ls[i], sh);
    // the split's partials: o_part [split][Tq][H][D], ml_part [split][Tq][H] = (max in log2 units, sum)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ia = wi + wmma_row(i, hi);
        if (ia >= Tq) continue;
        float* op = o_part + (((size_t) split * Tq + ia) * H + h) * kFaD + ln;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j) op[16 * j] = o[j][i];
        if (ln == 0) ml_part[((size_t) split * Tq + ia) * H + h] = make_float2(mx[i], ls[i]);
    }
#else
    __trap();   // (never launched: attn_flash_available() is false without the matrix cores)
#endif
}

template <class KVT>
void attn_flash_rdna(const uint16_t* q16, int q_ld, const KVT* kc, const KVT* vc, int kv_cap, int H, int KV, int Tq,
                     int qpos0, float qk_scale, const float* bias, int extent, int window, int n_split, float* o_part,
                     float2* ml_part, cudaStream_t s) {
    const dim3 grid((unsigned) ((Tq + 16 * kRfNW - 1) / (16 * kRfNW)), (unsigned) H, (unsigned) n_split);
    attn_flash_rdna_kernel<KVT><<<grid, kRfNW * 32, 0, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale,
                                                            bias, extent, window, n_split, o_part, ml_part);
    check("attn_flash_rdna");
}
