#include "spmm_cpu_opt.h"
#include <x86intrin.h>
#include "omp.h"
void run_spmm_cpu_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_len)
{
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < num_v; ++i)
    {
        for (int j = ptr[i]; j < ptr[i + 1]; ++j)
        {
            for (int k = 0; k < feat_len; ++k)
            {
                vout[i * feat_len + k] += vin[idx[j] * feat_len + k] * val[j];
            }
        }
    }
    // #pragma omp parallel for schedule(guided)
    //     for (int i = 0; i < num_v; ++i)
    //     {
    // #pragma omp simd
    //         for (int k = 0; k < feat_len; ++k)
    //         {
    //             __m128 sum = _mm_setzero_ps(); // 创建一个全零的 XMM 寄存器
    //             int begin = ptr[i];
    //             int end = ptr[i + 1];
    //             int del = end - begin;
    //             end = begin + del - del % 4;
    //             float my_vin[4];
    //             for (int j = ptr[i]; j < end; j += 4)
    //             {
    //                 for (int m = 0; m < 4; m++)
    //                 {
    //                     my_vin[m] = vin[idx[j + m] * feat_len + k];
    //                 }
    //                 __m128 val_values = _mm_loadu_ps(&val[j]);
    //                 __m128 vin_values = _mm_loadu_ps(&my_vin[0]);
    //                 __m128 product = _mm_mul_ps(vin_values, val_values);
    //                 sum = _mm_add_ps(sum, product);
    //             }
    //             float temp[4];
    //             _mm_store_ps(temp, sum);
    //             float tot = 0.0f;
    //             for (int i = 0; i < 4; i++)
    //                 tot += temp[i];
    //             for (int j = end; j < ptr[i + 1]; j++)
    //                 tot += val[j] * vin[idx[j] * feat_len + k];
    //             *(vout + i * feat_len + k) = tot;
    //             // _mm_storeu_ps((vout + i * feat_len + k), tot); // 存储 XMM 寄存器中的结果到临时数组
    //         }
    //     }
}

void SpMMCPUOpt::preprocess(float *vin, float *vout)
{
}

void SpMMCPUOpt::run(float *vin, float *vout)
{
    run_spmm_cpu_placeholder(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
