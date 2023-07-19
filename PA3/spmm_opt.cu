#include "spmm_opt.h"
#include <iostream>
using std::cout;
const int warp_size = 32;
const int size = 2;
__device__ int arr[264339468 + 10];
const int num = 32;
bool use_num = false;
int e_or_v;
typedef void (*SpmmKernelFn)(int *, int *, float *, float *, float *, int, int);
SpmmKernelFn kernel_ptr;
__global__ void spmm_kernel_32_num(int *ptr, int *idx, float *val, float *vin, float *vout, int num_e, int INFEATURE)
{
    int i = blockIdx.x;
    if (i * num * 2 >= num_e)
        return;
    int j = blockIdx.y * warp_size + threadIdx.y;
    __shared__ int sm_y[warp_size * 2];
    __shared__ int sm_x[warp_size * 2];
    __shared__ float sm_v[warp_size * 2];
    int begin = i * num * 2;
    int end = min(i * num * 2 + num * 2, num_e);
    if (begin + threadIdx.y < end)
    {
        sm_y[threadIdx.y] = idx[begin + threadIdx.y];
        sm_x[threadIdx.y] = arr[begin + threadIdx.y];
        sm_v[threadIdx.y] = val[begin + threadIdx.y];
    }
    if (begin + 32 + threadIdx.y < end)
    {
        sm_y[threadIdx.y + 32] = idx[begin + threadIdx.y + 32];
        sm_x[threadIdx.y + 32] = arr[begin + threadIdx.y + 32];
        sm_v[threadIdx.y + 32] = val[begin + threadIdx.y + 32];
    }
    int cur_line = sm_x[0];
    float result = 0;
    for (int m = 0; m < end - begin; m++)
    {
        if (cur_line != sm_x[m])
        {
            atomicAdd(vout + cur_line * INFEATURE + j, result);
            cur_line = sm_x[m];
            result = 0;
        }
        float t_val = sm_v[m];
        int t_index_y = sm_y[m] * INFEATURE + j;

        result += t_val * vin[t_index_y];
    }
    atomicAdd(vout + cur_line * INFEATURE + j, result);
}

__global__ void spmm_kernel_32_line(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int i = blockIdx.x;
    if (i >= num_v)
        return;
    int j = blockIdx.y * warp_size + threadIdx.y;
    int begin = ptr[i];
    int end = ptr[i + 1];
    if (begin == end)
        return;
    __shared__ int sm_k[warp_size];
    __shared__ float sm_v[warp_size];
    float result;
    result = 0.0f;
    for (int cur = begin; cur < end; cur += warp_size)
    {
        if (cur + threadIdx.y < end)
        {
            sm_k[threadIdx.y] = idx[cur + threadIdx.y];
            sm_v[threadIdx.y] = val[cur + threadIdx.y];
        }
        // __syncwarp();
        int cur_end = min(warp_size, end - cur);
        for (int c = 0; c < cur_end; c++)
        {
            float val = sm_v[c];
            int t_idx = sm_k[c] * INFEATURE + j;
            result += val * vin[t_idx];
        }
        // __syncwarp();
    }
    int re = i * INFEATURE + j;
    vout[re] = result;
}

__global__ void spmm_kernel_256_num(int *ptr, int *idx, float *val, float *vin, float *vout, int num_e, int INFEATURE)
{
    int i = blockIdx.x;
    if (i * num * 2 >= num_e)
        return;
    int j = blockIdx.y * warp_size * size * 2 + threadIdx.y;
    __shared__ int sm_y[warp_size * 2];
    __shared__ int sm_x[warp_size * 2];
    __shared__ float sm_v[warp_size * 2];
    int begin = i * num * 2;
    int end = min(i * num * 2 + num * 2, num_e);
    if (begin + threadIdx.y < end)
    {
        sm_y[threadIdx.y] = idx[begin + threadIdx.y];
        sm_x[threadIdx.y] = arr[begin + threadIdx.y];
        sm_v[threadIdx.y] = val[begin + threadIdx.y];
    }
    if (begin + 32 + threadIdx.y < end)
    {
        sm_y[threadIdx.y + 32] = idx[begin + threadIdx.y + 32];
        sm_x[threadIdx.y + 32] = arr[begin + threadIdx.y + 32];
        sm_v[threadIdx.y + 32] = val[begin + threadIdx.y + 32];
    }
    int cur_line = sm_x[0];
    float result[size * 2];
#pragma unroll
    for (int k = 0; k < size * 2; k++)
        result[k] = 0.0f;
    for (int m = 0; m < end - begin; m++)
    {
        if (cur_line != sm_x[m])
        {
            for (int k = 0; k < 2 * size; k++)
                atomicAdd(vout + cur_line * INFEATURE + j + k * warp_size, result[k]);
            cur_line = sm_x[m];
#pragma unroll
            for (int k = 0; k < 2 * size; k++)
                result[k] = 0.0f;
        }
        float t_val = sm_v[m];
        int t_index_y = sm_y[m] * INFEATURE + j;
#pragma unroll
        for (int k = 0; k < 2 * size; k++)
        {
            result[k] += t_val * vin[t_index_y + k * warp_size];
        }
    }
#pragma unroll
    for (int k = 0; k < 2 * size; k++)
        atomicAdd(vout + cur_line * INFEATURE + j + k * warp_size, result[k]);
}

__global__ void spmm_kernel_256_line(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int i = blockIdx.x;
    if (i >= num_v)
        return;
    int j = blockIdx.y * warp_size * size + threadIdx.y;
    int begin = ptr[i];
    int end = ptr[i + 1];
    if (begin == end)
        return;
    __shared__ int sm_k[warp_size];
    __shared__ float sm_v[warp_size];
    float result[size];
#pragma unroll
    for (int k = 0; k < size; k++)
        result[k] = 0.0f;

    for (int cur = begin; cur < end; cur += warp_size)
    {
        if (cur + threadIdx.y < end)
        {
            sm_k[threadIdx.y] = idx[cur + threadIdx.y];
            sm_v[threadIdx.y] = val[cur + threadIdx.y];
        }
        int cur_end = min(warp_size, end - cur);
#pragma unroll
        for (int c = 0; c < cur_end; c++)
        {
            float val = sm_v[c];
            int t_idx = sm_k[c] * INFEATURE + j;
#pragma unroll
            for (int k = 0; k < size; k++)
                result[k] += val * vin[t_idx + k * warp_size];
        }
    }
    int re = i * INFEATURE + j;
#pragma unroll
    for (int k = 0; k < size; k++)
        vout[re + k * warp_size] = result[k];
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: your code
    if (feat_in == 32)
    {
        block.x = 1;
        block.y = warp_size;
        grid.y = feat_in / (warp_size);
        float pre_line = num_e / num_v;
        if (pre_line <= 7.5f)
        {
            use_num = true;
            grid.x = (num_e + num * 2 - 1) / (2 * num);
            int temp[264339468 + 10];
            int *my_d_ptr = new int[num_v + 1];
            cudaMemcpy(my_d_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            long long cur = 0;
            long long *dim = new long long[1];
            for (int i = 0; i < num_v; i++)
            {
                for (int j = my_d_ptr[i]; j < my_d_ptr[i + 1]; j++)
                {
                    temp[cur++] = i;
                }
            }
            assert(cudaSuccess == cudaMemcpyToSymbol(arr, temp, cur * sizeof(int)));
            assert(cudaSuccess == cudaMemset(vout, 0, sizeof(int) * num_v * feat_in));
            kernel_ptr = spmm_kernel_32_num;
            e_or_v = num_e;
        }
        else
        {
            grid.x = num_v;
            kernel_ptr = spmm_kernel_32_line;
            e_or_v = num_v;
        }
    }
    else
    {
        block.x = 1;
        block.y = warp_size;
        grid.y = feat_in / (warp_size * size);
        float pre_line = num_e / num_v;
        if (pre_line <= 7.5f)
        {
            use_num = true;
            grid.x = (num_e + num * 2 - 1) / (2 * num);
            grid.y = feat_in / (warp_size * size * 2);
            int temp[264339468 + 10];
            int *my_d_ptr = new int[num_v + 1];
            cudaMemcpy(my_d_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            long long cur = 0;
            long long *dim = new long long[1];
            for (int i = 0; i < num_v; i++)
            {
                for (int j = my_d_ptr[i]; j < my_d_ptr[i + 1]; j++)
                {
                    temp[cur++] = i;
                }
            }
            assert(cudaSuccess == cudaMemcpyToSymbol(arr, temp, cur * sizeof(int)));
            assert(cudaSuccess == cudaMemset(vout, 0, sizeof(int) * num_v * feat_in));
            kernel_ptr = spmm_kernel_256_num;
            e_or_v = num_e;
        }
        else
        {
            grid.x = num_v;
            kernel_ptr = spmm_kernel_256_line;
            e_or_v = num_v;
        }
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    kernel_ptr<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, e_or_v, feat_in);
}