// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include "assert.h"

const int INF = 100001;
const int BATCH_LEN = 16;
const int BATCH_NUM = 3;
const int BLOCK_LEN = BATCH_LEN * BATCH_NUM;
namespace
{
    __device__ int get_graph(int n, int *graph, int x, int y)
    {
        return (x < n && y < n) ? graph[x * n + y] : INF;
    }
    __device__ void set_graph(int n, int *graph, int x, int y, int val)
    {
        if (x < n && y < n)
            graph[x * n + y] = val;
    }
    __global__ void step1(int n, int *graph, int p)
    {
        auto i = p * BLOCK_LEN + threadIdx.y;
        auto j = p * BLOCK_LEN + threadIdx.x;
        __shared__ int cur[BLOCK_LEN][BLOCK_LEN];
        for (int a = 0; a < BATCH_NUM; a++)
        {
            for (int b = 0; b < BATCH_NUM; b++)
            {
                cur[threadIdx.y + a * BATCH_LEN][threadIdx.x + b * BATCH_LEN] = get_graph(n, graph, i + a * BATCH_LEN, j + b * BATCH_LEN);
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BLOCK_LEN; k++)
        {
            for (int a = 0; a < BATCH_NUM; a++)
            {
                for (int b = 0; b < BATCH_NUM; b++)
                {
                    auto x = threadIdx.y + a * BATCH_LEN;
                    auto y = threadIdx.x + b * BATCH_LEN;
                    cur[x][y] = min(cur[x][y], cur[x][k] + cur[k][y]);
                }
            }
            __syncthreads();
        }
        for (int a = 0; a < BATCH_NUM; a++)
        {
            for (int b = 0; b < BATCH_NUM; b++)
            {
                auto x = threadIdx.y + a * BATCH_LEN;
                auto y = threadIdx.x + b * BATCH_LEN;
                set_graph(n, graph, i + a * BATCH_LEN, j + b * BATCH_LEN, cur[x][y]);
            }
        }
    }
    __global__ void step2(int n, int *graph, int p)
    {
        auto rank_start = p * BLOCK_LEN;
        __shared__ int center_block[BLOCK_LEN][BLOCK_LEN];
        for (int a = 0; a < BATCH_NUM; a++)
        {
            for (int b = 0; b < BATCH_NUM; b++)
            {
                center_block[threadIdx.y + a * BATCH_LEN][threadIdx.x + b * BATCH_LEN] = get_graph(n, graph, rank_start + threadIdx.y + a * BATCH_LEN, rank_start + threadIdx.x + b * BATCH_LEN);
            }
        }
        __shared__ int graph_xy[BLOCK_LEN][BLOCK_LEN];
        int block_start_x;
        int block_start_y;
        bool is_x = blockIdx.y == 0;
        if (is_x)
        {
            block_start_x = blockIdx.x * BLOCK_LEN;
            if (block_start_x >= rank_start)
                block_start_x += BLOCK_LEN;
            block_start_y = rank_start;
        }
        else
        {
            block_start_x = rank_start;
            block_start_y = blockIdx.x * BLOCK_LEN;
            if (block_start_y >= rank_start)
                block_start_y += BLOCK_LEN;
        }
        for (int a = 0; a < BATCH_NUM; a++)
            for (int b = 0; b < BATCH_NUM; b++)
                graph_xy[threadIdx.y + a * BATCH_LEN][threadIdx.x + b * BATCH_LEN] = get_graph(n, graph, block_start_x + threadIdx.y + a * BATCH_LEN, block_start_y + threadIdx.x + b * BATCH_LEN);

        __syncthreads();
        int dis[BATCH_NUM][BATCH_NUM];
        for (int a = 0; a < BATCH_NUM; a++)
        {
            for (int b = 0; b < BATCH_NUM; b++)
            {
                auto x = threadIdx.y + a * BATCH_LEN;
                auto y = threadIdx.x + b * BATCH_LEN;
                dis[a][b] = graph_xy[x][y];
            }
        }
#pragma unroll
        for (int k = 0; k < BLOCK_LEN; k++)
        {
            for (int a = 0; a < BATCH_NUM; a++)
            {
                for (int b = 0; b < BATCH_NUM; b++)
                {
                    auto x = threadIdx.y + a * BATCH_LEN;
                    auto y = threadIdx.x + b * BATCH_LEN;
                    if (is_x)
                        dis[a][b] = min(dis[a][b], graph_xy[x][k] + center_block[k][y]);
                    else
                        dis[a][b] = min(dis[a][b], center_block[x][k] + graph_xy[k][y]);
                }
            }
            __syncthreads();
        }
        for (int a = 0; a < BATCH_NUM; a++)
            for (int b = 0; b < BATCH_NUM; b++)
                set_graph(n, graph, block_start_x + threadIdx.y + a * BATCH_LEN, block_start_y + threadIdx.x + b * BATCH_LEN, dis[a][b]);
    }
    __global__ void step3(int n, int *graph, int p) // nblock * nblock  blocks
    {
        auto block_start_x = blockIdx.y * BLOCK_LEN;
        auto block_start_y = blockIdx.x * BLOCK_LEN;
        if (block_start_x >= p * BLOCK_LEN)
            block_start_x += BLOCK_LEN;
        if (block_start_y >= p * BLOCK_LEN)
            block_start_y += BLOCK_LEN;
        auto rank_start = p * BLOCK_LEN;
        __shared__ int c_block[BLOCK_LEN][BLOCK_LEN];
        __shared__ int x_block[BLOCK_LEN][BLOCK_LEN];
        __shared__ int y_block[BLOCK_LEN][BLOCK_LEN];
        for (int a = 0; a < BATCH_NUM; a++)
        {
            for (int b = 0; b < BATCH_NUM; b++)
            {
                c_block[threadIdx.y + a * BATCH_LEN][threadIdx.x + b * BATCH_LEN] = get_graph(n, graph, block_start_x + threadIdx.y + a * BATCH_LEN, block_start_y + threadIdx.x + b * BATCH_LEN);
                x_block[threadIdx.y + a * BATCH_LEN][threadIdx.x + b * BATCH_LEN] = get_graph(n, graph, block_start_x + threadIdx.y + a * BATCH_LEN, rank_start + threadIdx.x + b * BATCH_LEN);
                y_block[threadIdx.y + a * BATCH_LEN][threadIdx.x + b * BATCH_LEN] = get_graph(n, graph, rank_start + threadIdx.y + a * BATCH_LEN, block_start_y + threadIdx.x + b * BATCH_LEN);
            }
        }
        __syncthreads();
        int dis[BATCH_NUM][BATCH_NUM];
        for (int a = 0; a < BATCH_NUM; a++)
            for (int b = 0; b < BATCH_NUM; b++)
                dis[a][b] = c_block[threadIdx.y + a * BATCH_LEN][threadIdx.x + b * BATCH_LEN];
#pragma unroll
        for (int k = 0; k < BLOCK_LEN; k++)
        {
            for (int a = 0; a < BATCH_NUM; a++)
            {
                for (int b = 0; b < BATCH_NUM; b++)
                {
                    auto x = threadIdx.y + a * BATCH_LEN;
                    auto y = threadIdx.x + b * BATCH_LEN;
                    dis[a][b] = min(dis[a][b], x_block[x][k] + y_block[k][y]);
                }
            }
        }
        for (int a = 0; a < BATCH_NUM; a++)
            for (int b = 0; b < BATCH_NUM; b++)
                set_graph(n, graph, block_start_x + threadIdx.y + a * BATCH_LEN, block_start_y + threadIdx.x + b * BATCH_LEN, dis[a][b]);
    }
}

void apsp(int n, /* device */ int *graph)
{
    int block_num = (n - 1) / BLOCK_LEN + 1;
    dim3 thr(BATCH_LEN, BATCH_LEN);
    dim3 block1(1);
    dim3 block2(block_num - 1, 2);
    dim3 block3(block_num - 1, block_num - 1);

    for (int p = 0; p < block_num; p++)
    {
        step1<<<block1, thr>>>(n, graph, p);
        step2<<<block2, thr>>>(n, graph, p);
        step3<<<block3, thr>>>(n, graph, p);
    }
}
