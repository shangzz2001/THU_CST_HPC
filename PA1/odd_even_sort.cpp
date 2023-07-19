#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void Worker::sort()
{
    /** Your code ... */
    // you can use variables in class Worker: n, nprocs, rank, block_len, data
    // openmpi odd-even sort
    // merge sort
    void merge(int *a, int *b, int *c, int n, int m)
    {
        int i = 0, j = 0, k = 0;
        while (i < n && j < m)
        {
            if (a[i] < b[j])
                c[k++] = a[i++];
            else
                c[k++] = b[j++];
        }
        while (i < n)
            c[k++] = a[i++];
        while (j < m)
            c[k++] = b[j++];
    }
}
