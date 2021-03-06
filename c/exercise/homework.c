#define _GNU_SOURCE
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>
#include <numa.h>
#include <numaif.h>

#define C_MEMCPY "C library: memcpy"
#define SINGLE_THREAD "Singlethreading"
#define MULTI_THREAD "Multithreading"
#define MULTI_AFFINITY "Multithreading with affinity"
#define MEM_LOCAL "Multithreading with numa_alloc_local"
#define MEM_INTER "Multithreading with numa_alloc_interleaved"

/* You may need to define struct here */

/*!
 * \brief subroutine function
 *
 * \param arg, input arguments pointer
 * \return void*, return pointer
 */

typedef struct args_t {
    float *dst;
    float *src;
    size_t len;
} args_t;

void *mt_memcpy(void *arg) {
  args_t *mem = (args_t *)arg;
  float *in = (float *)mem->src;
  float *out = (float *)mem->dst;
  size_t size = mem->len;

  for (size_t i = 0; i < size / 4; ++i) {
    out[i] = in[i];
  }
  if (size % 4) {
    memcpy(out + size / 4, in + size / 4, size % 4);
  }
  pthread_exit(NULL);
}


void multi_thread_memcpy(void *dst, const void *src, size_t size, int k) {
    pthread_t thr[k];
    args_t args[k];
    float *st_src = (float *)src;
    float *st_dst = (float *)dst;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (int i=0; i < k; ++i)
    {
    	args[i].src = st_src;
	args[i].dst = st_dst;
	args[i].len = size / k;
	st_src = st_src + size / (4*k);
	st_dst = st_dst + size / (4*k);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    float time_creat_struct = (end.tv_sec - start.tv_sec) * 1.0e6 +
                       (end.tv_nsec - start.tv_nsec) * 1.0e-3;
    printf("Structures Creating Time %.2f s.\n",
           (time_creat_struct*1000.0) );
    
    args[k-1].len = size / k + size % k;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (int i=0; i < k; ++i)
    {
	if (pthread_create(&thr[i], NULL,
	mt_memcpy, (void *)&args[i]) != 0)
	{
	    fprintf(stderr, "pthread_create failed.");
          exit(1);
	}
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    float time_creat_thread = (end.tv_sec - start.tv_sec) * 1.0e6 + (end.tv_nsec - start.tv_nsec) * 1.0e-3;
    printf("Threads Creating Time  %.2f s.\n",
             (time_creat_thread*1000.0) );

    for (int i = 0; i < k; ++i) {
      if ( pthread_join(thr[i], NULL) != 0 ) {
        fprintf(stderr, "pthread_join failed.\n");
        exit(1);
      }
    }
}




void multi_thread_memcpy_with_affinity(void *dst, const void *src, size_t size, int k) {
    pthread_t thr[k];
    args_t args[k];
    float *st_src = (float *)src;
    float *st_dst = (float *)dst;
    cpu_set_t cpu_set[k];
    pthread_attr_t attr[k];
    int flag = 1;

    if (flag) {
      for (int i = 0; i < k; ++i) {
      if ( pthread_attr_init(&attr[i]) != 0 ) {
        fprintf(stderr, "pthread_attr_init failed.\n");
        exit(1);
      }
    }
    int cpu;
    assert( syscall(SYS_getcpu, &cpu, NULL, NULL) == 0 );
    printf("Main thread: on the %d cpu \n", cpu);
    int start = cpu % 2;
    printf("Affinity mask on CPUs (%d, %d, %d, ...  %s)\n", start,
           start + 2, start + 4, start ? "2n+1" : "2n");
    size_t nprocs = get_nprocs();
    int i = 0, j = start;
    while (i < 1) {

      printf("%d", i);

      if (j == cpu) {
        j += 2;
        continue;
      }
      if (j >= nprocs) {
        j = start;
      }

      CPU_ZERO(&cpu_set[i]);
      CPU_SET(j, &cpu_set[i]);
      pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpu_set[i]);
      ++i;
      j += 2;
    }
  }
  else {
    for (int i = 0; i < k; ++i) {
      if ( pthread_attr_init(&attr[i]) != 0 ) {
        fprintf(stderr, "pthread_attr_init failed.\n");
        exit(1);
      }
    }
  }

    for (int i=0; i < k; ++i)
    {
        args[i].src = st_src;
        args[i].dst = st_dst;
        args[i].len = size / k;
        st_src = st_src + size / (4*k);
        st_dst = st_dst + size / (4*k);
    }
    
    args[k-1].len = size / k + size % k;
    
    for (int i=0; i < k; ++i)
    {
        if (pthread_create(&thr[i], &attr[i],
        mt_memcpy, (void *)&args[i]) != 0)
        {
            fprintf(stderr, "pthread_create failed.");
          exit(1);
	      }
    }
   
    for (int i = 0; i < k; ++i) {
      if ( pthread_join(thr[i], NULL) != 0 ) {
        fprintf(stderr, "pthread_join failed.\n");
        exit(1);
      }
}
}


void single_thread_memcpy(void *dst, const void *src, size_t size) {
  float *in = (float *)src;
  float *out = (float *)dst;

  for (size_t i = 0; i < size / 4; ++i) {
    out[i] = in[i];
  }
  if (size % 4) {
    memcpy(out + size / 4, in + size / 4, size % 4);
  }
}

int execute(const char *command, int len, int k)
{
  /* allocate memory */
  float *dst = (float *) malloc( len * sizeof(float) );
  float *src = (float *) malloc( len * sizeof(float) );
  assert(dst != NULL);
  assert(src != NULL);

  /* warmup */
  memcpy(dst, src, len * sizeof(float));

  /* timing the memcpy */
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  if ( strcmp(command, C_MEMCPY)==0 )
  {
    memcpy(dst, src, len*sizeof(float));
  }
  else if ( strcmp(command, SINGLE_THREAD)==0 )
  {
    single_thread_memcpy(dst, src, len*sizeof(float));
  }
  else if ( strcmp(command, MULTI_THREAD)==0 )
  {
    multi_thread_memcpy(dst, src, len*sizeof(float), k);
  }
  else if ( strcmp(command, MULTI_AFFINITY)==0 )
  {
    multi_thread_memcpy_with_affinity(dst, src, len*sizeof(float), k);
  }
  else
  {
    fprintf(stderr, "execution failure.\n");
    goto out;
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  /* check correctness (with "warmup" disabled) */
  assert( memcmp(src, dst, len*sizeof(float)) == 0 );

  float delta_us = (end.tv_sec - start.tv_sec) * 1.0e6 +
                     (end.tv_nsec - start.tv_nsec) * 1.0e-3;
  printf("[%s]\tThe throughput is %.2f Gbps.\n",
          command, len*sizeof(float)*8 / (delta_us*1000.0) );

out: 
  /* free the memory */
  free(dst);
  free(src);

  return 0;
}

#ifdef BUILD_BONUS
int execute_numa(const char *command, int len, int k)
{
  /* allocate memory */
  float *dst, *src;
  if ( strcmp(command, MEM_LOCAL)==0 )
  {
    /* allocate memory (`*src`, `*dst`) locally on the current node */
    //TODO: (Bonus) Your code here.
  }
  else if ( strcmp(command, MEM_INTER)==0 )
  {
    /* allocate memory (`*src`, `*dst`) interleaved on each node */
    //TODO: (Bonus) Your code here.
  }
  else
  {
    fprintf(stderr, "numa execution failure.\n");
    return -1;
  }
  assert(dst != NULL);
  assert(src != NULL);

  /* warmup */
  memcpy(dst, src, len * sizeof(float));

  /* timing the memcpy */
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  multi_thread_memcpy_with_interleaved_affinity(dst, src, len*sizeof(float), k);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  /* check correctness (with "warmup" disabled) */
  assert( memcmp(src, dst, len*sizeof(float)) == 0 );

  float delta_us = (end.tv_sec - start.tv_sec) * 1.0e6 +
                     (end.tv_nsec - start.tv_nsec) * 1.0e-3;
  printf("[%s]\tThe throughput is %.2f Gbps.\n",
          command, len*sizeof(float)*8 / (delta_us*1000.0) );
  
  /* free `*dst` and `*src` */
  //TODO: (Bonus) Your code here.

  return 0;
}
#endif

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr,
            "Error: The program accepts exact 2 intergers.\n The first is the "
            "vector size and the second is the number of threads.\n");
    exit(1);
  }
  const int len = atoi(argv[1]);
  const int k = atoi(argv[2]);
  if (len < 0 || k < 1) {
    fprintf(stderr, "Error: invalid arguments.\n");
    exit(1);
  }
  // printf("Vector size=%d\tthreads len=%d.\n", len, k);

  /* C library's memcpy (1 byte) */
  execute(C_MEMCPY, len, k);
  /* single-threaded memcpy (4 bytes) */
  execute(SINGLE_THREAD, len, k);
  /* multi-threaded memcpy */
  execute(MULTI_THREAD, len, k);
  /* multi-threaded memcpy with affinity set */
  execute(MULTI_AFFINITY, len, k);

#ifdef BUILD_BONUS
  /* Bonus: multi-threaded memcpy with local NUMA memory policy */
  execute_numa(MEM_LOCAL, len, k);
  /* Bonus: multi-threaded memcpy with interleaved NUMA memory policy */
  execute_numa(MEM_INTER, len, k);
#endif

  return 0;
}
