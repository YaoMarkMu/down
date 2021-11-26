#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <sys/sysinfo.h>

int main(int argc, char *argv[]) {
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  if ( pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset) == 0 )
  {
    printf("Before setting: pthread_setaffinity_np() contained:\n");
    printf("CPU ");
    for (int j = 0; j < CPU_SETSIZE; j++)
      if (CPU_ISSET(j, &cpuset))
        printf("%d ", j);
    printf("\n");
  }

  printf("Set affinity mask to include CPUs (1, 3, 5, ...  2n+1)\n");
  CPU_ZERO(&cpuset);
  for (int j = 0; j < get_nprocs(); j++)
  {
    if (j % 2 == 1) {
      CPU_SET(j, &cpuset);
    }
  }
  if ( pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0 )
  {
    fprintf(stderr, "setaffinity failed.\n");
    return -1;
  }

  if ( pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset) == 0 )
  {
    printf("After setting: pthread_setaffinity_np() contained:\n");
    printf("CPU ");
    for (int j = 0; j < CPU_SETSIZE; j++)
    {
      if (CPU_ISSET(j, &cpuset)) {
        printf("%d ", j);
      }
    }
    printf("\n");
  }

  return 0;
}
