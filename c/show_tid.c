/**
 * References:
 * 1. `gettid`: https://man7.org/linux/man-pages/man2/gettid.2.html#VERSIONS
 * 2. POSIX thread: https://en.wikipedia.org/wiki/POSIX_Threads
 * 3. "Threads in Linux are LWP": https://stackoverflow.com/questions/28476456/threads-and-lwp-in-linux
 * (For Linux kernel , there is no concept of thread. Each thread is viewed by kernel as a separate process)
*/
#include <pthread.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

#define gettidv1() syscall(__NR_gettid) // new form
#define gettidv2() syscall(SYS_gettid)  // traditional form

void *entry_point(void *arg)
{
  char *idx_str = (char *) arg;

  printf("the pthread_%s id is %ld\n", idx_str, pthread_self());
  printf("the thread_%s's PID is %d\n", idx_str, getpid());
  printf("The LWP (TID) of thread_%s is: %d\n", idx_str, (unsigned int)gettidv1());
  printf("\n");

  return NULL;
}

int main(int argc, char *argv[]) {
  int ret = 0;
  pthread_t p1, p2;

  printf("The main thread's pthread id is %ld\n", pthread_self());
  printf("The main thread's PID is %d\n", getpid());
  printf("The LWP of main thread is: %d\n", (unsigned int)gettidv1());
  printf("\n");

  // create two threads
  if ( (ret=pthread_create(&p1, NULL, entry_point, "1")) != 0 )
  {
    fprintf(stderr, "pthread_create failed for thread_1\n");
    return ret;
  }
  if ( (ret=pthread_create(&p2, NULL, entry_point, "2")) != 0 )
  {
    fprintf(stderr, "pthread_create failed for thread_2\n");
    return ret;
  }

  // join to terminate
  pthread_join(p1, NULL);
  pthread_join(p2, NULL);

  return ret;
}
