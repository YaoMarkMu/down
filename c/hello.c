#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


void *entry_point(void *arg) {
  printf("Hello from thread %d\n", *(int *)arg);
  return NULL;
}

int main(int argc, char *argv[]) {
  int ret;
  const int N = 8;
  pthread_t p[N];
  int args[N];

  printf("main: begin\n");

  // create N threads
  for (int i = 0; i < N; ++i) {
    args[i] = i;
    if ( (ret = pthread_create(&p[i], NULL, entry_point, &args[i])) != 0 ) {
      fprintf(stderr, "error: creation failed. ret=%d\n", ret);
      exit(1);
    }
  }

  // join waits for the threads to finish
  for (int i = 0; i < N; ++i) {
    if ( (ret = pthread_join(p[i], NULL)) != 0 ) {
      fprintf(stderr, "error: join failed. ret=%d\n", ret);
      exit(1);
    }
  }

  printf("main: end\n");
  return 0;
}
