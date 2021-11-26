#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct _arg_t {
  int a;
  int b;
} arg_t;

typedef struct _ret_t {
  int x;
  int y;
} ret_t;

void *entry_point(void *arg) {
  arg_t *p_arg = (arg_t *)arg;
  printf("%d %d\n", p_arg->a, p_arg->b);

  /* Wrong Way */
  // ret_t r;
  // r.x = 1;
  // r.x = 2;
  // return (void *)&r;

  /* Right Way */
  ret_t *r = (ret_t *)malloc(sizeof(ret_t));
  r->x = 1;
  r->y = 2;
  return (void *)r;
}

int main(int argc, char *argv[]) {
  int ret;
  pthread_t p;
  ret_t *r;

  arg_t args;
  args.a = 10;
  args.b = 20;

  if ( (ret=pthread_create(&p, NULL, entry_point, &args)) != 0 )
  {
    fprintf(stderr, "pthread_created failed.");
    return ret;
  }
  if ( (ret=pthread_join(p, (void **)&r)) != 0 )
  {
    fprintf(stderr, "pthread_join failed.");
  }

  printf("returned %d %d\n", r->x, r->y);
  free(r);

  return 0;
}
