#define _GNU_SOURCE /* To get pthread_getattr_np() declaration */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

// static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void print_error_msg(const char *idx_str, const char *msg, int ret)
{
  switch (ret)
  {  
  case EAGAIN:  // `pthread_create`: system-imposed limit, or insufficient resources.
    fprintf(stderr, "[%s] %s: EAGAIN\n", idx_str, msg);
    break;
  case EPERM:   // `pthread_create`: no permission to set params in `attr`.
    fprintf(stderr, "[%s] %s: EPERM\n", idx_str, msg);
    break;
  case EDEADLK: // `pthread_join`: deadlock detected.
    fprintf(stderr, "[%s] %s: EDEADLK\n", idx_str, msg);
    break;
  case EINVAL:  // `pthread_create` or `pthread_join` or `pthread_attr_getstack`: invalid value.
    fprintf(stderr, "[%s] %s: EINVAL\n", idx_str, msg);
    break;
  case ESRCH:   // `pthread_join`: no thread id found.
    fprintf(stderr, "[%s] %s: ESRCH\n", idx_str, msg);
    break;
  case ENOMEM:  // `pthread_getattr_np`: insufficient memory.
    fprintf(stderr, "[%s] %s: ENOMEM\n", idx_str, msg);
    break;
  default:
    fprintf(stderr, "[%s] %s: %d\n", idx_str, msg, ret);
    break;
  }
}

void *entry_point(void *arg) {
  int ret = 0;
  pthread_attr_t attr;
  size_t stack_size, guard_size;
  void *stack_addr;

  char *idx_str = (char *) arg;
  pthread_t thread = pthread_self();

  // try to get the attribute of the current thread.
  if ( (ret=pthread_getattr_np(thread, &attr)) != 0 )
  {
    print_error_msg(idx_str, "pthread_getattr_np failed", ret);
    return NULL;
  }

  // get the guard size (unit: byte)
  if ( (ret=pthread_attr_getguardsize(&attr, &guard_size)) != 0 )
  {
    print_error_msg(idx_str, "pthread_attr_getguardsize failed", ret);
  }
  // get the stack attribute values
  if ( (ret=pthread_attr_getstack(&attr, &stack_addr, &stack_size)) !=0 )
  {
    print_error_msg(idx_str, "pthread_attr_getstack failed", ret);
  }

  // print the stack information
  printf("Thread %s (id=%lu) stack:\n", idx_str, thread);
  printf("\t start address\t= %p\n", stack_addr);
  printf("\t end address  \t= %p\n", stack_addr + stack_size);
  printf("\t stack size   \t= %.2f MB\n", stack_size / 1024.0 / 1024.0);
  printf("\t guard size   \t= %lu B\n", guard_size);

  return NULL;
}

int main(int argc, char *argv[]) {
  int ret = 0;
  pthread_t p1, p2;

  // run the routine in main thread
  entry_point("main");

  // run the routine in thread p1
  if ( (ret=pthread_create(&p1, NULL, entry_point, "p1")) != 0 )
  {
    print_error_msg("p1", "pthread_create failed", ret);
    return ret;
  }
  if ( (ret=pthread_join(p1, NULL)) != 0 )
  {
    print_error_msg("p1", "pthread_join failed", ret);
  }
  
  // run the routine in thread p2
  if ( (ret=pthread_create(&p2, NULL, entry_point, "p2")) != 0 )
  {
    print_error_msg("p2", "pthread_create failed", ret);
    return ret;
  }
  if ( (ret=pthread_join(p2, NULL)) != 0 )
  {
    print_error_msg("p2", "pthread_join failed", ret);
  }

  return ret;
}
