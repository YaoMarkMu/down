#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <unistd.h>

typedef struct _account_t {
    int balance;
    pthread_mutex_t m;
    int aid;
} account_t;

typedef struct _transfer_args_t {
    account_t* from;
    account_t* to;
    int amount;
} transfer_arg_t;

void* deadlock_transfer(void *arg) {
    transfer_arg_t* tran = (transfer_arg_t*)(arg);

    printf("%d -> %d transfer $%d\n", tran->from->aid, tran->to->aid, tran->amount);

    pthread_mutex_lock(&tran->from->m);
    sleep(1);
    pthread_mutex_lock(&tran->to->m);

    tran->from->balance -= tran->amount;
    tran->to->balance += tran->amount;

    pthread_mutex_unlock(&tran->to->m);
    pthread_mutex_unlock(&tran->from->m);
    return NULL;
}

void create_account(account_t* acct, int balance) {
    static int aid = 0;
    acct->balance = balance;
    acct->aid = ++aid;
    pthread_mutex_init(&acct->m, NULL);
}

int main(int argc, char* argv[]) {
    pthread_t p1, p2;

    /* create two accounts */
    account_t a1, a2;
    create_account(&a1, 1000);
    create_account(&a2, 1000);
    printf("main begin: a1's balance=%d, a2's balance=%d\n", a1.balance, a2.balance);

    /* create transfer arguments */
    transfer_arg_t arg1, arg2;
    //
    arg1.from = &a1;
    arg1.to = &a2;
    arg1.amount = 100;
    //
    arg2.from = &a2;
    arg2.to = &a1;
    arg2.amount = 100;

    /* apply the two transactions */
    if ( pthread_create(&p1, NULL, deadlock_transfer, &arg1) != 0 )
    {
        fprintf(stderr, "pthread_create failed.\n");
        exit(1);
    }
    if ( pthread_create(&p2, NULL, deadlock_transfer, &arg2) != 0 )
    {
        fprintf(stderr, "pthread_create failed.\n");
        exit(1);
    }
    assert( pthread_join(p1, NULL) == 0 );
    assert( pthread_join(p2, NULL) == 0 );

    printf("main end: a1's balance=%d, a2's balance=%d\n", a1.balance, a2.balance);

    pthread_mutex_destroy(&a1.m);
    pthread_mutex_destroy(&a2.m);

    return 0;
}
