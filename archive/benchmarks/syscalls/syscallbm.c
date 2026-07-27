#include <bits/time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#define RND 25
#define NANO 1000000000

void timespec_diff(struct timespec *t_diff, const struct timespec *t1,
                   const struct timespec *t2) {
  t_diff->tv_nsec = t2->tv_nsec - t1->tv_nsec;
  bool borrow = t_diff->tv_nsec < 0;
  t_diff->tv_sec = t2->tv_sec - t1->tv_sec;
  if (borrow) {
    t_diff->tv_sec -= 1;
    t_diff->tv_nsec += NANO;
  }
}

int main(int argc, char *argv[]) {

  struct timespec START_TIME;
  struct timespec END_TIME;

  int iters = atoi(argv[1]);
  for (int i = 0; i < RND; i++) {
    pid_t getpid(void);
  }

  clock_gettime(CLOCK_MONOTONIC, &START_TIME);

  for (int i = 0; i < iters; i++) {

    pid_t getpid(void);
  }

  clock_gettime(CLOCK_MONOTONIC, &END_TIME);

  struct timespec *difference = malloc(sizeof(struct timespec));

  if (difference) {
    timespec_diff(difference, &START_TIME, &END_TIME);
  } else {
    printf("Malloc failed!");
  }

  long long ret_nano = difference->tv_nsec + (difference->tv_sec * NANO);

  printf("The time taken for %d syscalls is: %lld", iters, ret_nano);

  return 0;
}
