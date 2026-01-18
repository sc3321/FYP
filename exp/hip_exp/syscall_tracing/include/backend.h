#pragma once
#include <cstddef>
#include <stdlib.h>
#include <cstdio>


struct Backend {
  virtual ~Backend() = default;

  virtual const char* name() const = 0;
  virtual void* dev_malloc(std::size_t bytes) = 0;
  virtual void  dev_free(void* p) = 0;
  virtual void  h2d(void* d, const void* h, std::size_t bytes) = 0;
  virtual void  add_one(void* d, int N) = 0;
  virtual void  d2h(void* h, const void* d, std::size_t bytes) = 0;
  virtual void  sync() = 0;
};

