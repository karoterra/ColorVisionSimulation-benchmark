#pragma once

#include <cstdint>

#ifndef CL_KERNEL_SOURCE
#define CL_KERNEL_SOURCE(x) #x
#endif  // CL_KERNEL_SOURCE

namespace cvs {

struct BGRA {
  uint8_t b;
  uint8_t g;
  uint8_t r;
  uint8_t a;
};

enum class Deficiency {
  Protan,
  Deutan,
  Tritan,
};

};  // namespace cvs
