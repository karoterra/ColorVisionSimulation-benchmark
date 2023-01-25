#pragma once

#include <CL/cl.hpp>
#include <string>

#include "cvs.h"

namespace cvs::daltonlens_cl {

const std::string kernel_source =
#include "kernel.cl"
    ;

class Simulator {
 public:
  Simulator(cl::Context& context, cl::CommandQueue& queue)
      : context(context), queue(queue), program(context, kernel_source, true) {
    brettel1997 = cl::Kernel(program, "Brettel1997");
    vienot1999 = cl::Kernel(program, "Vienot1999");
  }

  void Brettel1997(Deficiency deficiency, float severity, const BGRA* src,
                   BGRA* dst, size_t len);
  void Vienot1999(Deficiency deficiency, float severity, const BGRA* src,
                  BGRA* dst, size_t len);

 private:
  cl::Context& context;
  cl::CommandQueue& queue;
  cl::Program program;

  cl::Kernel brettel1997;
  cl::Kernel vienot1999;
};

};  // namespace cvs::daltonlens_cl
