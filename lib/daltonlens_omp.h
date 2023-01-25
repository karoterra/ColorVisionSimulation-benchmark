#pragma once

#include <cstdint>

#include "cvs.h"

namespace cvs::daltonlens_omp {

void SimulateBrettel1997(Deficiency deficiency, float severity, const BGRA *src,
                         BGRA *dst, int len);

void SimulateVienot1999(Deficiency deficiency, float severity, const BGRA *src,
                        BGRA *dst, int len);

};  // namespace cvs::daltonlens_omp
