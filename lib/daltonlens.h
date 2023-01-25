#pragma once

#include <cstdint>

#include "cvs.h"

namespace cvs::daltonlens {

void SimulateBrettel1997(Deficiency deficiency, float severity, const BGRA *src,
                         BGRA *dst, size_t len);

void SimulateVienot1999(Deficiency deficiency, float severity, const BGRA *src,
                        BGRA *dst, size_t len);

};  // namespace cvs::daltonlens
