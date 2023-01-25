#include "daltonlens_omp.h"

#include <omp.h>

#include <cmath>

static float linearRGB_from_sRGB(uint8_t v) {
  float fv = v / 255.f;
  if (fv < 0.04045f) return fv / 12.92f;
  return pow((fv + 0.055f) / 1.055f, 2.4f);
}

static uint8_t sRGB_from_linearRGB(float v) {
  if (v <= 0.f) return 0;
  if (v >= 1.f) return 255;
  if (v < 0.0031308f) return 0.5f + (v * 12.92f * 255.f);
  return 0.f + 255.f * (powf(v, 1.f / 2.4f) * 1.055f - 0.055f);
}

struct Brettel1997Params {
  float mat1[9];
  float mat2[9];
  float normal[3];
};

static Brettel1997Params brettel_protan_params = {
  .mat1 = {
    0.14980, 1.19548, -0.34528,
    0.10764, 0.84864, 0.04372,
    0.00384, -0.00540, 1.00156,
  },
  .mat2 = {
    0.14570, 1.16172, -0.30742,
    0.10816, 0.85291, 0.03892,
    0.00386, -0.00524, 1.00139,
  },
  .normal = { 0.00048, 0.00393, -0.00441 },
};
static Brettel1997Params brettel_deutan_params = {
  .mat1 = {
    0.36477, 0.86381, -0.22858,
    0.26294, 0.64245, 0.09462,
    -0.02006, 0.02728, 0.99278,
  },
  .mat2 = {
    0.37298, 0.88166, -0.25464,
    0.25954, 0.63506, 0.10540,
    -0.01980, 0.02784, 0.99196,
  },
  .normal = { -0.00281, -0.00611, 0.00892 },
};
static Brettel1997Params brettel_tritan_params = {
  .mat1 = {
    1.01277, 0.13548, -0.14826,
    -0.01243, 0.86812, 0.14431,
    0.07589, 0.80500, 0.11911,
  },
  .mat2 = {
    0.93678, 0.18979, -0.12657,
    0.06154, 0.81526, 0.12320,
    -0.37562, 1.12767, 0.24796,
  },
  .normal = { 0.03901, -0.02788, -0.01113 },
};

void cvs::daltonlens_omp::SimulateBrettel1997(Deficiency deficiency,
                                              float severity, const BGRA *src,
                                              BGRA *dst, int len) {
  const Brettel1997Params *params = nullptr;
  switch (deficiency) {
    case Deficiency::Protan:
      params = &brettel_protan_params;
      break;
    case Deficiency::Deutan:
      params = &brettel_deutan_params;
      break;
    case Deficiency::Tritan:
      params = &brettel_tritan_params;
      break;
  }

#pragma omp parallel for
  for (int i = 0; i < len; i++) {
    const float rgb[3] = {
      linearRGB_from_sRGB(src[i].r),
      linearRGB_from_sRGB(src[i].g),
      linearRGB_from_sRGB(src[i].b),
    };

    const float *n = params->normal;
    const float dot = rgb[0] * n[0] + rgb[1] * n[1] + rgb[2] * n[2];
    const float *mat = dot >= 0 ? params->mat1 : params->mat2;

    float rgb_cvd[3] = {
      mat[0] * rgb[0] + mat[1] * rgb[1] + mat[2] * rgb[2],
      mat[3] * rgb[0] + mat[4] * rgb[1] + mat[5] * rgb[2],
      mat[6] * rgb[0] + mat[7] * rgb[1] + mat[8] * rgb[2],
    };

    rgb_cvd[0] = rgb_cvd[0] * severity + rgb[0] * (1.f - severity);
    rgb_cvd[1] = rgb_cvd[1] * severity + rgb[1] * (1.f - severity);
    rgb_cvd[2] = rgb_cvd[2] * severity + rgb[2] * (1.f - severity);

    dst[i].r = sRGB_from_linearRGB(rgb_cvd[0]);
    dst[i].g = sRGB_from_linearRGB(rgb_cvd[1]);
    dst[i].b = sRGB_from_linearRGB(rgb_cvd[2]);
    dst[i].a = src[i].a;
  }
}

static float vienot_protan_mat[] = {
  0.11238,  0.88762, 0.00000,  0.11238, 0.88762,
  -0.00000, 0.00401, -0.00401, 1.00000,
};

static float vienot_deutan_mat[] = {
  0.29275,  0.70725,  0.00000, 0.29275, 0.70725,
  -0.00000, -0.02234, 0.02234, 1.00000,
};

static float vienot_tritan_mat[] = {
  1.00000, 0.14461,  -0.14461, 0.00000, 0.85924,
  0.14076, -0.00000, 0.85924,  0.14076,
};

void cvs::daltonlens_omp::SimulateVienot1999(Deficiency deficiency,
                                             float severity, const BGRA *src,
                                             BGRA *dst, int len) {
  const float *mat = nullptr;
  switch (deficiency) {
    case Deficiency::Protan:
      mat = vienot_protan_mat;
      break;
    case Deficiency::Deutan:
      mat = vienot_deutan_mat;
      break;
    case Deficiency::Tritan:
      mat = vienot_tritan_mat;
      break;
  }

#pragma omp parallel for
  for (int i = 0; i < len; i++) {
    const float rgb[3] = {
      linearRGB_from_sRGB(src[i].r),
      linearRGB_from_sRGB(src[i].g),
      linearRGB_from_sRGB(src[i].b),
    };

    float rgb_cvd[3] = {
      mat[0] * rgb[0] + mat[1] * rgb[1] + mat[2] * rgb[2],
      mat[3] * rgb[0] + mat[4] * rgb[1] + mat[5] * rgb[2],
      mat[6] * rgb[0] + mat[7] * rgb[1] + mat[8] * rgb[2],
    };

    rgb_cvd[0] = rgb_cvd[0] * severity + rgb[0] * (1.f - severity);
    rgb_cvd[1] = rgb_cvd[1] * severity + rgb[1] * (1.f - severity);
    rgb_cvd[2] = rgb_cvd[2] * severity + rgb[2] * (1.f - severity);

    dst[i].r = sRGB_from_linearRGB(rgb_cvd[0]);
    dst[i].g = sRGB_from_linearRGB(rgb_cvd[1]);
    dst[i].b = sRGB_from_linearRGB(rgb_cvd[2]);
    dst[i].a = src[i].a;
  }
}
