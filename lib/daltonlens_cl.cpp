#include "daltonlens_cl.h"

struct Brettel1997Params {
  float mat1[9];
  float mat2[9];
  float normal[3];
};

static Brettel1997Params brettel_protan_params = {
  .mat1 = {
     1.00156, -0.00540,  0.00384,
     0.04372,  0.84864,  0.10764,
    -0.34528,  1.19548,  0.14980,
  },
  .mat2 = {
     1.00139, -0.00524,  0.00386,
     0.03892,  0.85291,  0.10816,
    -0.30742,  1.16172,  0.14570,
  },
  .normal = { -0.00441, 0.00393, 0.00048 },
};
static Brettel1997Params brettel_deutan_params = {
  .mat1 = {
     0.99278,  0.02728, -0.02006,
     0.09462,  0.64245,  0.26294,
    -0.22858,  0.86381,  0.36477,
  },
  .mat2 = {
     0.99196,  0.02784, -0.01980,
     0.10540,  0.63506,  0.25954,
    -0.25464,  0.88166,  0.37298,
  },
  .normal = { 0.00892, -0.00611, -0.00281 },
};
static Brettel1997Params brettel_tritan_params = {
  .mat1 = {
     0.11911,  0.80500,  0.07589,
     0.14431,  0.86812, -0.01243,
    -0.14826,  0.13548,  1.01277,
  },
  .mat2 = {
     0.24796,  1.12767, -0.37562,
     0.12320,  0.81526,  0.06154,
    -0.12657,  0.18979,  0.93678,
  },
  .normal = { -0.01113, -0.02788, 0.03901 },
};

void cvs::daltonlens_cl::Simulator::Brettel1997(Deficiency deficiency,
                                                float severity, const BGRA* src,
                                                BGRA* dst, size_t len) {
  const Brettel1997Params* params = nullptr;
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

  cl::Buffer buf_src(context, CL_MEM_READ_ONLY, len * sizeof(cvs::BGRA));
  cl::Buffer buf_dst(context, CL_MEM_WRITE_ONLY, len * sizeof(cvs::BGRA));
  cl::Buffer buf_params(context, CL_MEM_READ_ONLY, sizeof(Brettel1997Params));

  queue.enqueueWriteBuffer(buf_src, CL_TRUE, 0, sizeof(cvs::BGRA) * len, src);
  // queue.enqueueWriteBuffer(buf_dst, CL_TRUE, 0, sizeof(cvs::BGRA) * len,
  // dst);
  queue.enqueueWriteBuffer(buf_params, CL_TRUE, 0, sizeof(Brettel1997Params),
                           params);

  brettel1997.setArg(0, buf_src);
  brettel1997.setArg(1, buf_dst);
  brettel1997.setArg(2, buf_params);
  brettel1997.setArg(3, severity);

  queue.enqueueNDRangeKernel(brettel1997, cl::NullRange, cl::NDRange(len));

  queue.finish();

  queue.enqueueReadBuffer(buf_dst, CL_TRUE, 0, sizeof(cvs::BGRA) * len, dst);
}

static float vienot_protan_mat[3][3] = {
  { 1.00000, -0.00401, 0.00401 },
  { -0.00000, 0.88762, 0.11238 },
  { 0.00000, 0.88762, 0.11238 },
};

static float vienot_deutan_mat[3][3] = {
  { 1.00000, 0.02234, -0.02234 },
  { -0.00000, 0.70725, 0.29275 },
  { 0.00000, 0.70725, 0.29275 },
};

static float vienot_tritan_mat[3][3] = {
  { 0.14076, 0.85924, -0.00000 },
  { 0.14076, 0.85924, 0.00000 },
  { -0.14461, 0.14461, 1.00000 },
};

void cvs::daltonlens_cl::Simulator::Vienot1999(Deficiency deficiency,
                                               float severity, const BGRA* src,
                                               BGRA* dst, size_t len) {
  const float* mat = nullptr;
  switch (deficiency) {
    case Deficiency::Protan:
      mat = (float*)vienot_protan_mat;
      break;
    case Deficiency::Deutan:
      mat = (float*)vienot_deutan_mat;
      break;
    case Deficiency::Tritan:
      mat = (float*)vienot_tritan_mat;
      break;
  }

  cl::Buffer buf_src(context, CL_MEM_READ_ONLY, len * sizeof(cvs::BGRA));
  cl::Buffer buf_dst(context, CL_MEM_WRITE_ONLY, len * sizeof(cvs::BGRA));
  cl::Buffer buf_mat(context, CL_MEM_READ_ONLY, sizeof(vienot_protan_mat));

  queue.enqueueWriteBuffer(buf_src, CL_TRUE, 0, sizeof(cvs::BGRA) * len, src);
  queue.enqueueWriteBuffer(buf_mat, CL_TRUE, 0, sizeof(vienot_protan_mat), mat);

  vienot1999.setArg(0, buf_src);
  vienot1999.setArg(1, buf_dst);
  vienot1999.setArg(2, buf_mat);
  vienot1999.setArg(3, severity);

  queue.enqueueNDRangeKernel(vienot1999, cl::NullRange, cl::NDRange(len));

  queue.finish();

  queue.enqueueReadBuffer(buf_dst, CL_TRUE, 0, sizeof(cvs::BGRA) * len, dst);
}
