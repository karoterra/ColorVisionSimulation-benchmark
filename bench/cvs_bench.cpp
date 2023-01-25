#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "cvs.h"
#include "daltonlens.h"
#include "daltonlens_cl.h"
#include "daltonlens_omp.h"

using cvs::BGRA;
using cvs::Deficiency;

const int kMaxSize = 10'000'000;

#define BM_RANGE RangeMultiplier(10)->Range(10, kMaxSize)

class MyFixture : public benchmark::Fixture {
 public:
  std::vector<BGRA> src;
  std::vector<BGRA> dst;

  std::mt19937 mt;

  MyFixture() {
    src.resize(kMaxSize);
    dst.resize(kMaxSize);

    std::random_device rnd;
    mt.seed(rnd());

    for (size_t i = 0; i < kMaxSize; i++) {
      ((uint32_t*)(src.data()))[i] = mt();
    }
  }
};

class CLFixture : public MyFixture {
 public:
  cl::Context context;
  cl::CommandQueue queue;
  cvs::daltonlens_cl::Simulator sim;

  CLFixture()
      : context(CL_DEVICE_TYPE_DEFAULT), queue(context), sim(context, queue) {
    for (int i = 0; i < 10; i++) {
      sim.Vienot1999(Deficiency::Protan, 1.f, src.data(), dst.data(), kMaxSize);
    }
  }
};

void copy(BGRA* src, BGRA* dst, size_t len) {
  for (size_t i = 0; i < len; i++) {
    dst[i] = src[i];
  }
}

BENCHMARK_DEFINE_F(MyFixture, Copy)(benchmark::State& st) {
  size_t size = st.range(0);
  for (auto _ : st) {
    copy(src.data(), dst.data(), size);
  }
}
BENCHMARK_REGISTER_F(MyFixture, Copy)->BM_RANGE;

BENCHMARK_DEFINE_F(MyFixture, DaltonLensBrettel1997)(benchmark::State& st) {
  size_t size = st.range(0);
  for (auto _ : st) {
    cvs::daltonlens::SimulateBrettel1997(Deficiency::Protan, 1.f, src.data(),
                                         dst.data(), size);
  }
}
BENCHMARK_REGISTER_F(MyFixture, DaltonLensBrettel1997)->BM_RANGE;

BENCHMARK_DEFINE_F(MyFixture, DaltonLensVienot1999)(benchmark::State& st) {
  size_t size = st.range(0);
  for (auto _ : st) {
    cvs::daltonlens::SimulateVienot1999(Deficiency::Protan, 1.f, src.data(),
                                        dst.data(), size);
  }
}
BENCHMARK_REGISTER_F(MyFixture, DaltonLensVienot1999)->BM_RANGE;

BENCHMARK_DEFINE_F(MyFixture, DaltonLensOMPBrettel1997)(benchmark::State& st) {
  int size = st.range(0);
  for (auto _ : st) {
    cvs::daltonlens_omp::SimulateBrettel1997(Deficiency::Protan, 1.f,
                                             src.data(), dst.data(), size);
  }
}
BENCHMARK_REGISTER_F(MyFixture, DaltonLensOMPBrettel1997)->BM_RANGE;

BENCHMARK_DEFINE_F(MyFixture, DaltonLensOMPVienot1999)(benchmark::State& st) {
  int size = st.range(0);
  for (auto _ : st) {
    cvs::daltonlens_omp::SimulateVienot1999(Deficiency::Protan, 1.f, src.data(),
                                            dst.data(), size);
  }
}
BENCHMARK_REGISTER_F(MyFixture, DaltonLensOMPVienot1999)->BM_RANGE;

BENCHMARK_DEFINE_F(CLFixture, Brettel1997)(benchmark::State& st) {
  size_t size = st.range(0);
  for (auto _ : st) {
    sim.Brettel1997(Deficiency::Protan, 1.f, src.data(), dst.data(), size);
  }
}
BENCHMARK_REGISTER_F(CLFixture, Brettel1997)->BM_RANGE;

BENCHMARK_DEFINE_F(CLFixture, Vienot1999)(benchmark::State& st) {
  size_t size = st.range(0);
  for (auto _ : st) {
    sim.Vienot1999(Deficiency::Protan, 1.f, src.data(), dst.data(), size);
  }
}
BENCHMARK_REGISTER_F(CLFixture, Vienot1999)->BM_RANGE;

BENCHMARK_MAIN();
