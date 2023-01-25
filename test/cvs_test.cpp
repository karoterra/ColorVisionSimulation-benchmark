#include <cmath>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#include "cvs.h"
#include "daltonlens.h"
#include "daltonlens_cl.h"
#include "daltonlens_omp.h"

namespace fs = std::filesystem;

struct Image {
  int width;
  int height;
  std::vector<cvs::BGRA> pixels;

  Image(int w, int h) : width(w), height(h), pixels(w * h) {}
};

struct TestCase {
  cvs::Deficiency deficiency;
  float severity;
  std::string param_str;
};

const auto kTestCases = std::vector<TestCase>{
  { cvs::Deficiency::Protan, 1.f, "protan_1.0" },
  { cvs::Deficiency::Protan, 0.55f, "protan_0.55" },
  { cvs::Deficiency::Deutan, 1.f, "deutan_1.0" },
  { cvs::Deficiency::Deutan, 0.55f, "deutan_0.55" },
  { cvs::Deficiency::Tritan, 1.f, "tritan_1.0" },
  { cvs::Deficiency::Tritan, 0.55f, "tritan_0.55" },
};

Image load_image(const fs::path& p) {
  Image im(0, 0);
  int comp;
  auto raw = stbi_load(p.string().c_str(), &im.width, &im.height, &comp, 4);
  if (!raw) {
    return im;
  }

  im.pixels.resize(im.width * im.height);
  for (size_t i = 0; i < im.pixels.size(); i++) {
    im.pixels[i] = cvs::BGRA{
      raw[i * 4 + 2],
      raw[i * 4 + 1],
      raw[i * 4 + 0],
      raw[i * 4 + 3],
    };
  }

  stbi_image_free(raw);

  return im;
}

void write_image(const fs::path& p, const Image& im) {
  std::vector<stbi_uc> raw(im.width * im.height * 4);
  for (size_t i = 0; i < im.pixels.size(); i++) {
    raw[i * 4 + 0] = im.pixels[i].r;
    raw[i * 4 + 1] = im.pixels[i].g;
    raw[i * 4 + 2] = im.pixels[i].b;
    raw[i * 4 + 3] = im.pixels[i].a;
  }
  stbi_write_png(p.string().c_str(), im.width, im.height, 4, raw.data(), 0);
}

template <typename T>
T abs_diff(const T a, const T b) {
  if (a < b) return b - a;
  return a - b;
}

using SimFunc = std::function<void(const Image&, Image&, const TestCase&)>;
void test(const fs::path& input_dir, const fs::path& output_dir,
          const std::string& impl_name, const std::string& method_name,
          SimFunc simulate) {
  const auto im_input = load_image(input_dir / "input.png");
  Image im_simulated(im_input.width, im_input.height);
  const auto impl_dir = output_dir / impl_name;
  fs::create_directories(impl_dir);

  for (const auto& tc : kTestCases) {
    simulate(im_input, im_simulated, tc);
    const auto filename = std::format("{}_{}.png", method_name, tc.param_str);
    write_image(impl_dir / filename, im_simulated);

    const auto im_ref = load_image(input_dir / filename);
    Image im_abs(im_input.width, im_input.height);
    cvs::BGRA max_diff{ 0, 0, 0, 0 };
    for (size_t i = 0; i < im_input.pixels.size(); i++) {
      auto px_sim = im_simulated.pixels[i];
      auto px_ref = im_ref.pixels[i];
      auto px = cvs::BGRA{
        abs_diff(px_sim.b, px_ref.b),
        abs_diff(px_sim.g, px_ref.g),
        abs_diff(px_sim.r, px_ref.r),
        255,
      };
      im_abs.pixels[i] = px;
      max_diff = cvs::BGRA{
        std::max(px.b, max_diff.b),
        std::max(px.g, max_diff.g),
        std::max(px.r, max_diff.r),
        0,
      };
    }
    write_image(impl_dir / ("abs_" + filename), im_abs);
    std::cout << std::format(
                     "impl: {}, method: {}, param: {}, max diff: ({},{},{})",
                     impl_name, method_name, tc.param_str, max_diff.r,
                     max_diff.g, max_diff.b)
              << std::endl;
  }
}

int main(int argc, const char* argv[]) {
  if (argc <= 2) {
    std::cout << "Usage: cvs_test <input dir> <output dir>" << std::endl;
    return 1;
  }

  fs::path input_dir = argv[1];
  fs::path output_dir = argv[2];
  fs::create_directories(output_dir);

  // DaltonLens
  test(input_dir, output_dir, "daltonlens", "brettel1997",
       [](const Image& src, Image& dst, const TestCase& tc) {
         cvs::daltonlens::SimulateBrettel1997(
             tc.deficiency, tc.severity, src.pixels.data(), dst.pixels.data(),
             src.pixels.size());
       });

  test(input_dir, output_dir, "daltonlens", "vienot1999",
       [](const Image& src, Image& dst, const TestCase& tc) {
         cvs::daltonlens::SimulateVienot1999(
             tc.deficiency, tc.severity, src.pixels.data(), dst.pixels.data(),
             src.pixels.size());
       });

  // OpenMP
  test(input_dir, output_dir, "daltonlens_omp", "brettel1997",
       [](const Image& src, Image& dst, const TestCase& tc) {
         cvs::daltonlens_omp::SimulateBrettel1997(
             tc.deficiency, tc.severity, src.pixels.data(), dst.pixels.data(),
             src.pixels.size());
       });

  test(input_dir, output_dir, "daltonlens_omp", "vienot1999",
       [](const Image& src, Image& dst, const TestCase& tc) {
         cvs::daltonlens_omp::SimulateVienot1999(
             tc.deficiency, tc.severity, src.pixels.data(), dst.pixels.data(),
             src.pixels.size());
       });

  // OpenCL
  {
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);
    cl::CommandQueue queue(context);
    cvs::daltonlens_cl::Simulator sim(context, queue);

    test(input_dir, output_dir, "daltonlens_cl", "brettel1997",
         [&](const Image& src, Image& dst, const TestCase& tc) {
           sim.Brettel1997(tc.deficiency, tc.severity, src.pixels.data(),
                           dst.pixels.data(), src.pixels.size());
         });

    test(input_dir, output_dir, "daltonlens_cl", "vienot1999",
         [&](const Image& src, Image& dst, const TestCase& tc) {
           sim.Vienot1999(tc.deficiency, tc.severity, src.pixels.data(),
                          dst.pixels.data(), src.pixels.size());
         });
  }

  return 0;
}
