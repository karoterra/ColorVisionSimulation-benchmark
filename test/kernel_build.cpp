#include <CL/cl.hpp>
#include <format>
#include <iostream>
#include <string>

#include "daltonlens_cl.h"

std::string BuildStatusToString(cl_build_status status) {
  switch (status) {
    case CL_BUILD_NONE:
      return "CL_BUILD_NONE";
    case CL_BUILD_ERROR:
      return "CL_BUILD_ERROR";
    case CL_BUILD_SUCCESS:
      return "CL_BUILD_SUCCESS";
    case CL_BUILD_IN_PROGRESS:
      return "CL_BUILD_IN_PROGRESS";
  }
  return "unknown";
}

std::string BuildResultToString(cl_int result) {
  switch (result) {
    case CL_SUCCESS:
      return "Success";
    case CL_INVALID_PROGRAM:
      return "Invalid program";
    case CL_INVALID_VALUE:
      return "Invalid value";
    case CL_INVALID_DEVICE:
      return "Invalid device";
    case CL_INVALID_BINARY:
      return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:
      return "Invalid build options";
    case CL_INVALID_OPERATION:
      return "Invalid operation";
    case CL_COMPILER_NOT_AVAILABLE:
      return "Compiler not available";
    case CL_BUILD_PROGRAM_FAILURE:
      return "Build program failure";
    case CL_OUT_OF_RESOURCES:
      return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:
      return "Out of host memory";
  }
  return "unknown";
}

int main(int argc, const char* argv[]) {
  cl::Device device = cl::Device::getDefault();
  cl::Context context(device);
  cl::Program program(context, cvs::daltonlens_cl::kernel_source);
  auto result = program.build("-Werror");
  auto status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
  std::cout << BuildResultToString(result) << std::endl;
  std::cout << "Status: " << BuildStatusToString(status) << std::endl;
  std::cout << "Options: "
            << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device)
            << std::endl;
  std::cout << "Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
            << std::endl;

  std::cout << "Source:\n" << cvs::daltonlens_cl::kernel_source << std::endl;

  return result != CL_SUCCESS;
}
