add_library(libcvs)
target_sources(libcvs
    PUBLIC
        cvs.h
        daltonlens.h
        daltonlens_cl.h
        daltonlens_omp.h
    PRIVATE
        daltonlens.cpp
        daltonlens_cl.cpp
        daltonlens_omp.cpp
        kernel.cl
)
target_include_directories(libcvs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libcvs PUBLIC OpenMP::OpenMP_CXX OpenCL::OpenCL)
