add_executable(cvs_bench cvs_bench.cpp)
target_link_libraries(cvs_bench
    PRIVATE
        libcvs
        benchmark::benchmark
        benchmark::benchmark_main
)
