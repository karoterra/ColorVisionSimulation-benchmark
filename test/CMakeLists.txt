add_executable(cvs_test cvs_test.cpp)
target_include_directories(cvs_test PRIVATE ${STB_INCLUDE_DIRS})
target_link_libraries(cvs_test PRIVATE libcvs)

add_test(NAME cvs_test
    COMMAND
        cvs_test
        ${PROJECT_SOURCE_DIR}/test/images
        ${PROJECT_SOURCE_DIR}/test_out
)

add_executable(kernel_build kernel_build.cpp)
target_link_libraries(kernel_build PRIVATE libcvs)

add_test(NAME kernel_build COMMAND kernel_build)
