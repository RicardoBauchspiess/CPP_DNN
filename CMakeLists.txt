cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpptorch)

set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cpptorch cifardata.cpp DataSet.hpp DataSet.cpp cifar10.h cifar10.cpp SubsetRandomSampler.hpp SubsetRandomSampler.cpp dataaugmentation.hpp dataaugmentation.cpp )
target_link_libraries(cpptorch "${TORCH_LIBRARIES}")
target_link_libraries( cpptorch ${OpenCV_LIBS}) # link OpenCV libraries , hightgui.so not found by cmake so this hack
MESSAGE("OpenCV_LIBS: " ${OpenCV_LIBS} )  #display opencv libs found
set_property(TARGET cpptorch PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET cpptorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:cpptorch>)
endif (MSVC)

