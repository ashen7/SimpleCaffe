#开启cuda支持选项
project(SimpleCaffe LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.5.1)

#这也是开启cuda支持的方法
if (CUDA_ENABLE)
    enable_language(CUDA)
endif()

# caffe宏定义开关
set(DEBUG OFF)
set(GOOGLE_TEST ON)
set(CPU_ONLY OFF)
set(USE_CUDNN ON)
set(USE_OPENCV ON)

#设置模式
if (${DEBUG} STREQUAL "ON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()

if (${CPU_ONLY} STREQUAL "ON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPU_ONLY")
else()
    find_package(CUDA REQUIRED)
endif()

#g++ 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -g -O3 -W -Wall -Wfatal-errors")
#nvcc 编译选项
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-O3;-gencode arch=compute_61,code=sm_61;-std=c++11;)

message("current caffe mode(CPU : GPU): ${caffe_mode}")
message("cudnn: ${USE_CUDNN}")
message("opencv: ${USE_OPENCV}")
message("g++ compile flags: ${CMAKE_CXX_FLAGS}")
message("nvcc compile flags: ${CUDA_NVCC_FLAGS}")

#根目录
set(CAFFE_ROOT_DIR .)
set(CUDA_ROOT_DIR /usr/local/cuda)
set(THIRD_PARTY_LIBRARY /home/yipeng/thirdlib)

#caffe源码目录
set(CAFFE_SRC_DIR ${CAFFE_ROOT_DIR}/src/caffe)
#caffe头文件目录
set(CAFFE_INC_DIR ${CAFFE_ROOT_DIR}/src)

#源文件
set(MY_CAFFE_SOURCE ${CAFFE_SRC_DIR}/proto/caffe.pb.cc
                    ${CAFFE_SRC_DIR}/common.cpp
                    ${CAFFE_SRC_DIR}/util/math_functions.cpp
                    ${CAFFE_SRC_DIR}/synced_memory.cpp
                    ${CAFFE_SRC_DIR}/tensor.cpp)

if (${CPU_ONLY} STREQUAL "OFF")
    set(MY_CAFFE_SOURCE ${MY_CAFFE_SOURCE}
                        ${CAFFE_SRC_DIR}/util/math_functions.cu)
    set(LINK_LIBRARY ${LINK_LIBRARY} cudart cublas curand)
endif()

if (${GOOGLE_TEST} STREQUAL "ON")
    set(MY_CAFFE_SOURCE ${MY_CAFFE_SOURCE}
                        ${CAFFE_ROOT_DIR}/src/gtest/gtest_main.cpp
                        ${CAFFE_SRC_DIR}/test/test_common.cpp
                        ${CAFFE_SRC_DIR}/test/test_tensor.cpp src/caffe/test/test_filler.cpp src/caffe/layer.cpp src/caffe/layer.hpp)
    set(LINK_LIBRARY ${LINK_LIBRARY} gtest)
endif()

#头文件路径
set(GTEST_INC ${THIRD_PARTY_LIBRARY}/gtest/include)
set(GLOG_INC ${THIRD_PARTY_LIBRARY}/glog/include)
set(GFLAGS_INC ${THIRD_PARTY_LIBRARY}/gflags/include)
set(OPENCV_INC ${THIRD_PARTY_LIBRARY}/opencv/include)
set(OPENBLAS_INC ${THIRD_PARTY_LIBRARY}/openblas/include)
set(PROTOBUF_INC ${THIRD_PARTY_LIBRARY}/protobuf/include)
set(CUDA_INC ${CUDA_ROOT_DIR}/include)

#库文件路径
set(GTEST_LIB ${THIRD_PARTY_LIBRARY}/gtest/lib)
set(GLOG_LIB ${THIRD_PARTY_LIBRARY}/glog/lib)
set(GFLAGS_LIB ${THIRD_PARTY_LIBRARY}/gflags/lib)
set(OPENCV_LIB ${THIRD_PARTY_LIBRARY}/opencv/lib)
set(OPENBLAS_LIB ${THIRD_PARTY_LIBRARY}/openblas/lib)
set(PROTOBUF_LIB ${THIRD_PARTY_LIBRARY}/protobuf/lib)
set(CUDA_LIB ${CUDA_ROOT_DIR}/lib64)

#链接库
set(LINK_LIBRARY ${LINK_LIBRARY} glog gflags opencv_core
                 opencv_highgui opencv_imgproc opencv_imgcodecs
                 opencv_videoio openblas protobuf pthread)

#头文件路径添加
include_directories(${CAFFE_INC_DIR}
                    ${GTEST_INC}
                    ${GLOG_INC}
                    ${GFLAGS_INC}
                    ${OPENCV_INC}
                    ${OPENBLAS_INC}
                    ${PROTOBUF_INC}
                    ${CUDA_INC})

#库文件路径添加
link_directories(${GTEST_LIB}
                 ${GLOG_LIB}
                 ${GFLAGS_LIB}
                 ${OPENCV_LIB}
                 ${OPENBLAS_LIB}
                 ${PROTOBUF_LIB}
                 ${CUDA_LIB})

#g++编译链接host端 nvcc编译链接device端 生成可执行文件
add_executable(simple_caffe.bin ${MY_CAFFE_SOURCE})
#cuda_add_executable(my_caffe.bin ${my_caffe_source})
target_link_libraries(simple_caffe.bin ${LINK_LIBRARY})
