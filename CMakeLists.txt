cmake_minimum_required(VERSION 3.5)
project(detect_test)
set(CMAKE_CXX_STANDARD 11)
include_directories(
    #OpenVINO推理引擎的头文件
    /home/sms/intel/openvino_2021/deployment_tools/inference_engine/include/
    /home/sms/intel/openvino_2021/deployment_tools/ngraph/include/
    .
)
#查找必要的依赖包
find_package(OpenCV REQUIRED)
find_package(InferenceEngine)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -pthread -lstdc++")

# 编译detector.so
add_library (detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/detector.cpp)
target_include_directories(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
)
target_link_libraries(detector
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
)

add_executable(detect_test main.cpp)
target_link_libraries(detect_test
        ${OpenCV_LIBS}
        detector
)


#("----------------------FMT----------------------")
#fmt
find_package(fmt REQUIRED)
target_link_libraries(detect_test
                        fmt::fmt)