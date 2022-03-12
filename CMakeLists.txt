cmake_minimum_required(VERSION 3.5)
project(detect_test)
set(CMAKE_CXX_STANDARD 11)
include_directories(
    #OpenVINO推理引擎的头文件
    /opt/intel/openvino_2021/deployment_tools/inference_engine/include/
    /opt/intel/openvino_2021/deployment_tools/ngraph/include/
)

#查找必要的依赖包
find_package(OpenCV REQUIRED)
set(InferenceEngine_DIR "/opt/intel/openvino_2021/deployment_tools/inference_engine/share")
find_package(InferenceEngine)
set(ngraph_DIR "/opt/intel/openvino_2021/deployment_tools/ngraph")
find_package(ngraph REQUIRED)
set(ngraph_LIBRARIES "/opt/intel/openvino_2021/deployment_tools/ngraph/lib/libngraph.so")
set(ngraph_INCLUDE_DIRS "/opt/intel/openvino_2021/deployment_tools/ngraph/include/")


# 编译detector.so

add_library (detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/detector.cpp)
target_include_directories(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
        PUBLIC ${ngraph_INCLUDE_DIRS}
)
target_link_libraries(detector
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)

add_executable(detect_test main.cpp)
target_link_libraries(detect_test
                        PRIVATE 
                        ${InferenceEngine_LIBRARIES}
                        ${OpenCV_LIBS} 
                        ${NGRAPH_LIBRARIES}
                        detector
                        )


# cmake_minimum_required(VERSION 3.10.0)
# project(main)
# find_package(ngraph REQUIRED)
# find_package(InferenceEngine REQUIRED)
# find_package(OpenCV REQUIRED)
# add_executable(${PROJECT_NAME} main.cpp)
# target_link_libraries(${PROJECT_NAME} 
#                         PRIVATE 
#                         ${InferenceEngine_LIBRARIES}
#                         ${OpenCV_LIBS} 
#                         ${NGRAPH_LIBRARIES}
#                         )