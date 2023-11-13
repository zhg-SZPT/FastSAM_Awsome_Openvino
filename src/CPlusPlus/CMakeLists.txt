cmake_minimum_required(VERSION 3.10)
project(FastSAM)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

enable_language(CXX)


# 设置这个openvino安装runtime目录
set(OpenVINO_DIR "/usr/local/runtime")

include_directories(${OpenVINO_DIR}/include)
link_directories(${OpenVINO_DIR}/lib/intel64)


# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



file(GLOB_RECURSE SRC_LIST "./*.cpp")	#遍历获取source/路径下所有的*.cpp和*.c文件列表


foreach(SRC ${SRC_LIST})
    message("src:" ${SRC})
endforeach()


add_executable(
    ${PROJECT_NAME}
    ${SRC_LIST}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    openvino
)