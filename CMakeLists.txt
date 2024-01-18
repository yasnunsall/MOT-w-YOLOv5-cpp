cmake_minimum_required(VERSION 3.26)
project(multi_object_tracking)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(multi_object_tracking main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OpenCV_LIBS})