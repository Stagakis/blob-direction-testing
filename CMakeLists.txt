cmake_minimum_required(VERSION 3.0)
project(Blob_direction)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
message("Opencv include dirs variable: ")
message("${OpenCV_INCLUDE_DIRS}")

message("Opencv cmake dir path: ")
message("${OpenCV_DIR}")

message("Opencv include dirs variable: ")
message("${OpenCV_INCLUDE_DIRS}")

message("Opencv libs variable: ")
message("${OpenCV_LIBS}")


include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ./include/ )



add_executable(Blob_direction src/main.cpp include/ORBextractor.h src/ORBextractor.cc src/helpers.cpp include/helpers.h) 

target_link_libraries( Blob_direction ${OpenCV_LIBS} )

message("Cmake prefix path: ")
message("${CMAKE_SYSTEM_PREFIX_PATH}")

