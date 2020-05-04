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

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS}
					${PROJECT_SOURCE_DIR}
					${PROJECT_SOURCE_DIR}/include
					${EIGEN3_INCLUDE_DIR}
					${Pangolin_INCLUDE_DIRS}
 )
include_directories( ./include/ )
include_directories( . )

file( GLOB SRCS src/*.cpp src/*.cc )

add_executable(Blob_direction 
	${SRCS}
	#src/main.cpp 
	#src/ORBextractor.cc 
	#src/helpers.cpp  
	#src/ThreeFrameProcesser.cpp
	#src/BlobExtractor.cpp
	#src/ORBmatcher.cc
#include/MapPoint.h
#include/KeyFrame.h
#include/Frame.h

include/helpers.h
include/ORBextractor.h
include/ThreeFrameProcesser.h
include/BlobExtractor.h
) 

target_link_libraries( Blob_direction 
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

message("Cmake prefix path: ")
message("${CMAKE_SYSTEM_PREFIX_PATH}")

