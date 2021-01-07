cmake_minimum_required(VERSION 3.8)

project(ExamProject)

if (WIN32)
	set (OpenCV_DIR $ENV{OpenCV})
endif ()

find_package(OpenCV REQUIRED)

include_directories(
	include
	${OpenCV_INCLUDE_DIRS}
	"include/utilities.h"
)

add_executable(
	${PROJECT_NAME}
	"src/main.cpp"
	"src/utilities.cpp"
)

target_link_libraries(
	${PROJECT_NAME}
	${OpenCV_LIBS}
)
