cmake_minimum_required(VERSION 2.8)
project(VisualOdometry)

find_package(OpenCV 4.0 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
)

file(GLOB src
    "src/*.h"
    "src/*.cpp"
)

add_executable( ${PROJECT_NAME} ${src} )

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
)
