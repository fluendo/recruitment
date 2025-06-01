cmake_minimum_required(VERSION 3.16)
project(videoplayer-gtk)
include(FindPkgConfig)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

pkg_search_module (GSTREAMER REQUIRED gstreamer-1.0)
pkg_search_module (GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
pkg_search_module (GTK REQUIRED gtk+-3.0 )

message(STATUS "GSTREAMER_INCLUDE_DIRS = ${GSTREAMER_INCLUDE_DIRS}")
message(STATUS "GSTREAMER_LIBRARIES = ${GSTREAMER_LIBRARIES}")

include_directories(${GSTREAMER_INCLUDE_DIRS} ${GTK_INCLUDE_DIRS} ${GSTREAMER_VIDEO_INCLUDE_DIRS})

set(videoplayer_SOURCES videoplayer.c)
add_executable(videoplayer
    ${videoplayer_SOURCES}
)
target_link_libraries(videoplayer ${GTK_LIBRARIES} ${GSTREAMER_LIBRARIES} ${GSTREAMER_VIDEO_LIBRARIES})