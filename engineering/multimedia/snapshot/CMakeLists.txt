project(snapsshot-example)
cmake_minimum_required(VERSION 2.8.9)
include(FindPkgConfig)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

pkg_search_module (GSTREAMER REQUIRED gstreamer-1.0)
pkg_search_module (GDKPIXBUF gdk-pixbuf-2.0 REQUIRED)

include_directories(${GSTREAMER_INCLUDE_DIRS} ${GDKPIXBUF_INCLUDE_DIRS})

set(snapshot_SOURCES snapshot.c)
add_executable(snapshot
    ${snapshot_SOURCES}
)
target_link_libraries(snapshot ${GDKPIXBUF_LIBRARIES} ${GSTREAMER_LIBRARIES})
