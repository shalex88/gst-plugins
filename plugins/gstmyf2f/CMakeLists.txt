set(PROJECT_NAME gstmyf2f)
include(cmake/SetVersion.cmake)

if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE} AND DEFINED ENV{SDKTARGETSYSROOT})
    message(STATUS "Cross compilation")
    message(STATUS "Toolchain: $ENV{CMAKE_TOOLCHAIN_FILE}")
    message(STATUS "Sysroot: $ENV{SDKTARGETSYSROOT}")
else()
    message(STATUS "Native compilation")
    include(cmake/EnableVcpkg.cmake)
endif()

cmake_minimum_required(VERSION 3.22)
project(${PROJECT_NAME} LANGUAGES CXX VERSION ${VERSION})
add_compile_definitions(APP_NAME="${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_search_module(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_search_module(GST-BASE REQUIRED IMPORTED_TARGET gstreamer-base-1.0)
find_package(yaml-cpp CONFIG REQUIRED)

set(DEEPSTREAM_INCLUDE_DIRS "${CMAKE_SYSROOT}/opt/nvidia/deepstream/deepstream/sources/includes")
set(DEEPSTREAM_LIBRARY_DIRS "${CMAKE_SYSROOT}/opt/nvidia/deepstream/deepstream/lib")
if(NOT EXISTS "${DEEPSTREAM_INCLUDE_DIRS}" OR NOT EXISTS "${DEEPSTREAM_LIBRARY_DIRS}")
    message(FATAL_ERROR "DeepStream SDK is not found!")
endif()

set(VP_API_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../processing-libs/)

set(SOURCE_DIR "src")
set(CONFIG_DIR "config")

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "${SOURCE_DIR}/*.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${VP_API_INCLUDE_DIR}
    ${DEEPSTREAM_INCLUDE_DIRS}
#    ${CMAKE_SYSROOT}/usr/src/jetson_multimedia_api/include/
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${DEEPSTREAM_LIBRARY_DIRS}
#    ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/nvidia/
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    PkgConfig::GSTREAMER
    PkgConfig::GST-BASE
    yaml-cpp
    nvbufsurface
    ${YAML_CPP_TARGET}
)

include(GNUInstallDirs)
if(NOT DEFINED INSTALL_ROOT)
    set(INSTALL_ROOT "/tmp/" CACHE PATH "Root install dir for ${PROJECT_NAME}" FORCE)
endif()
set(CMAKE_INSTALL_PREFIX "${INSTALL_ROOT}" CACHE PATH "Install prefix" FORCE)
set(PROJECT_INSTALL_ROOT "${INSTALL_ROOT}/gst-plugins/plugins/${PROJECT_NAME}" CACHE PATH "Root install dir for ${PROJECT_NAME}")
set(PROJECT_INSTALL_BINDIR "${PROJECT_INSTALL_ROOT}")
set(PROJECT_INSTALL_LIBDIR "${PROJECT_INSTALL_ROOT}/${CMAKE_INSTALL_LIBDIR}")
set(PROJECT_INSTALL_CONFIGDIR "${PROJECT_INSTALL_ROOT}/${CONFIG_DIR}")
set(CMAKE_INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_BUILD_RPATH  "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_SKIP_INSTALL_RPATH OFF)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
    BUILD_RPATH   "${CMAKE_BUILD_RPATH}"
)

install(TARGETS ${PROJECT_NAME}
    COMPONENT runtime
    RUNTIME DESTINATION "${PROJECT_INSTALL_ROOT}"
    LIBRARY DESTINATION "/usr/lib/aarch64-linux-gnu/gstreamer-1.0/"
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/config/"
    COMPONENT runtime
    DESTINATION "${PROJECT_INSTALL_CONFIGDIR}"
)

include(cmake/CreatePackage.cmake)