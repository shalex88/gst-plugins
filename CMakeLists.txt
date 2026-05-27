cmake_minimum_required(VERSION 3.22)
project(gst_plugins_superbuild LANGUAGES NONE)

include(ExternalProject)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Projects)

set(SUPERBUILD_PRESET "native-debug" CACHE STRING "Subproject preset used by the superbuild")
set(SUPERBUILD_INSTALL_ROOT "/tmp/" CACHE PATH "Install root propagated to subprojects")

if(DEFINED INSTALL_ROOT AND NOT "${INSTALL_ROOT}" STREQUAL "")
    set(SUPERBUILD_INSTALL_ROOT "${INSTALL_ROOT}" CACHE PATH "Install root propagated to subprojects" FORCE)
endif()

superbuild_add_discovered_projects()

superbuild_finalize()
