# Should be included only from the top level cmake and before project()

# Get the vcpkg baseline commit hash from the git tag
set(VCPKG_GIT_TAG "2025.04.09")
set(VCPKG_GIT_PATH "https://github.com/Microsoft/vcpkg.git")
execute_process(
        COMMAND git ls-remote ${VCPKG_GIT_PATH} refs/tags/${VCPKG_GIT_TAG}
        OUTPUT_VARIABLE VCPKG_BASELINE_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE VCPKG_BASELINE_RESULT
)

if(VCPKG_BASELINE_RESULT EQUAL 0 AND VCPKG_BASELINE_OUTPUT)
    string(REGEX REPLACE "^([a-f0-9]+).*" "\\1" VCPKG_BUILTIN_BASELINE "${VCPKG_BASELINE_OUTPUT}")
else()
    message(FATAL_ERROR "Failed to retrieve vcpkg baseline for tag ${VCPKG_GIT_TAG}")
endif()

configure_file(
        ${CMAKE_SOURCE_DIR}/vcpkg.json.in
        ${CMAKE_SOURCE_DIR}/vcpkg.json
        @ONLY
)

include(FetchContent)
FetchContent_Declare(
        vcpkg
        GIT_REPOSITORY ${VCPKG_GIT_PATH}
        GIT_TAG ${VCPKG_GIT_TAG}
)
FetchContent_MakeAvailable(vcpkg)

if(DEFINED VCPKG_CHAINLOAD_TOOLCHAIN_FILE)
    set(CMAKE_CROSSCOMPILING TRUE)
endif()

set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file" FORCE)