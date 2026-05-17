include(ExternalProject)
include(CMakeParseArguments)

function(superbuild_discover_projects out_var)
    file(GLOB_RECURSE preset_files RELATIVE "${CMAKE_SOURCE_DIR}" "*/CMakePresets.json")

    set(discovered)
    foreach(preset_file IN LISTS preset_files)
        get_filename_component(project_dir "${preset_file}" DIRECTORY)

        if(project_dir STREQUAL "")
            continue()
        endif()

        if(project_dir MATCHES "(^|/)build($|/)")
            continue()
        endif()

        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${project_dir}/CMakeLists.txt")
            continue()
        endif()

        get_filename_component(project_name "${project_dir}" NAME)
        list(APPEND discovered "${project_name}|${CMAKE_SOURCE_DIR}/${project_dir}")
    endforeach()

    list(REMOVE_DUPLICATES discovered)
    list(SORT discovered)
    set(${out_var} "${discovered}" PARENT_SCOPE)
endfunction()

function(superbuild_add_project)
    set(options)
    set(oneValueArgs NAME SOURCE_DIR ENABLED)
    set(multiValueArgs)
    cmake_parse_arguments(SB "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED SB_NAME OR NOT DEFINED SB_SOURCE_DIR)
        message(FATAL_ERROR "superbuild_add_project requires NAME and SOURCE_DIR")
    endif()

    if(NOT DEFINED SB_ENABLED)
        set(SB_ENABLED ON)
    endif()

    if(NOT SB_ENABLED)
        message(STATUS "Superbuild: skipping ${SB_NAME}")
        return()
    endif()

    set(extra_args)
    if(DEFINED CMAKE_TOOLCHAIN_FILE AND NOT "${CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
        list(APPEND extra_args -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    endif()

    if(DEFINED CMAKE_BUILD_TYPE AND NOT "${CMAKE_BUILD_TYPE}" STREQUAL "")
        list(APPEND extra_args -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
    endif()

    set(project_build_dir "<SOURCE_DIR>/build/${SUPERBUILD_PRESET}")

    ExternalProject_Add(${SB_NAME}
        SOURCE_DIR "${SB_SOURCE_DIR}"
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            --preset ${SUPERBUILD_PRESET}
            -S <SOURCE_DIR>
            -DINSTALL_ROOT=${SUPERBUILD_INSTALL_ROOT}
            ${extra_args}
        BUILD_COMMAND ${CMAKE_COMMAND} --build ${project_build_dir} --parallel
        INSTALL_COMMAND ""
        TEST_COMMAND ""
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
    )

    ExternalProject_Add_Step(${SB_NAME} package
        COMMAND ${CMAKE_COMMAND} --build ${project_build_dir} --target package
        DEPENDEES build
        USES_TERMINAL TRUE
    )

    ExternalProject_Add_StepTargets(${SB_NAME} package)

    set_property(GLOBAL APPEND PROPERTY SUPERBUILD_BUILD_TARGETS ${SB_NAME})
    set_property(GLOBAL APPEND PROPERTY SUPERBUILD_PACKAGE_TARGETS ${SB_NAME}-package)
endfunction()

function(superbuild_finalize)
    get_property(build_targets GLOBAL PROPERTY SUPERBUILD_BUILD_TARGETS)
    get_property(package_targets GLOBAL PROPERTY SUPERBUILD_PACKAGE_TARGETS)

    if(NOT build_targets)
        add_custom_target(superbuild ALL)
        return()
    endif()

    add_custom_target(superbuild ALL DEPENDS ${build_targets})

    if(package_targets)
        add_custom_target(package DEPENDS ${package_targets})
    endif()
endfunction()

function(superbuild_add_discovered_projects)
    superbuild_discover_projects(discovered_projects)

    if(NOT discovered_projects)
        message(WARNING "Superbuild: no subprojects discovered")
        return()
    endif()

    foreach(entry IN LISTS discovered_projects)
        string(REPLACE "|" ";" parts "${entry}")
        list(GET parts 0 project_name)
        list(GET parts 1 project_source_dir)
        superbuild_add_project(
            NAME "${project_name}"
            SOURCE_DIR "${project_source_dir}"
            ENABLED ON
        )
    endforeach()
endfunction()
