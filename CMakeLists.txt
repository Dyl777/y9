cmake_minimum_required(VERSION 3.15)
project(y9_pytorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Collect C source files from y9 (excluding test files and main files)
set(Y9_C_SOURCES)
file(GLOB ALL_C_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

foreach(source ${ALL_C_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    # Exclude test files and main files
    if(NOT name MATCHES "^test_" AND NOT name MATCHES "^vn_" AND NOT name STREQUAL "threadpool")
        list(APPEND Y9_C_SOURCES ${source})
    endif()
endforeach()

# Create a static library for C sources
add_library(y9_c STATIC ${Y9_C_SOURCES})
target_include_directories(y9_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create the nanobind module
nanobind_add_module(_core python_bindings.cpp)

# Include directories
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python_INCLUDE_DIRS}
)

# Link with the C library
target_link_libraries(_core PRIVATE
    y9_c
    ${Python_LIBRARIES}
)

# Install the module
install(TARGETS _core LIBRARY DESTINATION y9)
