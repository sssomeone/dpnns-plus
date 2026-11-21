cmake_minimum_required(VERSION 3.20)
project(dpnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
    add_compile_options(/utf-8 /bigobj)
endif()

# CGAL (header-only), its Boost headers, and oneTBB.
find_package(CGAL REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
include(CGAL_TBB_support)            # CGAL::TBB_support: TBB + CGAL_LINKED_WITH_TBB
# Boost.Multiprecision of some Boost versions does not compile with recent
# MSVC; CGAL then uses its own multiprecision fallback.
add_compile_definitions(CGAL_DO_NOT_USE_BOOST_MP)

add_library(dpnn STATIC src/dpnn.cpp)
target_include_directories(dpnn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(dpnn PRIVATE CGAL::CGAL PUBLIC CGAL::TBB_support)

# Link-time optimization in release builds where the toolchain supports it.
include(CheckIPOSupported)
check_ipo_supported(RESULT DPNN_IPO OUTPUT DPNN_IPO_ERROR)
if(DPNN_IPO)
    set_property(TARGET dpnn PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

option(DPNN_BUILD_EXAMPLES "Build the example program" ON)
if(DPNN_BUILD_EXAMPLES)
    add_executable(dpnn_example examples/example.cpp)
    target_link_libraries(dpnn_example PRIVATE dpnn)
endif()

install(TARGETS dpnn ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
