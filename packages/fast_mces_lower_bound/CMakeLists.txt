cmake_minimum_required(VERSION 3.15)
project(mces LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Boost REQUIRED COMPONENTS system serialization iostreams)
find_package(Eigen3 REQUIRED)

# Manually set RDKit include and library paths
set(RDKIT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/.pixi/envs/default/include/rdkit")
set(RDKIT_LIB_DIR "${CMAKE_SOURCE_DIR}/.pixi/envs/default/lib")

add_library(mces_lower_bound src/cpp_filter.cpp)

target_include_directories(mces_lower_bound PRIVATE
    ${RDKIT_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_directories(mces_lower_bound PRIVATE ${RDKIT_LIB_DIR})

target_link_libraries(mces_lower_bound PRIVATE
    RDKitGraphMol
    RDKitSmilesParse
    Eigen3::Eigen
    ${Boost_LIBRARIES}
)

install(TARGETS mces_lower_bound
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
install(FILES src/cpp_filter.hpp DESTINATION include/fast_mces_lower_bound)
