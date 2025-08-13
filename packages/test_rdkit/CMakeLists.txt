cmake_minimum_required(VERSION 3.10)
project(RDKitSV)

# Manually set RDKit include and library paths
set(RDKIT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/.pixi/envs/default/include/rdkit")
set(RDKIT_LIB_DIR "${CMAKE_SOURCE_DIR}/.pixi/envs/default/lib")

add_executable(example1 example1.cpp)
target_include_directories(example1 PRIVATE ${RDKIT_INCLUDE_DIR})
target_link_directories(example1 PRIVATE ${RDKIT_LIB_DIR})
target_link_libraries(example1 PRIVATE RDKitGraphMol RDKitSmilesParse RDKitRDGeometryLib RDKitRDGeneral RDKitSubgraphs)

# If you need Boost/Eigen, keep these lines as well
find_package(Boost REQUIRED COMPONENTS iostreams filesystem system)
find_package(Eigen3 REQUIRED)
target_include_directories(example1 PRIVATE ${Boost_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})
target_link_libraries(example1 PRIVATE ${Boost_LIBRARIES} Eigen3::Eigen)

