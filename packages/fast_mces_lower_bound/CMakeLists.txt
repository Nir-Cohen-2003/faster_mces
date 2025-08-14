cmake_minimum_required(VERSION 3.15)
project(mces LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Boost REQUIRED COMPONENTS system serialization iostreams)
# find_package(Eigen3 REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(OpenMP REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)
# now print whether nanobind was found, print anyway
if(TARGET nanobind::nanobind)
    message(STATUS "nanobind was found: nanobind::nanobind target is available.")
else()
    message(WARNING "nanobind was NOT found: nanobind::nanobind target is missing.")
endif()


# Manually set RDKit include and library paths
# set(RDKIT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/.pixi/envs/default/include/rdkit")
# set(RDKIT_LIB_DIR "${CMAKE_SOURCE_DIR}/.pixi/envs/default/lib")
set(RDKIT_INCLUDE_DIR "${RDKIT_INCLUDE_DIR}" CACHE PATH "RDKit include dir")
set(RDKIT_LIB_DIR "${RDKIT_LIB_DIR}" CACHE PATH "RDKit lib dir")
add_library(mces_lower_bound src/cpp_filter.cpp src/lap.cpp)

target_include_directories(mces_lower_bound PRIVATE
    ${RDKIT_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    # ${Eigen3_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_directories(mces_lower_bound PRIVATE ${RDKIT_LIB_DIR})

target_link_libraries(mces_lower_bound PRIVATE
    RDKitGraphMol
    RDKitSmilesParse
    # Eigen3::Eigen
    ${Boost_LIBRARIES}
    OpenMP::OpenMP_CXX
)

target_compile_definitions(mces_lower_bound PRIVATE USE_LAPJV)

install(TARGETS mces_lower_bound
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
install(FILES src/cpp_filter.hpp DESTINATION include/fast_mces_lower_bound)
nanobind_add_module(fast_mces_lower_bound src/pybind.cpp)
target_link_libraries(fast_mces_lower_bound PRIVATE mces_lower_bound OpenMP::OpenMP_CXX)
target_include_directories(fast_mces_lower_bound PRIVATE
    ${Python_INCLUDE_DIRS}
    ${RDKIT_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    # ${Eigen3_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
install(
    TARGETS fast_mces_lower_bound
    EXPORT fast_mces_lower_boundTargets
    LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)