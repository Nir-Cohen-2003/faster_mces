cmake_minimum_required(VERSION 3.15)
project(fast_mces_lower_bound LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add options so -DFAST_MCES_PROFILE=1 passed to cmake sets the variable
option(FAST_MCES_PROFILE "Enable profiling code" OFF)
option(FAST_MCES_ERROR_COUNT "Enable error counting" OFF)

if(FAST_MCES_PROFILE)
  message(STATUS "FAST_MCES_PROFILE enabled")
endif()
if(FAST_MCES_ERROR_COUNT)
  message(STATUS "FAST_MCES_ERROR_COUNT enabled")
endif()

find_package(Boost REQUIRED COMPONENTS system serialization iostreams)
# find_package(Eigen3 REQUIRED)

# Ensure CMake finds Python from the environment, not the system
# These must be set BEFORE find_package(Python)
set(Python_FIND_STRATEGY LOCATION CACHE STRING "Python find strategy")
set(Python_FIND_VIRTUALENV FIRST CACHE STRING "Python find virtualenv priority")
set(Python_FIND_REGISTRY NEVER CACHE STRING "Python find registry")
set(Python_FIND_FRAMEWORK NEVER CACHE STRING "Python find framework")

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)

# Print the Python version being used
message(STATUS "Using Python: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Python include dir: ${Python_INCLUDE_DIRS}")
message(STATUS "Python library dir: ${Python_LIBRARY_DIRS}")
message(STATUS "Python site-packages will be: ${Python_SITEARCH}")

find_package(OpenMP REQUIRED)

# Detect prefix: prefer PREFIX env var (conda/pixi), fallback to Python's sys.prefix
if(DEFINED ENV{PREFIX} AND NOT "$ENV{PREFIX}" STREQUAL "")
  set(_detected_prefix "$ENV{PREFIX}")
else()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE _detected_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()
message(STATUS "Detected prefix: ${_detected_prefix}")

# Set RDKit paths if not already set by cmake define
if(NOT RDKIT_INCLUDE_DIR)
  if(WIN32)
    set(RDKIT_INCLUDE_DIR "${_detected_prefix}/Library/include/rdkit" CACHE PATH "RDKit include dir")
  else()
    set(RDKIT_INCLUDE_DIR "${_detected_prefix}/include/rdkit" CACHE PATH "RDKit include dir")
  endif()
endif()
if(NOT RDKIT_LIB_DIR)
  if(WIN32)
    set(RDKIT_LIB_DIR "${_detected_prefix}/Library/lib" CACHE PATH "RDKit lib dir")
  else()
    set(RDKIT_LIB_DIR "${_detected_prefix}/lib" CACHE PATH "RDKit lib dir")
  endif()
endif()
message(STATUS "RDKIT_INCLUDE_DIR = ${RDKIT_INCLUDE_DIR}")
message(STATUS "RDKIT_LIB_DIR = ${RDKIT_LIB_DIR}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
file(TO_CMAKE_PATH "${nanobind_ROOT}" nanobind_ROOT)
message(STATUS "nanobind_ROOT: ${nanobind_ROOT}")

list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python site-packages from sysconfig: ${PYTHON_SITE_PACKAGES}")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox")
endif()

add_library(mces_lower_bound src/cpp_filter.cpp src/lap.cpp)
set_target_properties(mces_lower_bound PROPERTIES POSITION_INDEPENDENT_CODE ON)

# propagate compile-time flags to the target(s)
if(FAST_MCES_PROFILE)
  target_compile_definitions(mces_lower_bound PRIVATE FAST_MCES_PROFILE=1)
endif()
if(FAST_MCES_ERROR_COUNT)
  target_compile_definitions(mces_lower_bound PRIVATE FAST_MCES_ERROR_COUNT=1)
endif()

target_include_directories(mces_lower_bound PRIVATE
    ${RDKIT_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    # ${Eigen3_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_directories(mces_lower_bound PRIVATE ${RDKIT_LIB_DIR})

set(RDKIT_LIBS
    RDKitGraphMol
    RDKitSmilesParse
    RDKitInchi
    RDKitRDInchiLib
    RDKitMolStandardize
    RDKitFileParsers
    RDKitSubstructMatch
    RDKitFingerprints
    RDKitDepictor
)

target_link_libraries(mces_lower_bound PRIVATE
    ${RDKIT_LIBS}
    ${Boost_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# target_compile_definitions(mces_lower_bound PRIVATE USE_LAPJV)

install(TARGETS mces_lower_bound
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
install(FILES src/cpp_filter.hpp DESTINATION include/fast_mces_lower_bound)

nanobind_add_module(fast_mces_lower_bound STABLE_ABI src/pybind.cpp)
target_link_directories(fast_mces_lower_bound PRIVATE ${RDKIT_LIB_DIR})
target_link_libraries(fast_mces_lower_bound PRIVATE mces_lower_bound ${RDKIT_LIBS} OpenMP::OpenMP_CXX)

# also ensure module target gets same defs if needed
if(FAST_MCES_PROFILE)
  target_compile_definitions(fast_mces_lower_bound PRIVATE FAST_MCES_PROFILE=1)
endif()
if(FAST_MCES_ERROR_COUNT)
  target_compile_definitions(fast_mces_lower_bound PRIVATE FAST_MCES_ERROR_COUNT=1)
endif()

target_include_directories(fast_mces_lower_bound PRIVATE
    ${Python_INCLUDE_DIRS}
    ${RDKIT_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    # ${Eigen3_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
# Set RPATH for the Python extension.
# In a Conda/Pixi environment, the layout is:
# site-packages/fast_mces_lower_bound/fast_mces_lower_bound.abi3.so
# The RDKit libraries are in:
# lib/libRDKitGraphMol.so.1
# So we need to go up 3 levels from the .so to reach the prefix root, then into lib.
# $ORIGIN (fast_mces_lower_bound/) -> .. (site-packages/) -> .. (prefix/) -> lib/
# Thus $ORIGIN/../../lib or $ORIGIN/../../.. should work depending on how it's resolved.
# Standard Conda RPATH is often $ORIGIN/../../../lib or just $ORIGIN/../.. if it's in a subdirectory of prefix.
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(_rpath "@loader_path/../../../")
elseif(UNIX AND NOT APPLE)
    set(_rpath "$ORIGIN/../../..")
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(fast_mces_lower_bound PROPERTIES
        SKIP_BUILD_RPATH FALSE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${_rpath}"
    )
endif()

install(
    TARGETS fast_mces_lower_bound
    EXPORT fast_mces_lower_boundTargets
    LIBRARY DESTINATION .
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
