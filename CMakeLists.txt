##
#  CMake script for the EulerianSpray program:
##

# Set the name of the project and target:
set(TARGET "main")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    file(GLOB_RECURSE TARGET_SRC  "source/*.cpp")
#    file(GLOB_RECURSE TARGET_INC  "include/*.h")
#    set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
set(TARGET_SRC
  src/${TARGET}.cpp
  src/EulerianSprayProblem.cpp
  inc/EulerianSprayProblem.h
  )


cmake_minimum_required(VERSION 3.13.4)
# Find dealii
find_package(deal.II 9.3.1 #u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii 
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_invoke_autopilot()

# Include the directory for headers
target_include_directories(${TARGET} PRIVATE inc)