add_subdirectory(Statismo)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/Statismo/core/include
  ${CMAKE_CURRENT_SOURCE_DIR}/Statismo/ITK/include)

# message(STATUS "ITKInternalEigen3_DIR: ${ITKInternalEigen3_DIR}")
# include_directories(${ITKInternalEigen3_DIR})


ADD_ELXCOMPONENT( ActiveRegistrationModelShapeMetric ON
  elxActiveRegistrationModelShapeMetric.cxx
  elxActiveRegistrationModelShapeMetric.h
  elxActiveRegistrationModelShapeMetric.hxx
  itkActiveRegistrationModelShapeMetric.h
  itkActiveRegistrationModelShapeMetric.hxx
)

find_package(Eigen3 REQUIRED CONFIG)
target_link_libraries(ActiveRegistrationModelShapeMetric Eigen3::Eigen)

# If you see a compilation error along the lines of
#
#  /usr/local/include/ITK-4.8/itkhdf5/cpp/H5Exception.h:31:17: error: redefinition of 'Exception'
#
# it is because the itk hdf5 header guards deviate from those in statismo hdf5.
# See https://bugs.launchpad.net/ubuntu/+source/insighttoolkit4/+bug/1376385
#
# Dirty fix (linux):
#
#  sudo sed -i 's|#ifndef _|#ifndef __|g' /usr/include/ITK-4.8/itkhdf5/cpp/*
#  sudo sed -i 's|#define _|#define __|g' /usr/include/ITK-4.8/itkhdf5/cpp/*
#
# Dirty fix (mac):
#
#  sudo sed -i '' 's|#ifndef _|#ifndef __|g' /usr/local/include/ITK-4.8/itkhdf5/cpp/*
#  sudo sed -i '' 's|#define _|#define __|g' /usr/local/include/ITK-4.8/itkhdf5/cpp/*
