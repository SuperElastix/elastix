# include this file in your CMakeLists.txt file using the add_subdirectory command.
project(MevisDicomTiff)

if(ELASTIX_USE_MEVISDICOMTIFF)
  add_definitions(-D_ELASTIX_USE_MEVISDICOMTIFF)
  add_library(mevisdcmtiff STATIC
    itkMevisDicomTiffImageIO.cxx
    itkMevisDicomTiffImageIOFactory.cxx
    itkUseMevisDicomTiff.cxx
  )
else()
  # avoid dependencies, but compile the lib, so that customers of this
  # lib can just call RegisterMevisDicomTiff without checking.
  add_library(mevisdcmtiff STATIC
    itkUseMevisDicomTiff.cxx
  )
endif()

if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
  install(TARGETS mevisdcmtiff
    ARCHIVE DESTINATION ${ELASTIX_ARCHIVE_DIR}
    LIBRARY DESTINATION ${ELASTIX_LIBRARY_DIR}
    RUNTIME DESTINATION ${ELASTIX_RUNTIME_DIR}
    COMPONENT Development)
endif()

target_link_libraries(mevisdcmtiff ${ITK_LIBRARIES})
target_include_directories(mevisdcmtiff PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${ELASTIX_INCLUDE_DIR}/Common/MevisDicomTiff>)

# Group in IDE's like Visual Studio
set_property(TARGET mevisdcmtiff PROPERTY FOLDER "libraries")

# add mevisdcmtiff to the list of target_link_libraries in your cmakelists.txt file.

elastix_export_target(mevisdcmtiff)
