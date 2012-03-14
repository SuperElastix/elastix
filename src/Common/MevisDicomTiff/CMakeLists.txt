# include this file in your CMakeLists.txt file using the add_subdirectory command.
PROJECT( MevisDicomTiff )

IF( ELASTIX_USE_MEVISDICOMTIFF )
  ADD_DEFINITIONS( -D_ELASTIX_USE_MEVISDICOMTIFF )
ENDIF()

ADD_LIBRARY( mevisdcmtiff
  itkMevisDicomTiffImageIO.cxx
  itkMevisDicomTiffImageIOFactory.cxx
  itkUseMevisDicomTiff.cxx
)

TARGET_LINK_LIBRARIES( mevisdcmtiff ${ITK_LIBRARIES} )

# add mevisdcmtiff to the list of target_link_libraries in your cmakelists.txt file.

