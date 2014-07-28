# include this file in your CMakeLists.txt file using the add_subdirectory command.
PROJECT( MevisDicomTiff )

IF( ELASTIX_USE_MEVISDICOMTIFF )
  ADD_DEFINITIONS( -D_ELASTIX_USE_MEVISDICOMTIFF )
  ADD_LIBRARY( mevisdcmtiff
    itkMevisDicomTiffImageIO.cxx
    itkMevisDicomTiffImageIOFactory.cxx
    itkUseMevisDicomTiff.cxx
  )
ELSE()
  # avoid dependencies, but compile the lib, so that customers of this
  # lib can just call RegisterMevisDicomTiff without checking.
  ADD_LIBRARY( mevisdcmtiff
    itkUseMevisDicomTiff.cxx
  )
ENDIF()

TARGET_LINK_LIBRARIES( mevisdcmtiff ${ITK_LIBRARIES} )

# Group in IDE's like Visual Studio
SET_PROPERTY( TARGET mevisdcmtiff PROPERTY FOLDER "libraries" )

# add mevisdcmtiff to the list of target_link_libraries in your cmakelists.txt file.

