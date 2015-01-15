# include this file in your CMakeLists.txt file using the add_subdirectory command.
project( MevisDicomTiff )

if( ELASTIX_USE_MEVISDICOMTIFF )
  add_definitions( -D_ELASTIX_USE_MEVISDICOMTIFF )
  add_library( mevisdcmtiff
    itkMevisDicomTiffImageIO.cxx
    itkMevisDicomTiffImageIOFactory.cxx
    itkUseMevisDicomTiff.cxx
  )
else()
  # avoid dependencies, but compile the lib, so that customers of this
  # lib can just call RegisterMevisDicomTiff without checking.
  add_library( mevisdcmtiff
    itkUseMevisDicomTiff.cxx
  )
endif()

target_link_libraries( mevisdcmtiff ${ITK_LIBRARIES} )

# Group in IDE's like Visual Studio
set_property( TARGET mevisdcmtiff PROPERTY FOLDER "libraries" )

# add mevisdcmtiff to the list of target_link_libraries in your cmakelists.txt file.

