# include this file in your CMakeLists.txt file using the add_subdirectory command.
PROJECT( MevisDicomTiff )

ADD_LIBRARY( mevisdcmtiff
  itkMevisDicomTiffImageIO.cxx
  itkMevisDicomTiffImageIOFactory.cxx
  itkUseMevisDicomTiff.h
)

TARGET_LINK_LIBRARIES( mevisdcmtiff ${ITK_LIBRARIES} )

# add mevisdcmtiff to the list of target_link_libraries in your cmakelists.txt file.

