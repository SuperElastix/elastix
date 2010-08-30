# include this file in your CMakeLists.txt file using the subdirs command
# or the new add_subdirectory command.
PROJECT( MevisDicomTiff )

# Assume that ITK already has been included in the following way:
# Find ITK.
#FIND_PACKAGE(ITK)
#IF(ITK_FOUND)
#  INCLUDE(${ITK_USE_FILE})
#ELSE(ITK_FOUND)
#  MESSAGE(FATAL_ERROR
#          "Cannot build without ITK.  Please set ITK_DIR.")
#ENDIF(ITK_FOUND)

ADD_LIBRARY( mevisdcmtiff
 itkMevisDicomTiffImageIO.cxx
 itkMevisDicomTiffImageIOFactory.cxx
 itkUseMevisDicomTiff.h
)
TARGET_LINK_LIBRARIES( mevisdcmtiff ITKIO ITKCommon )



# add mevisdcmtiff to the list of target_link_libraries in your cmakelists.txt file.

