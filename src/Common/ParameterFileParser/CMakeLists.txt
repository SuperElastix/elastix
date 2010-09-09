# Sources of non-templated classes.
SET( param_SRCS
  itkParameterFileParser.h
  itkParameterFileParser.cxx
  itkParameterMapInterface.h
  itkParameterMapInterface.cxx
)

SOURCE_GROUP( "Parser" FILES ${param_SRCS} )

ADD_LIBRARY( param ${param_SRCS} )

TARGET_LINK_LIBRARIES( param ITKCommon )

# Group in IDE's like Visual Studio
SET_PROPERTY( TARGET param PROPERTY FOLDER "libraries" )
