# Sources of non-templated classes.
set( param_SRCS
  itkParameterFileParser.h
  itkParameterFileParser.cxx
  itkParameterMapInterface.h
  itkParameterMapInterface.cxx
)

source_group( "Parser" FILES ${param_SRCS} )

add_library( param STATIC ${param_SRCS} )

target_link_libraries( param ${ITK_LIBRARIES} )

# Group in IDE's like Visual Studio
set_property( TARGET param PROPERTY FOLDER "libraries" )
