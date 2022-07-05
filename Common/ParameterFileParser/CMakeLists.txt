# Sources of non-templated classes.
set(param_SRCS
  itkParameterFileParser.h
  itkParameterFileParser.cxx
  itkParameterMapInterface.h
  itkParameterMapInterface.cxx
)

source_group("Parser" FILES ${param_SRCS})

add_library(param STATIC ${param_SRCS})
install(TARGETS param
  ARCHIVE DESTINATION ${ELASTIX_ARCHIVE_DIR}
  LIBRARY DESTINATION ${ELASTIX_LIBRARY_DIR}
  RUNTIME DESTINATION ${ELASTIX_RUNTIME_DIR})

# Support #include <double-conversion.h>, which is either in the root of the
# ITK install directory (its "install prefix"), or at the following location
# of the ITK source tree (which is the parent directory of "{ITK_CMAKE_DIR}"):
target_include_directories(param PRIVATE
  ${ITK_CMAKE_DIR}/../Modules/ThirdParty/DoubleConversion/src/double-conversion)

target_link_libraries(param ${ITK_LIBRARIES})

# Group in IDE's like Visual Studio
set_property(TARGET param PROPERTY FOLDER "libraries")
