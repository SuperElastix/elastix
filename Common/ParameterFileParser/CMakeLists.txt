# Sources of non-templated classes.
set(param_SRCS
  itkParameterFileParser.h
  itkParameterFileParser.cxx
  itkParameterMapInterface.h
  itkParameterMapInterface.cxx
)

source_group("Parser" FILES ${param_SRCS})

add_library(param STATIC ${param_SRCS})
if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
  install(TARGETS param
    ARCHIVE DESTINATION ${ELASTIX_ARCHIVE_DIR}
    LIBRARY DESTINATION ${ELASTIX_LIBRARY_DIR}
    RUNTIME DESTINATION ${ELASTIX_RUNTIME_DIR}
    COMPONENT Development)
endif()

target_link_libraries(param ${ITK_LIBRARIES})
target_include_directories(param PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${ELASTIX_INCLUDE_DIR}/Common/ParameterFileParser>)
# Note: param sources include elxConfiguration.h and elxConversion.h from Core.
# This is a circular dependency (elxCore links param), so we add these as PRIVATE.
target_include_directories(param PRIVATE
  ${elastix_SOURCE_DIR}/Common
  ${elastix_SOURCE_DIR}/Core/Configuration
  ${elastix_SOURCE_DIR}/Core/Install
  ${elastix_SOURCE_DIR}/Core/Kernel
  ${elastix_BINARY_DIR}
)

# Group in IDE's like Visual Studio
set_property(TARGET param PROPERTY FOLDER "libraries")
