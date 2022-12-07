
ADD_ELXCOMPONENT( KNNGraphAlphaMutualInformationMetric
 elxKNNGraphAlphaMutualInformationMetric.h
 elxKNNGraphAlphaMutualInformationMetric.hxx
 elxKNNGraphAlphaMutualInformationMetric.cxx
 itkKNNGraphAlphaMutualInformationImageToImageMetric.h
 itkKNNGraphAlphaMutualInformationImageToImageMetric.hxx
)

if(USE_KNNGraphAlphaMutualInformationMetric)
  if(ELASTIX_NO_INSTALL_RUNTIME_LIBRARIES AND NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
    message(SEND_ERROR "USE_KNNGraphAlphaMutualInformationMetric with ELASTIX_NO_INSTALL_RUNTIME_LIBRARIES enabled requires ELASTIX_NO_INSTALL_DEVELOPMENT enabled")
  endif()
  add_subdirectory(KNN)
  target_include_directories(KNNGraphAlphaMutualInformationMetric
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/KNN)
  target_link_libraries(KNNGraphAlphaMutualInformationMetric KNNlib ANNlib)
  elastix_export_target(KNNlib)
  elastix_export_target(ANNlib)
endif()

