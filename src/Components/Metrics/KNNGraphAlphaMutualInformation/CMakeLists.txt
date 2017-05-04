
ADD_ELXCOMPONENT( KNNGraphAlphaMutualInformationMetric
 elxKNNGraphAlphaMutualInformationMetric.h
 elxKNNGraphAlphaMutualInformationMetric.hxx
 elxKNNGraphAlphaMutualInformationMetric.cxx
 itkKNNGraphAlphaMutualInformationImageToImageMetric.h
 itkKNNGraphAlphaMutualInformationImageToImageMetric.hxx
)

if( USE_KNNGraphAlphaMutualInformationMetric )
  add_subdirectory( KNN )
  target_include_directories( KNNGraphAlphaMutualInformationMetric
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/KNN )
  target_link_libraries( KNNGraphAlphaMutualInformationMetric KNNlib ANNlib )
endif()

