#---------------------------------------------------------------------
project(KNN)

# Set the files, so that they are shown in the editor
set(KNN_Files
  itkListSampleCArray.h
  itkListSampleCArray.hxx
  itkANNBinaryTreeCreator.h
  itkANNBinaryTreeCreator.cxx
  itkBinaryTreeBase.h
  itkBinaryTreeBase.hxx
  itkBinaryANNTreeBase.h
  itkBinaryANNTreeBase.hxx
  itkANNkDTree.h
  itkANNkDTree.hxx
  itkANNbdTree.h
  itkANNbdTree.hxx
  itkANNBruteForceTree.h
  itkANNBruteForceTree.hxx
  itkBinaryTreeSearchBase.h
  itkBinaryTreeSearchBase.hxx
  itkBinaryANNTreeSearchBase.h
  itkBinaryANNTreeSearchBase.hxx
  itkANNStandardTreeSearch.h
  itkANNStandardTreeSearch.hxx
  itkANNFixedRadiusTreeSearch.h
  itkANNFixedRadiusTreeSearch.hxx
  itkANNPriorityTreeSearch.h
  itkANNPriorityTreeSearch.hxx
)

# process the sub-directories
add_subdirectory(ann_1.1)

# Create the KNN library
add_library(KNNlib STATIC ${KNN_Files})

# Link it against the necessary libraries.
elastix_link_itk(KNNlib MODULES
  ITKCommon
  ITKStatistics
  ITKSpatialObjects
)
target_link_libraries(KNNlib ANNlib)

# Group in IDE's like Visual Studio
set_property(TARGET KNNlib PROPERTY FOLDER "libraries")
