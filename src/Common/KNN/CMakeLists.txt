#---------------------------------------------------------------------
PROJECT( KNN )

# Set the files, so that they are shown in the editor
SET( KNN_Files
  itkListSampleCArray.h
  itkListSampleCArray.txx
  itkANNBinaryTreeCreator.h
  itkANNBinaryTreeCreator.cxx
  itkBinaryTreeBase.h
  itkBinaryTreeBase.txx
  itkBinaryANNTreeBase.h
  itkBinaryANNTreeBase.txx
  itkANNkDTree.h
  itkANNkDTree.txx
  itkANNbdTree.h
  itkANNbdTree.txx
  itkANNBruteForceTree.h
  itkANNBruteForceTree.txx
  itkBinaryTreeSearchBase.h
  itkBinaryTreeSearchBase.txx
  itkBinaryANNTreeSearchBase.h
  itkBinaryANNTreeSearchBase.txx
  itkANNStandardTreeSearch.h
  itkANNStandardTreeSearch.txx
  itkANNFixedRadiusTreeSearch.h
  itkANNFixedRadiusTreeSearch.txx
  itkANNPriorityTreeSearch.h
  itkANNPriorityTreeSearch.txx
)

# process the sub-directories
ADD_SUBDIRECTORY( ann_1.1 )

# Create the KNN library
ADD_LIBRARY( KNNlib ${KNN_Files} )

# Link it against the necessary libraries.
TARGET_LINK_LIBRARIES( KNNlib ANNlib ITKBasicFilters ITKCommon ITKStatistics )

# Group in IDE's like Visual Studio
SET_PROPERTY( TARGET KNNlib PROPERTY FOLDER "libraries" )

