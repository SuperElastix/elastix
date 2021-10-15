/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkANNPriorityTreeSearch_hxx
#define itkANNPriorityTreeSearch_hxx

#include "itkANNPriorityTreeSearch.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <class TBinaryTree>
ANNPriorityTreeSearch<TBinaryTree>::ANNPriorityTreeSearch()
{
  this->m_ErrorBound = 0.0;
  this->m_BinaryTreeAskDTree = nullptr;
} // end Constructor


/**
 * ************************ SetBinaryTree *************************
 *
 */

template <class TBinaryTree>
void
ANNPriorityTreeSearch<TBinaryTree>::SetBinaryTree(BinaryTreeType * tree)
{
  this->Superclass::SetBinaryTree(tree);
  if (tree)
  {
    ANNPointSetType * ps = this->m_BinaryTreeAsITKANNType->GetANNTree();
    if (ps)
    {
      // ANNkDTreeType * testPtr = dynamic_cast<ANNkDTreeType *>( this->m_BinaryTreeAsITKANNType->GetANNTree() );
      ANNkDTreeType * testPtr = dynamic_cast<ANNkDTreeType *>(ps);
      if (testPtr)
      {
        if (testPtr != this->m_BinaryTreeAskDTree)
        {
          this->m_BinaryTreeAskDTree = testPtr;
          this->Modified();
        }
      }
      else
      {
        itkExceptionMacro(
          << "ERROR: The internal tree is not of ANNkd_tree type, which is required for priority search.");
      }
    }
    else
    {
      itkExceptionMacro(<< "ERROR: Tree is not generated.");
    }
  }
  else
  {
    this->m_BinaryTreeAskDTree = nullptr;
  }

} // end SetBinaryTree


/**
 * ************************ Search *************************
 *
 * The error bound eps is ignored.
 */

template <class TBinaryTree>
void
ANNPriorityTreeSearch<TBinaryTree>::Search(const MeasurementVectorType & qp,
                                           IndexArrayType &              ind,
                                           DistanceArrayType &           dists)
{
  /** Get k , dim and eps. */
  int    k = static_cast<int>(this->m_KNearestNeighbors);
  int    dim = static_cast<int>(this->m_DataDimension);
  double eps = this->m_ErrorBound;

  /** Allocate memory for ANN indices and distances arrays. */
  ANNIndexArrayType ANNIndices;
  ANNIndices = new ANNIndexType[k];

  ANNDistanceArrayType ANNDistances;
  ANNDistances = new ANNDistanceType[k];

  /** Alocate memory for ANN query point and copy qp to it. */
  ANNPointType ANNQueryPoint = annAllocPt(dim);
  for (int i = 0; i < dim; ++i)
  {
    ANNQueryPoint[i] = qp[i];
  }

  /** The actual ANN search. */
  this->m_BinaryTreeAskDTree->annkPriSearch(ANNQueryPoint, k, ANNIndices, ANNDistances, eps);
  // this->m_BinaryTree->GetANNTree()->annkPriSearch(
  // ANNQueryPoint, k, ANNIndices, ANNDistances, eps );

  /** Set the ANNIndices and ANNDistances in the corresponding itk::Array's.
   * Memory management is transfered to these itk::Array's, which have SmartPointers
   * and therefore don't have to be regarded anymore. No deallocation of
   * ANNIndices and ANNDistances is needed now.
   */
  ind.SetData(ANNIndices, k, true);
  dists.SetData(ANNDistances, k, true);

  /** Deallocate the temporary ANNQueryPoint. */
  annDeallocPt(ANNQueryPoint);

} // end Search


} // end namespace itk

#endif // end #ifndef itkANNPriorityTreeSearch_hxx
