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
#ifndef itkANNBruteForceTree_hxx
#define itkANNBruteForceTree_hxx

#include "itkANNBruteForceTree.h"
#include "itkANNBinaryTreeCreator.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <typename TListSample>
ANNBruteForceTree<TListSample>::ANNBruteForceTree()
{
  this->m_ANNTree = nullptr;
} // end Constructor


/**
 * ************************ Destructor *************************
 */

template <typename TListSample>
ANNBruteForceTree<TListSample>::~ANNBruteForceTree()
{
  ANNBinaryTreeCreator::DeleteANNBruteForceTree(this->m_ANNTree);
} // end Destructor


/**
 * ************************ GenerateTree *************************
 */

template <typename TListSample>
void
ANNBruteForceTree<TListSample>::GenerateTree()
{
  int dim = static_cast<int>(this->GetDataDimension());
  int nop = static_cast<int>(this->GetActualNumberOfDataPoints());

  ANNBinaryTreeCreator::DeleteANNBruteForceTree(this->m_ANNTree);

  this->m_ANNTree = ANNBinaryTreeCreator::CreateANNBruteForceTree(this->GetSample()->GetInternalContainer(), nop, dim);

} // end GenerateTree


} // end namespace itk

#endif // end #ifndef itkANNBruteForceTree_hxx
