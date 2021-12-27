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
#ifndef itkBinaryTreeSearchBase_hxx
#define itkBinaryTreeSearchBase_hxx

#include "itkBinaryTreeSearchBase.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <class TBinaryTree>
BinaryTreeSearchBase<TBinaryTree>::BinaryTreeSearchBase()
{
  this->m_BinaryTree = nullptr;
  this->m_KNearestNeighbors = 1;
} // end Constructor


/**
 * ************************ SetBinaryTree *************************
 */

template <class TBinaryTree>
void
BinaryTreeSearchBase<TBinaryTree>::SetBinaryTree(BinaryTreeType * tree)
{
  if (this->m_BinaryTree != tree)
  {
    this->m_BinaryTree = tree;
    if (tree)
    {
      this->m_DataDimension = this->m_BinaryTree->GetDataDimension();
    }
    this->Modified();
  }
} // end SetBinaryTree


/**
 * ************************ GetBinaryTree *************************
 */

template <class TBinaryTree>
auto
BinaryTreeSearchBase<TBinaryTree>::GetBinaryTree() const -> const BinaryTreeType *
{
  return this->m_BinaryTree.GetPointer();
} // end GetBinaryTree

} // end namespace itk

#endif // end #ifndef itkBinaryTreeSearchBase_hxx
