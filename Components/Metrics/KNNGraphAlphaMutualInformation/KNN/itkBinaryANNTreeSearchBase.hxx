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
#ifndef itkBinaryANNTreeSearchBase_hxx
#define itkBinaryANNTreeSearchBase_hxx

#include "itkBinaryANNTreeSearchBase.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <class TBinaryTree>
BinaryANNTreeSearchBase<TBinaryTree>::BinaryANNTreeSearchBase()
{
  this->m_BinaryTreeAsITKANNType = nullptr;
} // end Constructor


/**
 * ************************ SetBinaryTree *************************
 */

template <class TBinaryTree>
void
BinaryANNTreeSearchBase<TBinaryTree>::SetBinaryTree(BinaryTreeType * tree)
{
  this->Superclass::SetBinaryTree(tree);
  if (tree)
  {
    BinaryANNTreeType * testPtr = dynamic_cast<BinaryANNTreeType *>(tree);
    if (testPtr)
    {
      if (testPtr != this->m_BinaryTreeAsITKANNType)
      {
        this->m_BinaryTreeAsITKANNType = testPtr;
        this->Modified();
      }
    }
    else
    {
      itkExceptionMacro(<< "ERROR: The tree is not of type BinaryANNTreeBase.");
    }
  }
  else
  {
    if (this->m_BinaryTreeAsITKANNType.IsNotNull())
    {
      this->m_BinaryTreeAsITKANNType = nullptr;
      this->Modified();
    }
  }

} // end SetBinaryTree


/**
 * ************************ GetBinaryTree *************************
 *

template < class TBinaryTree >
  auto
  BinaryANNTreeSearchBase<TBinaryTree>
  ::GetBinaryTree() const -> const BinaryTreeType *
{
  return this->m_BinaryTree.GetPointer();
} // end GetBinaryTree
*/

} // end namespace itk

#endif // end #ifndef itkBinaryANNTreeSearchBase_hxx
