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

#ifndef __itkANNBinaryTreeCreator_cxx
#define __itkANNBinaryTreeCreator_cxx

#include "itkANNBinaryTreeCreator.h"

namespace itk
{

unsigned int ANNBinaryTreeCreator::m_NumberOfANNBinaryTrees = 0;

/**
 * ************************ CreateANNkDTree *************************
 */

ANNBinaryTreeCreator::ANNkDTreeType *
ANNBinaryTreeCreator::CreateANNkDTree(
  ANNPointArrayType pa, int n, int d, int bs,
  ANNSplitRuleType split )
{
  IncreaseReferenceCount();
  return new ANNkd_tree( pa, n, d, bs, split );
}   // end CreateANNkDTree


/**
 * ************************ CreateANNbdTree *************************
 */

ANNBinaryTreeCreator::ANNbdTreeType *
ANNBinaryTreeCreator::CreateANNbdTree(
  ANNPointArrayType pa, int n, int d, int bs,
  ANNSplitRuleType split, ANNShrinkRuleType shrink )
{
  IncreaseReferenceCount();
  return new ANNbd_tree( pa, n, d, bs, split, shrink );
}   // end CreateANNbdTree


/**
 * ************************ CreateANNBruteForceTree *************************
 */

ANNBinaryTreeCreator::ANNBruteForceTreeType *
ANNBinaryTreeCreator::CreateANNBruteForceTree(
  ANNPointArrayType pa, int n, int d )
{
  IncreaseReferenceCount();
  return new ANNbruteForce( pa, n, d );
}   // end CreateANNBruteForceTree


/**
 * ************************ DeleteANNkDTree *************************
 */

void
ANNBinaryTreeCreator::DeleteANNkDTree( ANNkDTreeType * & tree )
{
  if( tree )
  {
    delete tree;
    tree = 0;
    DecreaseReferenceCount();
  }
}   // end DeleteANNkDTree


/**
 * ************************ DeleteANNBruteForceTree *************************
 */

void
ANNBinaryTreeCreator::DeleteANNBruteForceTree( ANNBruteForceTreeType * & tree )
{
  if( tree )
  {
    delete tree;
    tree = 0;
    DecreaseReferenceCount();
  }
}   // end DeleteANNBruteForceTree


/**
 * ************************ IncreaseReferenceCount *************************
 */

void
ANNBinaryTreeCreator::IncreaseReferenceCount( void )
{
  m_NumberOfANNBinaryTrees++;
}   // end IncreaseReferenceCount


/**
 * ************************ DecreaseReferenceCount *************************
 */

void
ANNBinaryTreeCreator::DecreaseReferenceCount( void )
{
  m_NumberOfANNBinaryTrees--;
  if( m_NumberOfANNBinaryTrees == 0 )
  {
    annClose();
  }
}   // end DecreaseReferenceCount


} // end namespace itk

#endif // end #ifndef __itkANNBinaryTreeCreator_cxx
