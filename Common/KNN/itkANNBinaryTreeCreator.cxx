/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
