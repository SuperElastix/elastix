/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBinaryTreeSearchBase_hxx
#define __itkBinaryTreeSearchBase_hxx

#include "itkBinaryTreeSearchBase.h"

namespace itk
{

  /**
   * ************************ Constructor *************************
   */

  template < class TBinaryTree >
    BinaryTreeSearchBase<TBinaryTree>
    ::BinaryTreeSearchBase()
  {
    this->m_BinaryTree = 0;
    this->m_KNearestNeighbors = 1;
  } // end Constructor


  /**
   * ************************ Destructor *************************
   */

  template < class TBinaryTree >
    BinaryTreeSearchBase<TBinaryTree>
    ::~BinaryTreeSearchBase()
  {
  } // end Destructor


  /**
   * ************************ SetBinaryTree *************************
   */

  template < class TBinaryTree >
    void BinaryTreeSearchBase<TBinaryTree>
    ::SetBinaryTree( BinaryTreeType * tree )
  {
    if ( this->m_BinaryTree != tree )
    {
      this->m_BinaryTree = tree;
      if ( tree )
      {
        this->m_DataDimension = this->m_BinaryTree->GetDataDimension();
      }
      this->Modified();
    }
  } // end SetBinaryTree


  /**
   * ************************ GetBinaryTree *************************
   */

  template < class TBinaryTree >
    const typename BinaryTreeSearchBase<TBinaryTree>::BinaryTreeType *
    BinaryTreeSearchBase<TBinaryTree>
    ::GetBinaryTree( void ) const
  {
    return this->m_BinaryTree.GetPointer();
  } // end GetBinaryTree


} // end namespace itk


#endif // end #ifndef __itkBinaryTreeSearchBase_hxx
