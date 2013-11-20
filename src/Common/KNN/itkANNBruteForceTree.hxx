/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkANNBruteForceTree_hxx
#define __itkANNBruteForceTree_hxx

#include "itkANNBruteForceTree.h"
#include "itkANNBinaryTreeCreator.h"

namespace itk
{

  /**
   * ************************ Constructor *************************
   */

  template < class TListSample >
    ANNBruteForceTree<TListSample>
    ::ANNBruteForceTree()
  {
    this->m_ANNTree = 0;
  } // end Constructor


  /**
   * ************************ Destructor *************************
   */

  template < class TListSample >
    ANNBruteForceTree<TListSample>
    ::~ANNBruteForceTree()
  {
    ANNBinaryTreeCreator::DeleteANNBruteForceTree( this->m_ANNTree );
  } // end Destructor


  /**
   * ************************ GenerateTree *************************
   */

  template < class TListSample >
    void ANNBruteForceTree<TListSample>
    ::GenerateTree( void )
  {
    int dim = static_cast< int >( this->GetDataDimension() );
    int nop = static_cast< int >( this->GetActualNumberOfDataPoints() );

    ANNBinaryTreeCreator::DeleteANNBruteForceTree( this->m_ANNTree );

    this->m_ANNTree = ANNBinaryTreeCreator::CreateANNBruteForceTree(
      this->GetSample()->GetInternalContainer(), nop, dim );

  } // end GenerateTree


} // end namespace itk


#endif // end #ifndef __itkANNBruteForceTree_hxx
