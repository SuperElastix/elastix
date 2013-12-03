/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkANNPriorityTreeSearch_hxx
#define __itkANNPriorityTreeSearch_hxx

#include "itkANNPriorityTreeSearch.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template< class TBinaryTree >
ANNPriorityTreeSearch< TBinaryTree >
::ANNPriorityTreeSearch()
{
  this->m_ErrorBound         = 0.0;
  this->m_BinaryTreeAskDTree = 0;
}   // end Constructor


/**
 * ************************ Destructor *************************
 */

template< class TBinaryTree >
ANNPriorityTreeSearch< TBinaryTree >
::~ANNPriorityTreeSearch()
{}  // end Destructor

/**
 * ************************ SetBinaryTree *************************
 *
 */

template< class TBinaryTree >
void
ANNPriorityTreeSearch< TBinaryTree >
::SetBinaryTree( BinaryTreeType * tree )
{
  this->Superclass::SetBinaryTree( tree );
  if( tree )
  {
    ANNPointSetType * ps = this->m_BinaryTreeAsITKANNType->GetANNTree();
    if( ps )
    {
      //ANNkDTreeType * testPtr = dynamic_cast<ANNkDTreeType *>( this->m_BinaryTreeAsITKANNType->GetANNTree() );
      ANNkDTreeType * testPtr = dynamic_cast< ANNkDTreeType * >( ps );
      if( testPtr )
      {
        if( testPtr != this->m_BinaryTreeAskDTree )
        {
          this->m_BinaryTreeAskDTree = testPtr;
          this->Modified();
        }
      }
      else
      {
        itkExceptionMacro( << "ERROR: The internal tree is not of ANNkd_tree type, which is required for priority search." );
      }
    }
    else
    {
      itkExceptionMacro( << "ERROR: Tree is not generated." );
    }
  }
  else
  {
    this->m_BinaryTreeAskDTree = 0;
  }

}   // end SetBinaryTree


/**
 * ************************ Search *************************
 *
 * The error bound eps is ignored.
 */

template< class TBinaryTree >
void
ANNPriorityTreeSearch< TBinaryTree >
::Search( const MeasurementVectorType & qp, IndexArrayType & ind,
  DistanceArrayType & dists )
{
  /** Get k , dim and eps. */
  int    k   = static_cast< int >( this->m_KNearestNeighbors );
  int    dim = static_cast< int >( this->m_DataDimension );
  double eps = this->m_ErrorBound;

  /** Allocate memory for ANN indices and distances arrays. */
  ANNIndexArrayType ANNIndices;
  ANNIndices = new ANNIndexType[ k ];

  ANNDistanceArrayType ANNDistances;
  ANNDistances = new ANNDistanceType[ k ];

  /** Alocate memory for ANN query point and copy qp to it. */
  ANNPointType ANNQueryPoint = annAllocPt( dim );
  for( int i = 0; i < dim; i++ )
  {
    ANNQueryPoint[ i ] = qp[ i ];
  }

  /** The actual ANN search. */
  this->m_BinaryTreeAskDTree->annkPriSearch(
    ANNQueryPoint, k, ANNIndices, ANNDistances, eps );
  //this->m_BinaryTree->GetANNTree()->annkPriSearch(
  //ANNQueryPoint, k, ANNIndices, ANNDistances, eps );

  /** Set the ANNIndices and ANNDistances in the corresponding itk::Array's.
   * Memory management is transfered to these itk::Array's, which have SmartPointers
   * and therefore don't have to be regarded anymore. No deallocation of
   * ANNIndices and ANNDistances is needed now.
   */
  ind.SetData( ANNIndices, k, true );
  dists.SetData( ANNDistances, k, true );

  /** Deallocate the temporary ANNQueryPoint. */
  annDeallocPt( ANNQueryPoint );

}   // end Search


} // end namespace itk

#endif // end #ifndef __itkANNPriorityTreeSearch_hxx
