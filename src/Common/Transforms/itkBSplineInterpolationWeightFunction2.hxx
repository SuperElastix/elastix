/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBSplineInterpolationWeightFunction2_hxx
#define __itkBSplineInterpolationWeightFunction2_hxx

#include "itkBSplineInterpolationWeightFunction2.h"

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template< class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
BSplineInterpolationWeightFunction2< TCoordRep, VSpaceDimension, VSplineOrder >
::BSplineInterpolationWeightFunction2()
{
  /** Initialize the interpolation kernel. */
  this->m_Kernel = KernelType::New();

} // end Constructor


/**
 * ******************* Compute1DWeights *******************
 */

template< class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
void
BSplineInterpolationWeightFunction2< TCoordRep, VSpaceDimension, VSplineOrder >
::Compute1DWeights(
  const ContinuousIndexType & index,
  const IndexType & startIndex,
  OneDWeightsType & weights1D ) const
{
  /** Compute the 1D weights. */
  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    double x = index[ i ] - static_cast< double >( startIndex[ i ] );

//     for ( unsigned int k = 0; k < this->m_SupportSize[ i ]; ++k )
//     {
//       weights1D[ i ][ k ] = this->m_Kernel->Evaluate( x );
//       x -= 1.0;
//     }
    WeightArrayType weights;
    this->m_Kernel->Evaluate( x, weights );

    for( unsigned int k = 0; k < this->m_SupportSize[ i ]; ++k )
    {
      weights1D[ i ][ k ] = weights[ k ];
    }
  }

} // end Compute1DWeights()


} // end namespace itk

#endif
