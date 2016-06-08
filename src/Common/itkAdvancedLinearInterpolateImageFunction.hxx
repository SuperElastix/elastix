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
#ifndef __itkAdvancedLinearInterpolateImageFunction_hxx
#define __itkAdvancedLinearInterpolateImageFunction_hxx

#include "itkAdvancedLinearInterpolateImageFunction.h"

#include "vnl/vnl_math.h"

namespace itk
{

/**
 * ***************** Constructor ***********************
 */

template< class TInputImage, class TCoordRep >
AdvancedLinearInterpolateImageFunction< TInputImage, TCoordRep >
::AdvancedLinearInterpolateImageFunction()
{}

/**
 * ***************** EvaluateDerivativeAtContinuousIndex ***********************
 */

//template< class TInputImage, class TCoordRep >
//typename AdvancedLinearInterpolateImageFunction< TInputImage, TCoordRep >
//::CovariantVectorType
//AdvancedLinearInterpolateImageFunction< TInputImage, TCoordRep >
//::EvaluateDerivativeAtContinuousIndex(
//  const ContinuousIndexType & x ) const
//{
//  /**
//   * Compute base index = closest index below point
//   * Compute distance from point to base index
//   */
//  IndexType baseIndex;
//  //double    distance[ ImageDimension ];
//  for ( dim = 0; dim < ImageDimension; dim++ )
//  {
//    baseIndex[ dim ] = Math::Floor< IndexValueType >( index[dim] );
//    //distance[  dim ] = index[ dim ] - static_cast< double >( baseIndex[dim] );
//  }
//
//  IndexType neighIndex = baseIndex;
//  IndexType upIndex = baseIndex;
//
//  CovariantVectorType derivative;
//  for ( unsigned int dim = 0; dim < ImageDimension; ++dim )
//  {
//    ++upIndex[ dim ];
//    derivative[ dim ] =
//      this->GetInputImage()->GetPixel( upIndex )
//      - this->GetInputImage()->GetPixel( baseIndex );
//  }
//
//} // end EvaluateDerivativeAtContinuousIndex()

/**
 * ***************** EvaluateValueAndDerivativeOptimized ***********************
 */

template< class TInputImage, class TCoordRep >
void
AdvancedLinearInterpolateImageFunction< TInputImage, TCoordRep >
::EvaluateValueAndDerivativeOptimized(
  const Dispatch< 2 > &,
  const ContinuousIndexType & x,
  OutputType & value,
  CovariantVectorType & deriv ) const
{
  // Get some handles
  const InputImageType *        inputImage = this->GetInputImage();
  const InputImageSpacingType & spacing    = inputImage->GetSpacing();

  /** Create a possibly mirrored version of x. */
  ContinuousIndexType xm = x;
  double              deriv_sign[ ImageDimension ];
  for( unsigned int dim = 0; dim < ImageDimension; dim++ )
  {
    deriv_sign[ dim ] = 1.0 / spacing[ dim ];
    if( x[ dim ] < this->m_StartIndex[ dim ] )
    {
      xm[ dim ]          = 2.0 * this->m_StartIndex[ dim ] - x[ dim ];
      deriv_sign[ dim ] *= -1.0;
    }
    if( x[ dim ] > this->m_EndIndex[ dim ] )
    {
      xm[ dim ]          = 2.0 * this->m_EndIndex[ dim ] - x[ dim ];
      deriv_sign[ dim ] *= -1.0;
    }

    /** Separately deal with cases on the image edge. */
    if( Math::FloatAlmostEqual( xm[ dim ], static_cast< ContinuousIndexValueType >( this->m_EndIndex[ dim ] ) ) )
    {
      xm[ dim ] -= 0.000001;
    }
  }
  // if this is mirrored again outside the image domain, then too bad.

  /**
   * Compute base index = closest index below point
   * Compute distance from point to base index
   */
  IndexType baseIndex;
  double    dist[ ImageDimension ];
  double    dinv[ ImageDimension ];
  for( unsigned int dim = 0; dim < ImageDimension; dim++ )
  {
    baseIndex[ dim ] = Math::Floor< IndexValueType >( xm[ dim ] );

    dist[ dim ] = xm[ dim ] - static_cast< double >( baseIndex[ dim ] );
    dinv[ dim ] = 1.0 - dist[ dim ];
  }

  /** Get the 4 corner values. */
  const RealType val00 = inputImage->GetPixel( baseIndex );
  ++baseIndex[ 0 ];
  const RealType val10 = inputImage->GetPixel( baseIndex );
  --baseIndex[ 0 ]; ++baseIndex[ 1 ];
  const RealType val01 = inputImage->GetPixel( baseIndex );
  ++baseIndex[ 0 ];
  const RealType val11 = inputImage->GetPixel( baseIndex );

  /** Interpolate to get the value. */
  value = static_cast< OutputType >(
    val00 * dinv[ 0 ] * dinv[ 1 ]
    + val10 * dist[ 0 ] * dinv[ 1 ]
    + val01 * dinv[ 0 ] * dist[ 1 ]
    + val11 * dist[ 0 ] * dist[ 1 ] );

  /** Interpolate to get the derivative. */
  deriv[ 0 ] = deriv_sign[ 0 ] * ( dinv[ 1 ] * ( val10 - val00 ) + dist[ 1 ] * ( val11 - val01 ) );
  deriv[ 1 ] = deriv_sign[ 1 ] * ( dinv[ 0 ] * ( val01 - val00 ) + dist[ 0 ] * ( val11 - val10 ) );

  /** Take direction cosines into account. */
  CovariantVectorType orientedDerivative;
  inputImage->TransformLocalVectorToPhysicalVector( deriv, orientedDerivative );
  deriv = orientedDerivative;

} // end EvaluateValueAndDerivativeOptimized()


/**
 * ***************** EvaluateValueAndDerivativeOptimized ***********************
 */

template< class TInputImage, class TCoordRep >
void
AdvancedLinearInterpolateImageFunction< TInputImage, TCoordRep >
::EvaluateValueAndDerivativeOptimized(
  const Dispatch< 3 > &,
  const ContinuousIndexType & x,
  OutputType & value,
  CovariantVectorType & deriv ) const
{
  // Get some handles
  const InputImageType *        inputImage = this->GetInputImage();
  const InputImageSpacingType & spacing    = inputImage->GetSpacing();

  /** Create a possibly mirrored version of x. */
  ContinuousIndexType xm = x;
  double              deriv_sign[ ImageDimension ];
  for( unsigned int dim = 0; dim < ImageDimension; dim++ )
  {
    deriv_sign[ dim ] = 1.0 / spacing[ dim ];
    if( x[ dim ] < this->m_StartIndex[ dim ] )
    {
      xm[ dim ]          = 2.0 * this->m_StartIndex[ dim ] - x[ dim ];
      deriv_sign[ dim ] *= -1.0;
    }
    if( x[ dim ] > this->m_EndIndex[ dim ] )
    {
      xm[ dim ]          = 2.0 * this->m_EndIndex[ dim ] - x[ dim ];
      deriv_sign[ dim ] *= -1.0;
    }

    /** Separately deal with cases on the image edge. */
    if( Math::FloatAlmostEqual( xm[ dim ], static_cast< ContinuousIndexValueType >( this->m_EndIndex[ dim ] ) ) )
    {
      xm[ dim ] -= 0.000001;
    }
  }
  // if this is mirrored again outside the image domain, then too bad.

  /**
   * Compute base index = closest index below point
   * Compute distance from point to base index
   */
  IndexType baseIndex;
  double    dist[ ImageDimension ];
  double    dinv[ ImageDimension ];
  for( unsigned int dim = 0; dim < ImageDimension; dim++ )
  {
    baseIndex[ dim ] = Math::Floor< IndexValueType >( xm[ dim ] );

    dist[ dim ] = xm[ dim ] - static_cast< double >( baseIndex[ dim ] );
    dinv[ dim ] = 1.0 - dist[ dim ];
  }

  /** Get the 8 corner values. */
  const RealType val000 = inputImage->GetPixel( baseIndex );
  ++baseIndex[ 0 ];
  const RealType val100 = inputImage->GetPixel( baseIndex );
  ++baseIndex[ 1 ];
  const RealType val110 = inputImage->GetPixel( baseIndex );
  ++baseIndex[ 2 ];
  const RealType val111 = inputImage->GetPixel( baseIndex );
  --baseIndex[ 1 ];
  const RealType val101 = inputImage->GetPixel( baseIndex );
  --baseIndex[ 0 ];
  const RealType val001 = inputImage->GetPixel( baseIndex );
  ++baseIndex[ 1 ];
  const RealType val011 = inputImage->GetPixel( baseIndex );
  --baseIndex[ 2 ];
  const RealType val010 = inputImage->GetPixel( baseIndex );

  /** Interpolate to get the value. */
  value = static_cast< OutputType >(
    val000 * dinv[ 0 ] * dinv[ 1 ] * dinv[ 2 ]
    + val100 * dist[ 0 ] * dinv[ 1 ] * dinv[ 2 ]
    + val010 * dinv[ 0 ] * dist[ 1 ] * dinv[ 2 ]
    + val001 * dinv[ 0 ] * dinv[ 1 ] * dist[ 2 ]
    + val110 * dist[ 0 ] * dist[ 1 ] * dinv[ 2 ]
    + val011 * dinv[ 0 ] * dist[ 1 ] * dist[ 2 ]
    + val101 * dist[ 0 ] * dinv[ 1 ] * dist[ 2 ]
    + val111 * dist[ 0 ] * dist[ 1 ] * dist[ 2 ] );

  /** Interpolate to get the derivative. */
  deriv[ 0 ] = deriv_sign[ 0 ]
    * ( dinv[ 1 ] * dinv[ 2 ] * ( val100 - val000 )
    + dist[ 1 ] * dinv[ 2 ] * ( val110 - val010 )
    + dinv[ 1 ] * dist[ 2 ] * ( val101 - val001 )
    + dist[ 1 ] * dist[ 2 ] * ( val111 - val011 )
    );
  deriv[ 1 ] = deriv_sign[ 1 ]
    * ( dinv[ 0 ] * dinv[ 2 ] * ( val010 - val000 )
    + dist[ 0 ] * dinv[ 2 ] * ( val110 - val100 )
    + dinv[ 0 ] * dist[ 2 ] * ( val011 - val001 )
    + dist[ 0 ] * dist[ 2 ] * ( val111 - val101 )
    );
  deriv[ 2 ] = deriv_sign[ 2 ]
    * ( dinv[ 0 ] * dinv[ 1 ] * ( val001 - val000 )
    + dist[ 0 ] * dinv[ 1 ] * ( val101 - val100 )
    + dinv[ 0 ] * dist[ 1 ] * ( val011 - val010 )
    + dist[ 0 ] * dist[ 1 ] * ( val111 - val110 )
    );

  /** Take direction cosines into account. */
  CovariantVectorType orientedDerivative;
  inputImage->TransformLocalVectorToPhysicalVector( deriv, orientedDerivative );
  deriv = orientedDerivative;

} // end EvaluateValueAndDerivativeOptimized()


} // end namespace itk

#endif
