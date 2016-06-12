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

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkReducedDimensionBSplineInterpolateImageFunction.txx,v $
  Language:  C++
  Date:      $Date: 2008-11-10 16:55:00 $
  Version:   $Revision: 1.21 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkReducedDimensionBSplineInterpolateImageFunction_hxx
#define __itkReducedDimensionBSplineInterpolateImageFunction_hxx

#include "itkReducedDimensionBSplineInterpolateImageFunction.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

#include "itkVector.h"

#include "itkMatrix.h"
#include "vnl/vnl_matrix_ref.h"

namespace itk
{

/**
 * Constructor
 */
template< class TImageType, class TCoordRep, class TCoefficientType >
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::ReducedDimensionBSplineInterpolateImageFunction()
{
  m_SplineOrder = 0;
  unsigned int SplineOrder = 1;
  m_CoefficientFilter = CoefficientFilter::New();
  // ***TODO: Should we store coefficients in a variable or retrieve from filter?
  m_Coefficients = CoefficientImageType::New();
  this->SetSplineOrder( SplineOrder );
  this->m_UseImageDirection = true;
}


/**
 * Standard "PrintSelf" method
 */
template< class TImageType, class TCoordRep, class TCoefficientType >
void
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::PrintSelf(
  std::ostream & os,
  Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Spline Order: " << m_SplineOrder << std::endl;
  os << indent << "UseImageDirection = "
     << ( this->m_UseImageDirection ? "On" : "Off" ) << std::endl;

}


template< class TImageType, class TCoordRep, class TCoefficientType >
void
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::SetInputImage( const TImageType * inputData )
{
  if( inputData )
  {
    m_CoefficientFilter->SetInput( inputData );

    // the Coefficient Filter requires that the spline order and the input data be set.
    // TODO:  We need to ensure that this is only run once and only after both input and
    //        spline order have been set. Should we force an update after the
    //        splineOrder has been set also?

    m_CoefficientFilter->Update();
    m_Coefficients = m_CoefficientFilter->GetOutput();

    // Call the Superclass implementation after, in case the filter
    // pulls in  more of the input image
    Superclass::SetInputImage( inputData );

    m_DataLength = inputData->GetBufferedRegion().GetSize();
  }
  else
  {
    m_Coefficients = NULL;
  }
}


template< class TImageType, class TCoordRep, class TCoefficientType >
void
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::SetSplineOrder( unsigned int SplineOrder )
{
  if( SplineOrder == m_SplineOrder )
  {
    return;
  }

  m_SplineOrder = SplineOrder;
  m_CoefficientFilter->SetSplineOrder( SplineOrder );
  // Set spline order of coefficient filter for last dimension to zero,
  // to use nearest neighbour interpolation in the last dimension.
  m_CoefficientFilter->SetSplineOrder( ImageDimension - 1, 0 );

  //this->SetPoles();
  m_MaxNumberInterpolationPoints = 1;
  for( unsigned int n = 0; n < ImageDimension - 1; n++ )
  {
    m_MaxNumberInterpolationPoints *= ( m_SplineOrder + 1 );
  }
  this->GeneratePointsToIndex();
}


template< class TImageType, class TCoordRep, class TCoefficientType >
typename
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::OutputType
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::EvaluateAtContinuousIndex( const ContinuousIndexType & x ) const
{
  /** Allocate memory on the stack: */
  const unsigned int maxSplineOrder = 5;
  const unsigned int maxMatrixSize  = ( ImageDimension - 1 ) * ( maxSplineOrder + 1 );
  long               evaluateIndexData[ maxMatrixSize ];
  double             weightsData[ maxMatrixSize ];

  vnl_matrix_ref< long > EvaluateIndex( ImageDimension - 1, ( m_SplineOrder + 1 ), evaluateIndexData );

  // compute the interpolation indexes
  this->DetermineRegionOfSupport( EvaluateIndex, x, m_SplineOrder );

  // Determine weights
  vnl_matrix_ref< double > weights( ImageDimension - 1, ( m_SplineOrder + 1 ), weightsData );

  SetInterpolationWeights( x, EvaluateIndex, weights, m_SplineOrder );

  // Modify EvaluateIndex at the boundaries using mirror boundary conditions
  this->ApplyMirrorBoundaryConditions( EvaluateIndex, m_SplineOrder );

  // perform interpolation
  double    interpolated = 0.0;
  IndexType coefficientIndex;
  coefficientIndex[ ImageDimension - 1 ] = vnl_math_rnd( x[ ImageDimension - 1 ] );

  // Step through eachpoint in the N-dimensional interpolation cube.
  for( unsigned int p = 0; p < m_MaxNumberInterpolationPoints; p++ )
  {
    // translate each step into the N-dimensional index.
    //      IndexType pointIndex = PointToIndex( p );

    double w = 1.0;
    for( unsigned int n = 0; n < ImageDimension - 1; n++ )
    {
      w                    *= weights[ n ][ m_PointsToIndex[ p ][ n ] ];
      coefficientIndex[ n ] = EvaluateIndex[ n ][ m_PointsToIndex[ p ][ n ] ]; // Build up ND index for coefficients.
    }
    // Convert our step p to the appropriate point in ND space in the
    // m_Coefficients cube.
    interpolated += w * m_Coefficients->GetPixel( coefficientIndex );
  }
  return ( interpolated );

}


template< class TImageType, class TCoordRep, class TCoefficientType >
typename
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::CovariantVectorType
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::EvaluateDerivativeAtContinuousIndex( const ContinuousIndexType & x ) const
{
  /** Allocate memory on the stack: */
  const unsigned int maxSplineOrder = 5;
  const unsigned int maxMatrixSize  = ( ImageDimension - 1 ) * ( maxSplineOrder + 1 );
  long               evaluateIndexData[ maxMatrixSize ];
  double             weightsData[ maxMatrixSize ];
  double             weightsDerivativeData[ maxMatrixSize ];

  vnl_matrix_ref< long > EvaluateIndex( ImageDimension - 1, ( m_SplineOrder + 1 ), evaluateIndexData );

  // compute the interpolation indexes
  // TODO: Do we need to revisit region of support for the derivatives?
  this->DetermineRegionOfSupport( EvaluateIndex, x, m_SplineOrder );

  // Determine weights
  vnl_matrix_ref< double > weights( ImageDimension - 1, ( m_SplineOrder + 1 ), weightsData );

  SetInterpolationWeights( x, EvaluateIndex, weights, m_SplineOrder );

  vnl_matrix_ref< double > weightsDerivative( ImageDimension - 1, ( m_SplineOrder + 1 ), weightsDerivativeData );
  SetDerivativeWeights( x, EvaluateIndex, weightsDerivative, ( m_SplineOrder ) );

  // Modify EvaluateIndex at the boundaries using mirror boundary conditions
  this->ApplyMirrorBoundaryConditions( EvaluateIndex, m_SplineOrder );

  const InputImageType * inputImage = this->GetInputImage();
  const typename InputImageType::SpacingType & spacing = inputImage->GetSpacing();

  // Calculate derivative
  CovariantVectorType derivativeValue;
  derivativeValue[ ImageDimension - 1 ] = static_cast< OutputType >( 0.0 );
  double    tempValue;
  IndexType coefficientIndex;
  coefficientIndex[ ImageDimension - 1 ] = vnl_math_rnd( x[ ImageDimension - 1 ] );
  for( unsigned int n = 0; n < ImageDimension - 1; n++ )
  {
    derivativeValue[ n ] = 0.0;
    for( unsigned int p = 0; p < m_MaxNumberInterpolationPoints; p++ )
    {
      tempValue = 1.0;
      for( unsigned int n1 = 0; n1 < ImageDimension - 1; n1++ )
      {
        coefficientIndex[ n1 ] = EvaluateIndex[ n1 ][ m_PointsToIndex[ p ][ n1 ] ];

        if( n1 == n )
        {
          //w *= weights[ n ][ m_PointsToIndex[ p ][ n ] ];
          tempValue *= weightsDerivative[ n1 ][ m_PointsToIndex[ p ][ n1 ] ];
        }
        else
        {
          tempValue *= weights[ n1 ][ m_PointsToIndex[ p ][ n1 ] ];
        }
      }
      derivativeValue[ n ] += m_Coefficients->GetPixel( coefficientIndex ) * tempValue;
    }
    derivativeValue[ n ] /= spacing[ n ];  // take spacing into account
  }

  if( this->m_UseImageDirection )
  {
    CovariantVectorType orientedDerivative;
    inputImage->TransformLocalVectorToPhysicalVector( derivativeValue, orientedDerivative );
    return orientedDerivative;
  }

  return derivativeValue;
}


template< class TImageType, class TCoordRep, class TCoefficientType >
void
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::SetInterpolationWeights( const ContinuousIndexType & x, const vnl_matrix< long > & EvaluateIndex,
  vnl_matrix< double > & weights, unsigned int splineOrder ) const
{
  // For speed improvements we could make each case a separate function and use
  // function pointers to reference the correct weight order.
  // Left as is for now for readability.
  double w, w2, w4, t, t0, t1;

  switch( splineOrder )
  {
    case 3:
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        w                 = x[ n ] - (double)EvaluateIndex[ n ][ 1 ];
        weights[ n ][ 3 ] = ( 1.0 / 6.0 ) * w * w * w;
        weights[ n ][ 0 ] = ( 1.0 / 6.0 ) + 0.5 * w * ( w - 1.0 ) - weights[ n ][ 3 ];
        weights[ n ][ 2 ] = w + weights[ n ][ 0 ] - 2.0 * weights[ n ][ 3 ];
        weights[ n ][ 1 ] = 1.0 - weights[ n ][ 0 ] - weights[ n ][ 2 ] - weights[ n ][ 3 ];
      }
      break;
    case 0:
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        weights[ n ][ 0 ] = 1; // implements nearest neighbor
      }
      break;
    case 1:
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        w                 = x[ n ] - (double)EvaluateIndex[ n ][ 0 ];
        weights[ n ][ 1 ] = w;
        weights[ n ][ 0 ] = 1.0 - w;
      }
      break;
    case 2:
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        /* x */
        w                 = x[ n ] - (double)EvaluateIndex[ n ][ 1 ];
        weights[ n ][ 1 ] = 0.75 - w * w;
        weights[ n ][ 2 ] = 0.5 * ( w - weights[ n ][ 1 ] + 1.0 );
        weights[ n ][ 0 ] = 1.0 - weights[ n ][ 1 ] - weights[ n ][ 2 ];
      }
      break;
    case 4:
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        /* x */
        w                  = x[ n ] - (double)EvaluateIndex[ n ][ 2 ];
        w2                 = w * w;
        t                  = ( 1.0 / 6.0 ) * w2;
        weights[ n ][ 0 ]  = 0.5 - w;
        weights[ n ][ 0 ] *= weights[ n ][ 0 ];
        weights[ n ][ 0 ] *= ( 1.0 / 24.0 ) * weights[ n ][ 0 ];
        t0                 = w * ( t - 11.0 / 24.0 );
        t1                 = 19.0 / 96.0 + w2 * ( 0.25 - t );
        weights[ n ][ 1 ]  = t1 + t0;
        weights[ n ][ 3 ]  = t1 - t0;
        weights[ n ][ 4 ]  = weights[ n ][ 0 ] + t0 + 0.5 * w;
        weights[ n ][ 2 ]  = 1.0 - weights[ n ][ 0 ] - weights[ n ][ 1 ] - weights[ n ][ 3 ] - weights[ n ][ 4 ];
      }
      break;
    case 5:
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        /* x */
        w                 = x[ n ] - (double)EvaluateIndex[ n ][ 2 ];
        w2                = w * w;
        weights[ n ][ 5 ] = ( 1.0 / 120.0 ) * w * w2 * w2;
        w2               -= w;
        w4                = w2 * w2;
        w                -= 0.5;
        t                 = w2 * ( w2 - 3.0 );
        weights[ n ][ 0 ] = ( 1.0 / 24.0 ) * ( 1.0 / 5.0 + w2 + w4 ) - weights[ n ][ 5 ];
        t0                = ( 1.0 / 24.0 ) * ( w2 * ( w2 - 5.0 ) + 46.0 / 5.0 );
        t1                = ( -1.0 / 12.0 ) * w * ( t + 4.0 );
        weights[ n ][ 2 ] = t0 + t1;
        weights[ n ][ 3 ] = t0 - t1;
        t0                = ( 1.0 / 16.0 ) * ( 9.0 / 5.0 - t );
        t1                = ( 1.0 / 24.0 ) * w * ( w4 - w2 - 5.0 );
        weights[ n ][ 1 ] = t0 + t1;
        weights[ n ][ 4 ] = t0 - t1;
      }
      break;
    default:
      // SplineOrder not implemented yet.
      ExceptionObject err( __FILE__, __LINE__ );
      err.SetLocation( ITK_LOCATION );
      err.SetDescription( "SplineOrder must be between 0 and 5. Requested spline order has not been implemented yet." );
      throw err;
      break;
  }

}


template< class TImageType, class TCoordRep, class TCoefficientType >
void
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::SetDerivativeWeights( const ContinuousIndexType & x, const vnl_matrix< long > & EvaluateIndex,
  vnl_matrix< double > & weights, unsigned int splineOrder ) const
{
  // For speed improvements we could make each case a separate function and use
  // function pointers to reference the correct weight order.
  // Another possiblity would be to loop inside the case statement (reducing the number
  // of switch statement executions to one per routine call.
  // Left as is for now for readability.
  double w, w1, w2, w3, w4, w5, t, t0, t1, t2;
  int    derivativeSplineOrder = (int)splineOrder - 1;

  switch( derivativeSplineOrder )
  {

    // Calculates B(splineOrder) ( (x + 1/2) - xi) - B(splineOrder -1) ( (x - 1/2) - xi)
    case -1:
      // Why would we want to do this?
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        weights[ n ][ 0 ] = 0.0;
      }
      break;
    case 0:
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        weights[ n ][ 0 ] = -1.0;
        weights[ n ][ 1 ] =  1.0;
      }
      break;
    case 1:
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        w = x[ n ] + 0.5 - (double)EvaluateIndex[ n ][ 1 ];
        // w2 = w;
        w1 = 1.0 - w;

        weights[ n ][ 0 ] = 0.0 - w1;
        weights[ n ][ 1 ] = w1 - w;
        weights[ n ][ 2 ] = w;
      }
      break;
    case 2:

      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        w  = x[ n ] + .5 - (double)EvaluateIndex[ n ][ 2 ];
        w2 = 0.75 - w * w;
        w3 = 0.5 * ( w - w2 + 1.0 );
        w1 = 1.0 - w2 - w3;

        weights[ n ][ 0 ] = 0.0 - w1;
        weights[ n ][ 1 ] = w1 - w2;
        weights[ n ][ 2 ] = w2 - w3;
        weights[ n ][ 3 ] = w3;
      }
      break;
    case 3:

      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        w  = x[ n ] + 0.5 - (double)EvaluateIndex[ n ][ 2 ];
        w4 = ( 1.0 / 6.0 ) * w * w * w;
        w1 = ( 1.0 / 6.0 ) + 0.5 * w * ( w - 1.0 ) - w4;
        w3 = w + w1 - 2.0 * w4;
        w2 = 1.0 - w1 - w3 - w4;

        weights[ n ][ 0 ] = 0.0 - w1;
        weights[ n ][ 1 ] = w1 - w2;
        weights[ n ][ 2 ] = w2 - w3;
        weights[ n ][ 3 ] = w3 - w4;
        weights[ n ][ 4 ] = w4;
      }
      break;
    case 4:
      for( unsigned int n = 0; n < ImageDimension - 1; n++ )
      {
        w   = x[ n ] + .5 - (double)EvaluateIndex[ n ][ 3 ];
        t2  = w * w;
        t   = ( 1.0 / 6.0 ) * t2;
        w1  = 0.5 - w;
        w1 *= w1;
        w1 *= ( 1.0 / 24.0 ) * w1;
        t0  = w * ( t - 11.0 / 24.0 );
        t1  = 19.0 / 96.0 + t2 * ( 0.25 - t );
        w2  = t1 + t0;
        w4  = t1 - t0;
        w5  = w1 + t0 + 0.5 * w;
        w3  = 1.0 - w1 - w2 - w4 - w5;

        weights[ n ][ 0 ] = 0.0 - w1;
        weights[ n ][ 1 ] = w1 - w2;
        weights[ n ][ 2 ] = w2 - w3;
        weights[ n ][ 3 ] = w3 - w4;
        weights[ n ][ 4 ] = w4 - w5;
        weights[ n ][ 5 ] = w5;
      }
      break;

    default:
      // SplineOrder not implemented yet.
      ExceptionObject err( __FILE__, __LINE__ );
      err.SetLocation( ITK_LOCATION );
      err.SetDescription( "SplineOrder (for derivatives) must be between 1 and 5. Requested spline order has not been implemented yet." );
      throw err;
      break;
  }

}


// Generates m_PointsToIndex;
template< class TImageType, class TCoordRep, class TCoefficientType >
void
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::GeneratePointsToIndex()
{
  // m_PointsToIndex is used to convert a sequential location to an N-dimension
  // index vector.  This is precomputed to save time during the interpolation routine.
  m_PointsToIndex.resize( m_MaxNumberInterpolationPoints );
  for( unsigned int p = 0; p < m_MaxNumberInterpolationPoints; p++ )
  {
    int           pp = p;
    unsigned long indexFactor[ ImageDimension - 1 ];
    indexFactor[ 0 ] = 1;
    for( int j = 1; j < static_cast< int >( ImageDimension - 1 ); j++ )
    {
      indexFactor[ j ] = indexFactor[ j - 1 ] * ( m_SplineOrder + 1 );
    }
    for( int j = ( static_cast< int >( ImageDimension ) - 2 ); j >= 0; j-- )
    {
      m_PointsToIndex[ p ][ j ] = pp / indexFactor[ j ];
      pp                        = pp % indexFactor[ j ];
    }
  }
}


template< class TImageType, class TCoordRep, class TCoefficientType >
void
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::DetermineRegionOfSupport( vnl_matrix< long > & evaluateIndex,
  const ContinuousIndexType & x,
  unsigned int splineOrder ) const
{
  long indx;

// compute the interpolation indexes
  for( unsigned int n = 0; n < ImageDimension - 1; n++ )
  {
    if( splineOrder & 1 )     // Use this index calculation for odd splineOrder
    {
      indx = (long)vcl_floor( (float)x[ n ] ) - splineOrder / 2;
      for( unsigned int k = 0; k <= splineOrder; k++ )
      {
        evaluateIndex[ n ][ k ] = indx++;
      }
    }
    else                       // Use this index calculation for even splineOrder
    {
      indx = (long)vcl_floor( (float)( x[ n ] + 0.5 ) ) - splineOrder / 2;
      for( unsigned int k = 0; k <= splineOrder; k++ )
      {
        evaluateIndex[ n ][ k ] = indx++;
      }
    }
  }
}


template< class TImageType, class TCoordRep, class TCoefficientType >
void
ReducedDimensionBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType >
::ApplyMirrorBoundaryConditions( vnl_matrix< long > & evaluateIndex,
  unsigned int splineOrder ) const
{
  for( unsigned int n = 0; n < ImageDimension - 1; n++ )
  {
    long dataLength2 = 2 * m_DataLength[ n ] - 2;

    // apply the mirror boundary conditions
    // TODO:  We could implement other boundary options beside mirror
    if( m_DataLength[ n ] == 1 )
    {
      for( unsigned int k = 0; k <= splineOrder; k++ )
      {
        evaluateIndex[ n ][ k ] = 0;
      }
    }
    else
    {
      for( unsigned int k = 0; k <= splineOrder; k++ )
      {
        // btw - Think about this couldn't this be replaced with a more elagent modulus method?
        evaluateIndex[ n ][ k ]
          = ( evaluateIndex[ n ][ k ] < 0L ) ? ( -evaluateIndex[ n ][ k ] - dataLength2 * ( ( -evaluateIndex[ n ][ k ] ) / dataLength2 ) )
          : ( evaluateIndex[ n ][ k ] - dataLength2 * ( evaluateIndex[ n ][ k ] / dataLength2 ) );
        if( (long)m_DataLength[ n ] <= evaluateIndex[ n ][ k ] )
        {
          evaluateIndex[ n ][ k ] = dataLength2 - evaluateIndex[ n ][ k ];
        }
      }
    }
  }
}


} // namespace itk

#endif
