/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkRecursiveBSplineInterpolateImageFunction_hxx
#define __itkRecursiveBSplineInterpolateImageFunction_hxx

#include "itkRecursiveBSplineInterpolateImageFunction.h"

#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "itkMatrix.h"


namespace itk
{

/**
 * ******************* Constructor ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::RecursiveBSplineInterpolateImageFunction()
{
  this->m_UseImageDirection = true;

  /** Setup coefficient filter. */
  this->m_CoefficientFilter = CoefficientFilter::New();
  this->m_CoefficientFilter->SetSplineOrder( SplineOrder );
  this->m_Coefficients = CoefficientImageType::New();

  if( SplineOrder > 5 )
  {
    itkExceptionMacro( << "SplineOrder must be between 0 and 5. Requested spline order has not been implemented yet." );
  }
} // end Constructor()


/**
 * ******************* PrintSelf ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::PrintSelf(std::ostream & os,Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Spline Order: " << SplineOrder << std::endl;
  os << indent << "UseImageDirection = "
    << ( this->m_UseImageDirection ? "On" : "Off" ) << std::endl;
} // end PrintSelf()


/**
 * ******************* SetInputImage ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::SetInputImage( const TImageType *inputData )
{
  if( inputData )
  {
    Superclass::SetInputImage( inputData );

    this->m_CoefficientFilter->SetInput( inputData );
    this->m_CoefficientFilter->Update();
    this->m_Coefficients = m_CoefficientFilter->GetOutput();
    this->m_DataLength = inputData->GetBufferedRegion().GetSize();

    for( unsigned int n = 0; n < ImageDimension; ++n )
    {
      this->m_OffsetTable[n] = this->m_Coefficients->GetOffsetTable()[n];
    }

    this->m_Spacing = inputData->GetSpacing();
  }
  else
  {
    this->m_Coefficients = NULL;
  }
} // end SetInputImage()


/**
 * ******************* Evaluate ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >::OutputType
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::Evaluate( const PointType & point ) const
{
  ContinuousIndexType cindex;
  this->GetInputImage()->TransformPhysicalPointToContinuousIndex( point, cindex );
  return this->EvaluateAtContinuousIndex( cindex );
} // end Evaluate()


/**
 * ******************* EvaluateDerivative ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >::CovariantVectorType
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateDerivative( const PointType & point ) const
{
  ContinuousIndexType cindex;
  this->GetInputImage()->TransformPhysicalPointToContinuousIndex( point, cindex );
  return this->EvaluateDerivativeAtContinuousIndex( cindex );
} // end EvaluateDerivative()


/**
 * ******************* EvaluateValueAndDerivative ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateValueAndDerivative( const PointType & point, OutputType & value, CovariantVectorType & deriv ) const
{
  ContinuousIndexType cindex;
  this->GetInputImage()->TransformPhysicalPointToContinuousIndex( point, cindex );
  this->EvaluateValueAndDerivativeAtContinuousIndex( cindex, value, deriv );
} // end EvaluateValueAndDerivative()


/**
 * ******************* EvaluateAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::OutputType
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateAtContinuousIndex( const ContinuousIndexType & x ) const
{
  // Allocate memory on the stack
  // MS: use MaxNumberInterpolationPoints??
  //of: const unsigned int helper = ( SplineOrder + 1 ) * ImageDimension;
  long evaluateIndexData[(SplineOrder+1)*ImageDimension];
  long stepsData[(SplineOrder+1)*ImageDimension];
  double weightsData[(SplineOrder+1)*ImageDimension];
  vnl_matrix_ref<long> evaluateIndex(ImageDimension,SplineOrder+1,evaluateIndexData);
  double * weights = &(weightsData[0]);
  long * steps = &(stepsData[0]);

  // Compute the interpolation indexes
  this->DetermineRegionOfSupport( evaluateIndex, x );

  // Compute the B-spline weights
  SetInterpolationWeights( x, evaluateIndex, weights );

  // Modify evaluateIndex at the boundaries using mirror boundary conditions
  this->ApplyMirrorBoundaryConditions( evaluateIndex );

  OutputType interpolated = 0.0;

  // MS: should we store steps in a member variable for later use?
  //Calculate steps for image pointer
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * this->m_OffsetTable[ n ];
    }
  }

  //Call recursive sampling function
  interpolated = SampleFunction< ImageDimension, SplineOrder, TCoordRep >
    ::SampleValue( this->m_Coefficients->GetBufferPointer(), steps, weights );

  return interpolated;
} // end EvaluateAtContinuousIndex()


/**
 * ******************* EvaluateValueAndDerivativeAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateValueAndDerivativeAtContinuousIndex(
  const ContinuousIndexType & x,
  OutputType & value,
  CovariantVectorType & derivative ) const
{
  // MS: use MaxNumberInterpolationPoints??
  // Allocate memory on the stack
  long evaluateIndexData[(SplineOrder+1)*ImageDimension];
  long stepsData[(SplineOrder+1)*ImageDimension];
  double weightsData[(SplineOrder+1)*ImageDimension];
  double derivativeWeightsData[(SplineOrder+1)*ImageDimension];

  vnl_matrix_ref<long> evaluateIndex( ImageDimension, SplineOrder + 1, evaluateIndexData );
  double * weights = &(weightsData[0]);
  double * derivativeWeights = &(derivativeWeightsData[0]);
  long * steps = &(stepsData[0]);

  // Compute the interpolation indexes
  this->DetermineRegionOfSupport( evaluateIndex, x );

  // Compute the B-spline weights
  this->SetInterpolationWeights( x, evaluateIndex, weights );

  // Compute the B-spline derivative weights
  this->SetDerivativeWeights( x, evaluateIndex, derivativeWeights );

  // Modify EvaluateIndex at the boundaries using mirror boundary conditions
  this->ApplyMirrorBoundaryConditions( evaluateIndex );

  const InputImageType *inputImage = this->GetInputImage();

  // Calculate steps for coefficients pointer
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * this->m_OffsetTable[ n ];
    }
  }

  // Call recursive sampling function
  TCoordRep derivativeValue[ ImageDimension + 1 ];
  SampleFunction< ImageDimension, SplineOrder, TCoordRep >
    ::SampleValueAndDerivative( derivativeValue,
    this->m_Coefficients->GetBufferPointer(),
    steps,
    weights,
    derivativeWeights );

  // Extract the interpolated value and the derivative from the derivativeValue
  // vector. Element 0 contains the value, element 1 to ImageDimension+1 contains
  // the derivative in each dimension.
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    derivative[ n ] = derivativeValue[ n + 1 ] / this->m_Spacing[ n ];
  }

  /** Assign value and derivative. */
  value = derivativeValue[ 0 ];
  if( this->m_UseImageDirection )
  {
    CovariantVectorType orientedDerivative;
    inputImage->TransformLocalVectorToPhysicalVector( derivative, orientedDerivative );
    derivative = orientedDerivative;
  }
} // end EvaluateValueAndDerivativeAtContinuousIndex()


/**
 * ******************* EvaluateDerivativeAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::CovariantVectorType
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateDerivativeAtContinuousIndex( const ContinuousIndexType & x ) const
{
  // MS: We can avoid code duplication by letting this function call
  // EvaluateValueAndDerivativeAtContinuousIndex and then ignore the value
  // Would the performance penalty be large?

  // MS: use MaxNumberInterpolationPoints??
  // Allocate memory on the stack
  long evaluateIndexData[(SplineOrder+1)*ImageDimension];
  long stepsData[(SplineOrder+1)*ImageDimension];
  double weightsData[(SplineOrder+1)*ImageDimension];
  double derivativeWeightsData[(SplineOrder+1)*ImageDimension];

  vnl_matrix_ref<long> evaluateIndex( ImageDimension, SplineOrder + 1, evaluateIndexData );
  double * weights = &(weightsData[0]);
  double * derivativeWeights = &(derivativeWeightsData[0]);
  long * steps = &(stepsData[0]);

  // Compute the interpolation indexes
  this->DetermineRegionOfSupport( evaluateIndex, x );

  // Compute the B-spline weights
  this->SetInterpolationWeights( x, evaluateIndex, weights );

  // Compute the B-spline derivative weights
  this->SetDerivativeWeights( x, evaluateIndex, derivativeWeights );

  // Modify EvaluateIndex at the boundaries using mirror boundary conditions
  this->ApplyMirrorBoundaryConditions( evaluateIndex );

  const InputImageType *inputImage = this->GetInputImage();

  //Calculate steps for coefficients pointer
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * m_OffsetTable[ n ];
    }
  }

  // Call recursive sampling function. Since the value is computed almost for
  // free, both value and derivative are calculated.
  TCoordRep derivativeValue[ ImageDimension + 1 ];
  SampleFunction< ImageDimension, SplineOrder, TCoordRep >
    ::SampleValueAndDerivative( derivativeValue,
    this->m_Coefficients->GetBufferPointer(),
    steps,
    weights,
    derivativeWeights );

  CovariantVectorType derivative;

  // Extract the interpolated value and the derivative from the derivativeValue
  // vector. Element 0 contains the value, element 1 to ImageDimension+1 contains
  // the derivative in each dimension.
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    derivative[ n ] = derivativeValue[ n + 1 ] / this->m_Spacing[ n ];
  }

  if( this->m_UseImageDirection )
  {
    CovariantVectorType orientedDerivative;
    inputImage->TransformLocalVectorToPhysicalVector( derivative, orientedDerivative );
    return orientedDerivative;
  }

  return derivative;
} // end EvaluateDerivativeAtContinuousIndex()


/**
 * ******************* SetInterpolationWeights ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::SetInterpolationWeights(
  const ContinuousIndexType & x,
  const vnl_matrix< long > & evaluateIndex,
  double * weights ) const
{
  Vector< double, SplineOrder + 1 > weightsvec;
  const int idx = Math::Floor<int>( SplineOrder / 2.0 );

  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    weightsvec.Fill( 0.0 );

    double w = x[ n ] - (double)evaluateIndex[ n ][ idx ];
    BSplineWeights< SplineOrder, TCoefficientType >::GetWeights( weightsvec, w );
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      weights[ ( SplineOrder + 1 ) * n + k ] = weightsvec[ k ];
    }
  }
} // end SetInterpolationWeights()


/**
 * ******************* SetDerivativeWeights ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::SetDerivativeWeights(
  const ContinuousIndexType & x,
  const vnl_matrix< long > & evaluateIndex,
  double * weights ) const
{
  Vector< double, SplineOrder + 1 > weightsvec;
  const int idx = Math::Floor<int>( ( SplineOrder + 1 ) / 2.0 );

  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    weightsvec.Fill( 0.0 );
    const double w = x[ n ] - (double)evaluateIndex[ n ][ idx ] + 0.5;
    BSplineWeights< SplineOrder, TCoefficientType >::GetDerivativeWeights( weightsvec, w );

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      weights[ ( SplineOrder + 1 ) * n + k ] = weightsvec[ k ];
    }
  }
} // end SetDerivativeWeights()


/**
 * ******************* SetHessianWeights ***********************
 */

//template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
//void
//RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, splineOrder >
//::SetHessianWeights(const ContinuousIndexType & x,
//                    const vnl_matrix< long > & evaluateIndex,
//                    double * weights) const
//{
//    itk::Vector<double, splineOrder+1> weightsvec;
//    weightsvec.Fill( 0.0 );

//    for ( unsigned int n = 0; n < ImageDimension; n++ )
//    {
//        int idx = floor( splineOrder / 2.0 );//FIX
//        double w = x[n] - (double)evaluateIndex[n][idx];
//        this->m_BSplineWeightInstance->getHessianWeights(weightsvec, w);
//        for(unsigned int k = 0; k <= splineOrder; ++k)
//        {
//            weights[(splineOrder+1)*n+k] = weightsvec[k];
//        }
//        weightsvec.Fill( 0.0 );
//    }
//}


/**
 * ******************* DetermineRegionOfSupport ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::DetermineRegionOfSupport(vnl_matrix< long > & evaluateIndex, const ContinuousIndexType & x) const
{
  const float halfOffset = SplineOrder & 1 ? 0.0 : 0.5;
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    long indx = Math::Floor<long>( (float)x[ n ] + halfOffset ) - SplineOrder / 2;
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      evaluateIndex[ n ][ k ] = indx++;
    }
  }
} // end DetermineRegionOfSupport()


/**
 * ******************* ApplyMirrorBoundaryConditions ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::ApplyMirrorBoundaryConditions( vnl_matrix< long > & evaluateIndex ) const
{
  const IndexType startIndex = this->GetStartIndex();
  const IndexType endIndex = this->GetEndIndex();

  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    // apply the mirror boundary conditions
    // TODO:  We could implement other boundary options beside mirror
    if( m_DataLength[n] == 1 )
    {
      for( unsigned int k = 0; k <= SplineOrder; ++k )
      {
        evaluateIndex[ n ][ k ] = 0;
      }
    }
    else
    {
      for( unsigned int k = 0; k <= SplineOrder; ++k )
      {
        if( evaluateIndex[n][k] < startIndex[n] )
        {
          evaluateIndex[n][k] = startIndex[n] +
            ( startIndex[n] - evaluateIndex[n][k] );
        }
        if( evaluateIndex[n][k] >= endIndex[n] )
        {
          evaluateIndex[n][k] = endIndex[n] -
            ( evaluateIndex[n][k] - endIndex[n] );
        }
      }
    }
  }
} // end ApplyMirrorBoundaryConditions()


} // namespace itk

#endif
