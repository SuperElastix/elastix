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
#ifndef __itkRecursiveBSplineInterpolateImageFunction_hxx
#define __itkRecursiveBSplineInterpolateImageFunction_hxx

#include "itkRecursiveBSplineInterpolateImageFunction.h"
#include "itkRecursiveBSplineImplementation.h"

#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "itkMatrix.h"
//#include "emm_vec.hxx"

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

  this->m_RecursiveBSplineWeightFunction = RecursiveBSplineWeightFunctionType::New();
  this->m_Kernel = KernelType::New();
  this->m_DerivativeKernel = DerivativeKernelType::New();
  this->m_SecondOrderDerivativeKernel = SecondOrderDerivativeKernelType::New();

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
    this->m_Coefficients = this->m_CoefficientFilter->GetOutput();
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
 * ******************* EvaluateHessian ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >::MatrixType
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateHessian( const PointType & point ) const
{
  ContinuousIndexType cindex;
  this->GetInputImage()->TransformPhysicalPointToContinuousIndex( point, cindex );
  return this->EvaluateHessianAtContinuousIndex( cindex );
} // end EvaluateHessian()


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
 * ******************* EvaluateValueAndDerivativeAndHessian ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateValueAndDerivativeAndHessian( const PointType & point, OutputType & value, CovariantVectorType & deriv, MatrixType & hessian ) const
{
  ContinuousIndexType cindex;
  this->GetInputImage()->TransformPhysicalPointToContinuousIndex( point, cindex );
  this->EvaluateValueAndDerivativeAndHessianAtContinuousIndex( cindex, value, deriv, hessian );
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
  /** Define some constants. */
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  long evaluateIndexData[ numberOfWeights ];
  OffsetValueType stepsData[ numberOfWeights ];
  vnl_matrix_ref<long> evaluateIndex(ImageDimension,SplineOrder+1,evaluateIndexData);
  OffsetValueType * steps = &(stepsData[0]);

  typename WeightsType::ValueType weightsArray1D[ numberOfWeights ];
  WeightsType weights1D( weightsArray1D, numberOfWeights, false );
  typename WeightsType::ValueType * weightsPtr = &weights1D[0];

  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( x, weights1D, supportIndex );

  // Compute the interpolation indexes
  this->DetermineRegionOfSupport( evaluateIndex, x );

  // Modify evaluateIndex at the boundaries using mirror boundary conditions
  this->ApplyMirrorBoundaryConditions( evaluateIndex );

  // MS: should we store steps in a member variable for later use?
  //Calculate steps for image pointer
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * this->m_OffsetTable[ n ];
    }
  }

  TCoefficientType * coefficientPointer = const_cast< TCoefficientType* >( this->m_Coefficients->GetBufferPointer() );
  OutputType interpolated = RecursiveBSplineImplementation_GetSample< OutputType, ImageDimension, SplineOrder, OutputType*, USE_STEPS >
                ::GetSample(coefficientPointer, steps, weightsPtr );

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
  CovariantVectorType & derivative) const
{
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  long evaluateIndexData[ numberOfWeights ];
  OffsetValueType stepsData[ numberOfWeights ];

  vnl_matrix_ref<long> evaluateIndex( ImageDimension, SplineOrder + 1, evaluateIndexData );
  OffsetValueType * steps = &(stepsData[0]);

  /** Create storage for the B-spline interpolation weights. */
  typename WeightsType::ValueType weightsArray1D[ numberOfWeights ];
  WeightsType weights1D( weightsArray1D, numberOfWeights, false );

  typename WeightsType::ValueType derivativeWeightsArray1D[ numberOfWeights ];
  WeightsType derivativeWeights1D( derivativeWeightsArray1D, numberOfWeights, false );

  double * weightsPointer = &(weights1D[0]);
  double * derivativeWeightsPointer = &(derivativeWeights1D[0]);

  // Compute the interpolation indexes
  this->DetermineRegionOfSupport( evaluateIndex, x );

  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( x, weights1D, supportIndex );
  this->m_RecursiveBSplineWeightFunction->EvaluateDerivative( x, derivativeWeights1D, supportIndex );

  // Modify EvaluateIndex at the boundaries using mirror boundary conditions
  this->ApplyMirrorBoundaryConditions( evaluateIndex );

  // Calculate steps for coefficients pointer
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * this->m_OffsetTable[ n ];
    }
  }

  // Call recursive sampling function
  OutputType derivativeValue[ ImageDimension + 1 ];

  /** Recursively compute the spatial Jacobian. */
  TCoefficientType * coefficientPointer = const_cast< TCoefficientType* >( this->m_Coefficients->GetBufferPointer() );
  RecursiveBSplineImplementation_GetSpatialJacobian< OutputType*, ImageDimension, SplineOrder, OutputType*, USE_STEPS >
            ::GetSpatialJacobian( &derivativeValue[0], coefficientPointer, steps, weightsPointer, derivativeWeightsPointer  );


  // Extract the interpolated value and the derivative from the derivativeValue
  // vector. Element 0 contains the value, element 1 to ImageDimension+1 contains
  // the derivative in each dimension.
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    derivative[ n ] = derivativeValue[ n + 1 ] / this->m_Spacing[ n ];
  }

  /** Assign value and derivative. */
  value = derivativeValue[ 0 ];
  const InputImageType *inputImage = this->GetInputImage();
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
  OutputType value;
  CovariantVectorType derivative;
  this->EvaluateValueAndDerivativeAtContinuousIndex( x, value, derivative );

  return derivative;
} // end EvaluateDerivativeAtContinuousIndex()

/**
 * ******************* EvaluateValueAndDerivativeAndHessianAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateValueAndDerivativeAndHessianAtContinuousIndex(
  const ContinuousIndexType & x,
  OutputType & value,
  CovariantVectorType & derivative,
  MatrixType & sh) const
{
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  long evaluateIndexData[ numberOfWeights ];
  long stepsData[ numberOfWeights ];

  vnl_matrix_ref<long> evaluateIndex( ImageDimension, SplineOrder + 1, evaluateIndexData );
  long * steps = &(stepsData[0]);

  /** Create storage for the B-spline interpolation weights. */
  typename WeightsType::ValueType weightsArray1D[ numberOfWeights ];
  WeightsType weights1D( weightsArray1D, numberOfWeights, false );

  typename WeightsType::ValueType derivativeWeightsArray1D[ numberOfWeights ];
  WeightsType derivativeWeights1D( derivativeWeightsArray1D, numberOfWeights, false );

  typename WeightsType::ValueType hessianWeightsArray1D[ numberOfWeights ];
  WeightsType hessianWeights1D( hessianWeightsArray1D, numberOfWeights, false);

  double * weightsPointer = &(weights1D[0]);
  double * derivativeWeightsPointer = &(derivativeWeights1D[0]);
  double * hessianWeightsPointer = &(hessianWeights1D[0]);

  // Compute the interpolation indexes
  this->DetermineRegionOfSupport( evaluateIndex, x );

  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( x, weights1D, supportIndex );
  this->m_RecursiveBSplineWeightFunction->EvaluateDerivative( x, derivativeWeights1D, supportIndex );
  this->m_RecursiveBSplineWeightFunction->EvaluateSecondOrderDerivative(x, hessianWeights1D, supportIndex );

  // Modify EvaluateIndex at the boundaries using mirror boundary conditions
  this->ApplyMirrorBoundaryConditions( evaluateIndex );

  // Calculate steps for coefficients pointer
  for( unsigned int n = 0; n < ImageDimension; ++n )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * this->m_OffsetTable[ n ];
    }
  }

  // Call recursive sampling function
  OutputType hessian[ ( ImageDimension + 1 ) * ( ImageDimension + 2 ) / 2 ];

  /** Recursively compute the spatial Jacobian. */
  TCoefficientType * coefficientPointer = const_cast< TCoefficientType* >( this->m_Coefficients->GetBufferPointer() );

  RecursiveBSplineImplementation_GetSpatialHessian< OutputType*, ImageDimension, SplineOrder, OutputType*, USE_STEPS >
            ::GetSpatialHessian( &hessian[0], coefficientPointer, steps, weightsPointer, derivativeWeightsPointer, hessianWeightsPointer );

/** Copy the correct elements to the spatial Hessian.
 *
 * Upon return sh contains the spatial Hessian, spatial Jacobian and transformpoint. With
 * Hk = [ transformPoint     spatialJacobian'
 *        spatialJacobian    spatialHessian   ] .
 * (Hk specifies all info of dimension (element) k (< OutputDimension) of the point
 * and spatialJacobian is a vector of the derivative of this point with respect to the dimensions.)
 * The i,j (both < SpaceDimension) element of Hk is stored in:
 * i<=j : sh[ k +  OutputDimension * (i + j*(j+1)/2 ) ]
 * i>=j : sh[ k +  OutputDimension * (j + i*(i+1)/2 ) ]
 */

  value = hessian [ 0 ];

  unsigned int k = 1;
  for(unsigned int i = 0; i < ImageDimension; ++i)
  {
      for(unsigned int j = 0; j < ImageDimension; ++j)
      {
          if( i < j )
          {
            sh[ i ][ j ] = hessian[ ( i + 1 ) + ( j + 1 )*( j + 2 ) / 2 ]
                    / this->m_Spacing[ i ];
          }
          else
          {
            sh[ i ][ j ] = hessian[ ( j + 1 ) + ( i + 1 )*( i + 2 ) / 2 ]
                    / this->m_Spacing[ i ];
          }
      }
      derivative[ i ] = hessian[ k ] / this->m_Spacing[ i ];
      k += ( i + 2 );
  }

  const InputImageType *inputImage = this->GetInputImage();
  if( this->m_UseImageDirection )
  {
    CovariantVectorType orientedDerivative;
    MatrixType orientedHessian = inputImage->GetDirection()*sh*inputImage->GetDirection();
    inputImage->TransformLocalVectorToPhysicalVector( derivative, orientedDerivative );

    derivative = orientedDerivative;
    sh = orientedHessian;
  }

} // end EvaluateValueAndDerivativeAndHessianAtContinuousIndex()

/**
 * ******************* EvaluateHessianAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::MatrixType
RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateHessianAtContinuousIndex(
  const ContinuousIndexType & x ) const
{
  OutputType value;
  CovariantVectorType derivative;
  MatrixType hessian;
  this->EvaluateValueAndDerivativeAndHessianAtContinuousIndex( x, value, derivative, hessian );

  return hessian;
}

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
