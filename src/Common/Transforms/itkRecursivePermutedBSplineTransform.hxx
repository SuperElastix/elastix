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
#ifndef __itkRecursivePermutedBSplineTransform_hxx
#define __itkRecursivePermutedBSplineTransform_hxx

#include "itkRecursivePermutedBSplineTransform.h"


#include "itkRecursiveBSplineImplementation.h"

#include "emm_vec.hxx"
// Requires emm_vec from DPoot.
// NOTE: strangely enough, enabling 'whole program optimization' (configuration properties, c++, optimization) makes this version substantially (40%) slower.
//#define RECURSIVEBSPLINE_FORWARDTRANSFORMPOINT // if defined forwards the transformpoint to transformpoints to minimize code duplication.
// (Currently  much slower; did not yet investigate why)

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template <typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
RecursivePermutedBSplineTransform< TScalar, NDimensions, VSplineOrder >
::RecursivePermutedBSplineTransform() : Superclass()
{
  this->m_RecursiveBSplineWeightFunction = RecursiveBSplineWeightFunctionType::New();
  this->m_Kernel = KernelType::New();
  this->m_DerivativeKernel = DerivativeKernelType::New();
  this->m_SecondOrderDerivativeKernel = SecondOrderDerivativeKernelType::New();
} // end Constructor()


/**
 * ********************* TransformPoint ****************************
 */

template< typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
typename RecursivePermutedBSplineTransform<TScalar, NDimensions, VSplineOrder>
::OutputPointType
RecursivePermutedBSplineTransform<TScalar, NDimensions, VSplineOrder>
::TransformPoint( const InputPointType & point ) const
{
#ifdef RECURSIVEBSPLINE_FORWARDTRANSFORMPOINT
   std::vector< InputPointType > pointListIn( 1 );
   std::vector< OutputPointType > pointListOut( 1 );
   pointListIn[0] = point;
   this->TransformPoints( pointListIn, pointListOut );
   return pointListOut[0];

#else
   // Or with duplicate code:
   /** Define some constants. */
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  const unsigned int numberOfIndices = RecursiveBSplineWeightFunctionType::NumberOfIndices;

  /** Initialize output point. */
  OutputPointType outputPoint;
 // outputPoint.Fill( NumericTraits<TScalar>::Zero ); // not needed since it is always (re)initialized later.

  /** Allocate weights on the stack: */
  typename WeightsType::ValueType weightsArray1D[ numberOfWeights ];
  WeightsType weights1D( weightsArray1D, numberOfWeights, false );

  /** Check if the coefficient image has been set. */
  if( !this->m_CoefficientImages[ 0 ] )
  {
    itkWarningMacro( << "B-spline coefficients have not been set" );
    for( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      outputPoint[ j ] = point[ j ];
    }
    return outputPoint;
  }

  /** Convert to continuous index. */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( point, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  bool inside = this->InsideValidRegion( cindex );
  if( !inside )
  {
  //std::cerr << "." ;
  //this->num_outside_valid++;
    outputPoint = point;
    return outputPoint;
  }

  // Compute interpolation weighs and store them in weights1D
  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights1D, supportIndex );


  /** Initialize (helper) variables. */
  const OffsetValueType * bsplineOffsetTable = this->m_CoefficientImages[ 0 ]->GetOffsetTable();
  OffsetValueType totalOffsetToSupportIndex = 0;
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    totalOffsetToSupportIndex += supportIndex[ j ] * bsplineOffsetTable[ j ];
  }
  //ScalarType displacement[ SpaceDimension ];

  ScalarType* parameterPointer = const_cast< PixelType * >( ( this->m_InputParametersPointer->data_block() ) );
  typename WeightsType::ValueType * weightsPtr = & weights1D[0];

  typedef vec<ScalarType, SpaceDimension> vecPointType;
  typedef vecptr< ScalarType * , SpaceDimension> vecPointerType;
  vecPointerType mu( parameterPointer + totalOffsetToSupportIndex* SpaceDimension );
  vecPointType displacement = RecursiveBSplineImplementation_GetSample< vecPointType, SpaceDimension, SplineOrder, vecPointerType >
            ::GetSample( mu, bsplineOffsetTable, weightsPtr );
  displacement += vecPointType( & point[0] );
  displacement.store( &outputPoint[0] );

  return outputPoint;
#endif // end of select between call to transformPoints or duplicate code
}

template< typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void RecursivePermutedBSplineTransform<TScalar, NDimensions, VSplineOrder>
::TransformPoints( const std::vector< InputPointType >  & pointListIn, std::vector< OutputPointType >  & pointListOut  ) const
{
 typedef BSplineKernelFunction2< VSplineOrder > BSplineKernelType;
  /** Define some constants. */
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  const unsigned int numberOfIndices = RecursiveBSplineWeightFunctionType::NumberOfIndices;
  const unsigned int maxBlockSize = 64;
  const double halfSupportSize = static_cast< double >( SplineOrder - 1 ) / 2.0 ;
  if ( pointListIn.size()  != pointListOut.size() ) {
    std::cerr <<  "ERROR in TransformPoints: Number of input points is not equal to the number of output points. " << std::endl;
    return;
  }
  if( !this->m_CoefficientImages[ 0 ] )
    {
    itkWarningMacro( << "B-spline coefficients have not been set" );
    for(unsigned int i = 0 ; i < pointListIn.size() ; ++i ) {
      pointListOut[i] = pointListIn[i]; // full copy, or would reference update be allowed?
      //anyway, we dont want to reach this point.
    }
    return;
    }
  typename RegionType::SizeType reducedGridRegionSize   = this->m_GridRegion.GetSize();
  for (unsigned int i = 0; i <  SpaceDimension; ++i ) {
    reducedGridRegionSize[i] -=  SplineOrder;

  }
  /** Allocate weights on the stack: */
  typename WeightsType::ValueType weightsArray1D[ numberOfWeights * maxBlockSize];
  //WeightsType weights1D( weightsArray1D, numberOfWeights, false );
  double indexRemainder[ maxBlockSize * SpaceDimension];
  OffsetValueType totalOffsetToSupportIndex[ maxBlockSize ];
  bool isInside[ maxBlockSize ];

  /** Initialize (helper) variables. */
  const OffsetValueType * bsplineOffsetTable = this->m_CoefficientImages[ 0 ]->GetOffsetTable();
  OffsetValueType initialOffsetToSupportIndex = 0 ;
  ScalarType * bufferPointer =  const_cast< ScalarType * >( ( this->m_InputParametersPointer->data_block() ) ); //this->m_CoefficientImages[ 0 ]->GetBufferPointer() ;

  for (unsigned int pointIdxOuter = 0 ; pointIdxOuter <  pointListIn.size() ;  pointIdxOuter += maxBlockSize )
  {

    unsigned int numInBlock =  pointListIn.size() - pointIdxOuter; // take minimum of number of points remaining and maxBlockSize.
    if ( maxBlockSize<numInBlock ) { numInBlock = maxBlockSize; } ;

    unsigned int computeWeightsIdx = 0;
    for (unsigned int blockIdx = 0 ; blockIdx < numInBlock; ++blockIdx ) {

      /** Convert to continuous index. */
      ContinuousIndexType cindex;
      this->TransformPointToContinuousGridIndex( pointListIn[ pointIdxOuter + blockIdx] , cindex );

      IndexType startIndex;
      bool insidePnt = true;
      OffsetValueType totalOffsetToSupportIndexPnt = initialOffsetToSupportIndex;
      for (unsigned int i = 0; i <  SpaceDimension; ++i ) {
       startIndex[ i ] = Math::Floor< IndexValueType >( cindex[ i ] - halfSupportSize );

         indexRemainder[ computeWeightsIdx  + i ] = cindex[ i ] - static_cast< double >( startIndex[ i ] );

        // Next line assumes  this->m_GridRegion.GetIndex()  == 0.
          // If this cannot safely be assumed, the index should be subtracted from startIndex, and totalOffsetToSupportIndex should be initialized with the index (multiplied by the bsplineoffsettable).
          // The next line Uses conversion to unsigned to test for negative values with only 1 test.
       insidePnt &= ( static_cast< unsigned int >(startIndex[ i ]) ) < reducedGridRegionSize[ i ] ;
       totalOffsetToSupportIndexPnt += startIndex[ i ] * bsplineOffsetTable[ i ];
      }
      isInside[ blockIdx ] = insidePnt;
      if (insidePnt) { computeWeightsIdx += SpaceDimension; }; // hope that conditional is optimized away.
      totalOffsetToSupportIndex[ blockIdx ] = totalOffsetToSupportIndexPnt;
    } // split loop here, to break dependency chain and also because the rest should only be evaluated if isInside.

    // Separate loop to compute the weights. TODO: compute vectorized (SSE2 or AVX vectors; would be 2, respectively 4 times faster.)
    // Note that we only compute the weights of the inside points.
    // Also note that all dimensions are treated equally (as all dimensions of all points are merged in indexRemainder)
    typename WeightsType::ValueType * weightsPtr = &weightsArray1D[0];
    for (unsigned int i =0 ; i < computeWeightsIdx; ++i) {
      this->m_Kernel->Evaluate( indexRemainder[i] , weightsPtr );
      weightsPtr += (VSplineOrder+1);
    }

    // Perform actual interpolation:
    weightsPtr = &weightsArray1D[0];
    for (unsigned int blockIdx = 0 ; blockIdx < numInBlock; ++blockIdx ) {
      if (isInside[ blockIdx ]) {
        typedef vec<ScalarType, SpaceDimension> vecPointType;
        typedef vecptr< ScalarType * , SpaceDimension> vecPointerType;
        vecPointerType mu( bufferPointer + totalOffsetToSupportIndex[ blockIdx ] * SpaceDimension );
        //vecPointerType prefetch_mu( bufferPointer + totalOffsetToSupportIndex[ blockIdx +1 ] * SpaceDimension );
        vecPointType displacement = RecursiveBSplineImplementation_GetSample< vecPointType, SpaceDimension, SplineOrder, vecPointerType >
            ::GetSample( mu, bsplineOffsetTable, weightsPtr );
        displacement += vecPointType( & pointListIn[ pointIdxOuter + blockIdx ][0] );
        displacement.store( &pointListOut[ pointIdxOuter + blockIdx ][0] );

        weightsPtr += (VSplineOrder+1) * SpaceDimension; // TODO: this is number of weights, which is an already defined compile time constant. Find that name and use it.
      } else {
        // point is not inside
        pointListOut[ pointIdxOuter + blockIdx ] = pointListIn[ pointIdxOuter + blockIdx ];
      }
    }

  }
   return;
} // end TransformPoints()



/**
 * ********************* TransformPoint ****************************
 */

template< typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursivePermutedBSplineTransform<TScalar, NDimensions, VSplineOrder>
::TransformPoint(
  const InputPointType & point,
  OutputPointType & outputPoint,
  WeightsType & weights,
  ParameterIndexArrayType & indices,
  bool & inside ) const
{
  inside = true;
  InputPointType transformedPoint = point;

  /** Check if the coefficient image has been set. */
  if( !this->m_CoefficientImages[ 0 ] )
  {
    itkWarningMacro( << "B-spline coefficients have not been set" );
    for( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      outputPoint[ j ] = transformedPoint[ j ];
    }
    return;
  }

  /***/
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( point, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  inside = this->InsideValidRegion( cindex );
  if( !inside )
  {
    outputPoint = transformedPoint;
    return;
  }

  // Compute interpolation weighs and store them in weights
  // MS: compare with AdvancedBSplineDeformableTransform
  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights, supportIndex );
  /*{typename WeightsType::ValueType * weightsPtr = & weights[0];
  for (unsigned int i =0 ; i < computeWeightsIdx; ++i) {
    this->m_Kernel->Evaluate( cindex[i] , weightsPtr );
    weightsPtr += (VSplineOrder+1);
  }}*/
  typename WeightsType::ValueType * weightsPtr = & weights[0];
 /* // Allocation of memory
  long evaluateIndexData[ ( SplineOrder + 1 ) * SpaceDimension ];
  long stepsData[ ( SplineOrder + 1 ) * SpaceDimension ];
  vnl_matrix_ref<long> evaluateIndex( SpaceDimension, SplineOrder + 1, evaluateIndexData );
  double * weightsPointer = &(weights[0]);
  long * steps = &(stepsData[0]);

  for( unsigned int ii = 0; ii < SpaceDimension; ++ii )
  {
    for( unsigned int jj = 0; jj <= SplineOrder; ++jj )
    {
      evaluateIndex[ ii ][ jj ] = supportIndex[ ii ] + jj;
    }
  }

  IndexType offsetTable;
  for( unsigned int n = 0; n < SpaceDimension; ++n )
  {
    offsetTable[ n ] = this->m_CoefficientImages[ 0 ]->GetOffsetTable()[ n ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * offsetTable[ n ] * SpaceDimension;
    }
  }*/

/** Initialize (helper) variables. */
  const OffsetValueType * bsplineOffsetTable = this->m_CoefficientImages[ 0 ]->GetOffsetTable();
  OffsetValueType totalOffsetToSupportIndex = 0;
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    totalOffsetToSupportIndex += supportIndex[ j ] * bsplineOffsetTable[ j ];
  }

  //ScalarType displacement[ SpaceDimension ];

  ScalarType* parameterPointer = const_cast< PixelType * >( ( this->m_InputParametersPointer->data_block() ) );

  typedef vec<ScalarType, SpaceDimension> vecPointType;
  typedef vecptr< ScalarType * , SpaceDimension> vecPointerType;
  vecPointerType mu( parameterPointer + totalOffsetToSupportIndex * SpaceDimension );
  // Call recursive interpolate function
  vecPointType displacement = RecursiveBSplineImplementation_GetSample< vecPointType, SpaceDimension, SplineOrder, vecPointerType >
            ::GetSample( mu, bsplineOffsetTable, weightsPtr );
  displacement += vecPointType( & point[0] );
  displacement.store( &outputPoint[0] );
  
/*  
  outputPoint.Fill( NumericTraits<TScalar>::Zero );
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    const TScalar *basePointer = const_cast< PixelType * >( ( this->m_InputParametersPointer->data_block() ) );
    unsigned int c = 0;
    outputPoint[ j ] = RecursiveBSplineTransformImplementation< SpaceDimension, SplineOrder, TScalar >
      ::TransformPoint( basePointer,
      steps,
      weightsPointer,
      basePointer,
      indices,
      c );

    // The output point is the start point + displacement.
    outputPoint[ j ] += transformedPoint[ j ];
  } // end for*/

} // end TransformPoint()


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
typename RecursivePermutedBSplineTransform< TScalarType, NDimensions, VSplineOrder >::NumberOfParametersType
RecursivePermutedBSplineTransform< TScalarType, NDimensions, VSplineOrder >
::GetNumberOfNonZeroJacobianIndices( void ) const
{
  return  RecursiveBSplineImplementation_numberOfPointsInSupportRegion< SpaceDimension,  SplineOrder>::numberOfPointsInSupportRegion * SpaceDimension;
} // end GetNumberOfNonZeroJacobianIndices()



/**
 * ********************* GetJacobian ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursivePermutedBSplineTransform< TScalar, NDimensions, VSplineOrder >
::GetJacobian( const InputPointType & ipp, JacobianType & jacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
//  itkExceptionMacro( << "GetJacobian is currently not implemented in RecursivePermutedBSplineTransform." );
  /** This implements a sparse version of the Jacobian. */

  /** Sanity check. */
  if( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  /** Initialize. */
  const NumberOfParametersType nnzji = this->GetNumberOfNonZeroJacobianIndices();
  if( ( jacobian.cols() != nnzji ) || ( jacobian.rows() != SpaceDimension ) )
  {
    jacobian.SetSize( SpaceDimension, nnzji );
    jacobian.Fill( 0.0 );
  }
  /** NOTE: if the support region does not lie totally within the grid
   * we assume zero displacement and zero Jacobian.
   */
  if( !this->InsideValidRegion( cindex ) )
  {
    nonZeroJacobianIndices.resize( this->GetNumberOfNonZeroJacobianIndices() );
    for( NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i )
    {
      nonZeroJacobianIndices[ i ] = i;
    }
    jacobian.Fill( 0.0 );
    return;
  }

  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  const unsigned int numberOfIndices = RecursiveBSplineWeightFunctionType::NumberOfIndices;
  typename WeightsType::ValueType weightsArray1D[ numberOfWeights ];
  WeightsType weights1D( weightsArray1D, numberOfWeights, false );
  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights1D, supportIndex );

  //std::cerr << "GetJacobian in " << ipp << " cindex: " << cindex << " supportIndex " << supportIndex << " Nr of weights " << numberOfWeights << std::endl; // debug output.

  /** Recursively compute the full weights array .
   * The pointer has changed after this function call.
   */
  const unsigned int numberOfWeightsPerDimension = RecursiveBSplineImplementation_numberOfPointsInSupportRegion< SpaceDimension,  SplineOrder>::numberOfPointsInSupportRegion ;

  typename WeightsType::ValueType fullWeightsArray[ numberOfWeightsPerDimension ];
  typename WeightsType::ValueType  * fullWeightsArrayPtr = &fullWeightsArray[0];
  RecursiveBSplineImplementation_GetJacobian< typename WeightsType::ValueType * , SpaceDimension,  SplineOrder, typename WeightsType::ValueType>
    ::GetJacobian( fullWeightsArrayPtr, weightsArray1D, 1.0 );

  /** Put at the right positions in jacobian. */
  ParametersValueType * jacobianPointer = jacobian.data_block();
  for( unsigned int i = 0; i < numberOfWeightsPerDimension; ++i )
  {
    // create scaled identity matrix for each i. 
    // Note that we assume that only this function writes to jacobian. 
    // Hence any elements that we do not write to stay at the value that they are initialized with (= 0.0).
    for( unsigned int d = 0 ; d < SpaceDimension ; ++d ) {
      unsigned int index = (i * SpaceDimension + d)  +  (d * nnzji);  //(column) + (row)
      jacobianPointer[index] = fullWeightsArray[i];
      //jacobianPointer += SpaceDimension;
    }
  }
  //std::cerr << "Jacobian ( " << jacobian.rows()  << ", " << jacobian.cols() <<" ) " << jacobian << std::endl; // debug output.
  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  for (int i = 0 ; i < nonZeroJacobianIndices.size() ; ++i ) {
    nonZeroJacobianIndices[i]  = i;
  }
  this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportIndex );
  /*std::cerr << "Jacobian ( " << jacobian.rows()  << ", " << jacobian.cols() <<" ) " << jacobian << std::endl; // debug output.
std::cerr << "display " << nonZeroJacobianIndices.size() << " nonzero indices [";
 for (int i = 0 ; i < nonZeroJacobianIndices.size() ; ++i ) {
   std::cerr << nonZeroJacobianIndices[i] << " " ;
 }
 std::cerr << "]" << std::endl; // debug output.*/


} // end GetJacobian()


/**
 * ********************* EvaluateJacobianAndImageGradientProduct ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
  RecursivePermutedBSplineTransform< TScalar, NDimensions, VSplineOrder >
::EvaluateJacobianWithImageGradientProduct(
  const InputPointType & ipp,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType & imageJacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
 // itkExceptionMacro( << "EvaluateJacobianWithImageGradientProduct is currently not implemented in RecursivePermutedBSplineTransform." );
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  /** Initialize. */
  const NumberOfParametersType nnzji = this->GetNumberOfNonZeroJacobianIndices();

  /** NOTE: if the support region does not lie totally within the grid
   * we assume zero displacement and zero Jacobian.
   */
  if( !this->InsideValidRegion( cindex ) )
  {
    nonZeroJacobianIndices.resize( this->GetNumberOfNonZeroJacobianIndices() );
    for( NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i )
    {
      nonZeroJacobianIndices[ i ] = i;
    }
    return;
  }

  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  const unsigned int numberOfIndices = RecursiveBSplineWeightFunctionType::NumberOfIndices;
  typename WeightsType::ValueType weightsArray1D[ numberOfWeights ];
  WeightsType weights1D( weightsArray1D, numberOfWeights, false );
  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights1D, supportIndex );

  //std::cerr << "EvaluateJacobianWithImageGradientProduct in " << ipp << " cindex: " << cindex << " supportIndex" << supportIndex << " Nr of weights " << numberOfWeights << std::endl; // debug output.
 /** Recursively compute the inner product of the Jacobian and the moving image gradient.
   * The pointer has changed after this function call.
   */
  //ParametersValueType migArray[ SpaceDimension ];
  double migArray[ SpaceDimension ];//InternalFloatType
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    migArray[ j ] = movingImageGradient[ j ];
  }
  typedef vec< double, SpaceDimension > imgGradType;
  imgGradType mig( &migArray[0] );

  //const ParametersValueType * movingImageGradientPointer = movingImageGradient.GetDataPointer();
  ParametersValueType * imageJacobianPointer = imageJacobian.data_block();
  typedef vecptr< ParametersValueType *, SpaceDimension> imageJacobianVecPointerType;
  imageJacobianVecPointerType  imageJacobianVecPointer( imageJacobianPointer );
  RecursiveBSplineImplementation_GetJacobian< imageJacobianVecPointerType, SpaceDimension, SplineOrder, imgGradType >
    ::GetJacobian( imageJacobianVecPointer, weightsArray1D, mig );
  //std::cerr << "Jacobian mul " << imageJacobian << std::endl; // debug output.

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportIndex );
 /*std::cerr << "display " << nonZeroJacobianIndices.size() << " nonzero indices [";
 for (int i = 0 ; i < nonZeroJacobianIndices.size() ; ++i ) {
   std::cerr << nonZeroJacobianIndices[i] << " " ;
 }
 std::cerr << "]" << std::endl; // debug output.*/
} // end EvaluateJacobianWithImageGradientProduct()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursivePermutedBSplineTransform< TScalar, NDimensions, VSplineOrder >
::GetSpatialJacobian(
  const InputPointType & ipp,
  SpatialJacobianType & sj ) const
{
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and identity spatial Jacobian
  if( !this->InsideValidRegion( cindex ) )
  {
    sj.SetIdentity();
    return;
  }

  /** Create storage for the B-spline interpolation weights. */
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  const unsigned int numberOfIndices = RecursiveBSplineWeightFunctionType::NumberOfIndices;
  typename WeightsType::ValueType weightsArray1D[ numberOfWeights ];
  WeightsType weights1D( weightsArray1D, numberOfWeights, false );
  typename WeightsType::ValueType derivativeWeightsArray1D[ numberOfWeights ];
  WeightsType derivativeWeights1D( derivativeWeightsArray1D, numberOfWeights, false );

  double * weightsPointer = &(weights1D[0]);
  double * derivativeWeightsPointer = &(derivativeWeights1D[0]);

  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights1D, supportIndex );
  this->m_RecursiveBSplineWeightFunction->EvaluateDerivative( cindex, derivativeWeights1D, supportIndex );

  /** Compute the offset to the start index. */
  const OffsetValueType * bsplineOffsetTable = this->m_CoefficientImages[ 0 ]->GetOffsetTable();
  OffsetValueType totalOffsetToSupportIndex = 0;
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    totalOffsetToSupportIndex += supportIndex[ j ] * bsplineOffsetTable[ j ];
  }

  typedef vec<ScalarType, SpaceDimension> vecPointType;
  typedef vecptr< ScalarType * , SpaceDimension> vecPointerType;
  ScalarType* parameterPointer = const_cast< PixelType * >( ( this->m_InputParametersPointer->data_block() ) );
  vecPointerType mu( parameterPointer + totalOffsetToSupportIndex * SpaceDimension );
  double spatialJacobian[ SpaceDimension * ( SpaceDimension + 1 ) ];
  vecPointerType spatialJacobianV( &spatialJacobian[0] );
  /** Recursively compute the spatial Jacobian. */
  RecursiveBSplineImplementation_GetSpatialJacobian< vecPointerType, SpaceDimension, SplineOrder, vecPointerType >
            ::GetSpatialJacobian( spatialJacobianV, mu, bsplineOffsetTable, weightsPointer, derivativeWeightsPointer  );

  /** Copy the correct elements to the spatial Jacobian.
   * The first SpaceDimension elements are actually the displacement, i.e. the recursive
   * function GetSpatialJacobian() has the TransformPoint as a free by-product.
   */
  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    for( unsigned int j = 0; j < SpaceDimension; ++j )
    {
      sj( i, j ) = spatialJacobian[ i + ( j + 1 ) * SpaceDimension ];
    }
  }

  /** Take into account grid spacing and direction cosines. */
  sj = sj * this->m_PointToIndexMatrix2;

  /** Add the identity matrix, as this is a transformation, not displacement. */
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    sj( j, j ) += 1.0;
  }

  // \todo check if we first need to do the matrix multiplication and then
  // add the identity matrix, or vice versa.
  // ANSWER: no, as in TransformPoint the untransformed point is added, while the displacement (and hence the jacobian of that) is evaluated on the cindex.
} // end GetSpatialJacobian()



/**
 * ********************* GetSpatialHessian ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursivePermutedBSplineTransform< TScalar, NDimensions, VSplineOrder >
::GetSpatialHessian(
  const InputPointType & ipp,
  SpatialHessianType & sh ) const
{
  /** Convert the physical point to a continuous index, which
   * is needed for the evaluate functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and zero spatial Hessian
  if( !this->InsideValidRegion( cindex ) )
  {
    for( unsigned int i = 0; i < sh.Size(); ++i )
    {
      sh[ i ].Fill( 0.0 );
    }
    return;
  }

  /** Create storage for the B-spline interpolation weights. */
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  const unsigned int numberOfIndices = RecursiveBSplineWeightFunctionType::NumberOfIndices;
  typename WeightsType::ValueType weightsArray1D[ numberOfWeights ];
  WeightsType weights1D( weightsArray1D, numberOfWeights, false );
  typename WeightsType::ValueType derivativeWeightsArray1D[ numberOfWeights ];
  WeightsType derivativeWeights1D( derivativeWeightsArray1D, numberOfWeights, false );
  typename WeightsType::ValueType hessianWeightsArray1D[ numberOfWeights ];
  WeightsType hessianWeights1D( hessianWeightsArray1D, numberOfWeights, false );

  double * weightsPointer = &(weights1D[0]);
  double * derivativeWeightsPointer = &(derivativeWeights1D[0]);
  double * hessianWeightsPointer = &(hessianWeights1D[0]);

  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights1D, supportIndex );
  this->m_RecursiveBSplineWeightFunction->EvaluateDerivative( cindex, derivativeWeights1D, supportIndex );
  this->m_RecursiveBSplineWeightFunction->EvaluateSecondOrderDerivative( cindex, hessianWeights1D, supportIndex );

  /** Compute the offset to the start index. */
  const OffsetValueType * bsplineOffsetTable = this->m_CoefficientImages[ 0 ]->GetOffsetTable();
  OffsetValueType totalOffsetToSupportIndex = 0;
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    totalOffsetToSupportIndex += supportIndex[ j ] * bsplineOffsetTable[ j ];
  }

  typedef vec<ScalarType, SpaceDimension> vecPointType;
  typedef vecptr< ScalarType * , SpaceDimension> vecPointerType;
  ScalarType* parameterPointer = const_cast< PixelType * >( ( this->m_InputParametersPointer->data_block() ) );
  vecPointerType mu( parameterPointer + totalOffsetToSupportIndex * SpaceDimension );
  double spatialHessian[ SpaceDimension * ( SpaceDimension + 1 ) * ( SpaceDimension + 2 ) / 2 ];
  vecPointerType spatialHessianV( &spatialHessian[0] );
  /** Recursively compute the spatial Jacobian. */
  RecursiveBSplineImplementation_GetSpatialHessian< vecPointerType, SpaceDimension, SplineOrder, vecPointerType >
            ::GetSpatialHessian( spatialHessianV, mu, bsplineOffsetTable, weightsPointer, derivativeWeightsPointer, hessianWeightsPointer );

  /** Copy the correct elements to the spatial Hessian.
   * The first SpaceDimension elements are actually the displacement, i.e. the recursive
   * function GetSpatialHessian() has the TransformPoint as a free by-product.
   * In addition, the spatial Jacobian is a by-product.
   */
  unsigned int k = 2 * SpaceDimension;
  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    for( unsigned int j = 0; j < ( i + 1 ) * SpaceDimension; ++j )
    {
      sh[ j % SpaceDimension ]( i, j / SpaceDimension ) = spatialHessian[ k + j ];
    }
    k += ( i + 2 ) * SpaceDimension;
  }

  /** Mirror, as only the lower triangle is now filled. */
  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    for( unsigned int j = 0; j < SpaceDimension - 1; ++j )
    {
      for( unsigned int k = 1; k < SpaceDimension; ++k )
      {
        sh[ i ]( j, k ) = sh[ i ]( k, j );
      }
    }
  }

  /** Take into account grid spacing and direction matrix */
  for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
  {
    sh[ dim ] = this->m_PointToIndexMatrixTransposed2
      * ( sh[ dim ] * this->m_PointToIndexMatrix2 );
  }

} // end GetSpatialHessian()


/**
 * ********************* ComputeNonZeroJacobianIndices ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursivePermutedBSplineTransform< TScalar, NDimensions, VSplineOrder >
::ComputeNonZeroJacobianIndices(
  NonZeroJacobianIndicesType & nonZeroJacobianIndices,
  const RegionType & supportRegion  ) const
{
  std::cerr <<  "ERROR ComputeNonZeroJacobianIndices: This version should not be called for RecursivePermutedBSplineTransform . " << std::endl;
}

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursivePermutedBSplineTransform< TScalar, NDimensions, VSplineOrder >
::ComputeNonZeroJacobianIndices(
  NonZeroJacobianIndicesType & nonZeroJacobianIndices,
  const IndexType & supportIndex  ) const
{
  // Make sure the size is correct: 
  nonZeroJacobianIndices.resize( this->GetNumberOfNonZeroJacobianIndices() );

  // Compute the steps in index that are taken:
  const OffsetValueType * bsplineOffsetTable = this->m_CoefficientImages[ 0 ]->GetOffsetTable();

  OffsetValueType totalOffsetToSupportIndex = 0;
  OffsetValueType scaledGridOffsetTable[ SpaceDimension ];
  for (int i = 0; i < SpaceDimension ; ++i ) {
    scaledGridOffsetTable[i] = bsplineOffsetTable[i] * SpaceDimension;
    totalOffsetToSupportIndex += scaledGridOffsetTable[i] * supportIndex[i];
  }

  // declare and set the vector types that the recursion uses
  typedef unsigned long * nzjiPointerType; 
  typedef vecptr< nzjiPointerType, SpaceDimension> nzjiVecPointerType;
  //nonZeroJacobianIndices[0] = 123456;
  nzjiVecPointerType temp_nzji( & (nonZeroJacobianIndices[0]) );
  typedef vec< OffsetValueType, SpaceDimension> vecOffsetValueType; 
  OffsetValueType totalOffsetToSupportIndexArray[SpaceDimension]; 
  for (int i = 0; i < SpaceDimension+1; ++i ) {
    totalOffsetToSupportIndexArray[i] = totalOffsetToSupportIndex + i;
  }
  vecOffsetValueType vecTotalOffsetToSupportIndex( &totalOffsetToSupportIndexArray[0] );

  // recursive computeNonZeroJacobianIndices
  RecursiveBSplineImplementation_ComputeNonZeroJacobianIndices< nzjiVecPointerType , SpaceDimension, SplineOrder, vecOffsetValueType, 0>
    ::ComputeNonZeroJacobianIndices(temp_nzji, vecTotalOffsetToSupportIndex, &scaledGridOffsetTable[0] );

} // end ComputeNonZeroJacobianIndices()


} // end namespace itk


#endif
