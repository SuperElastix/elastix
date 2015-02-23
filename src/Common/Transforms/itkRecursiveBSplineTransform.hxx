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
#ifndef __itkRecursiveBSplineTransform_hxx
#define __itkRecursiveBSplineTransform_hxx

#include "itkRecursiveBSplineTransform.h"

//#define RECURSIVEVERSION3                    // use recusivebspline version 3. This uses an permuted parameter grid. 
											   // Elastix standard = [spatial_dimensions   vector_dimension], where vector_dimension iterates over [x,y,z]
											   // RecursiveVersion3 =[ vector dimension   spatial dimensions].
											   // The advantage of this is much better memory locality as well as much more convenient optimization:
//#define RECURSIVEVERSION3_OPTIMIZED_SSE2     // Enable a manually optimized SSE2 implementation of (currently 1) end case.
											   // Can only be used if also RECURSIVEVERSION3 is defined.
//#define RECURSIVEVERSION4                    // Test version for TransformPoints. Include SSE2 optimized small vector representations and return by value.
											   // Requires RECURSIVEVERSION3.
											   // Requires emm_vec from DPoot.
								               // Can be combined with RECURSIVEVERSION3_OPTIMIZED_SSE2 for a specialized faster end case version.
											   // NOTE: strangely enough, enabling 'whole program optimization' (configuration properties, c++, optimization) makes this version substantially (40%) slower. 
//#define RECUSIVEBSPLINE_FORWARDTRANSFORMPOINT // if defined forwards the transformpoint to transformpoints to minimize code duplication. 
												// (Currently horribly much slower; did not yet investigate why)

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template <typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
RecursiveBSplineTransform< TScalar, NDimensions, VSplineOrder >
::RecursiveBSplineTransform() : Superclass()
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
typename RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::OutputPointType
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::TransformPoint( const InputPointType & point ) const
{
#ifdef RECUSIVEBSPLINE_FORWARDTRANSFORMPOINT
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
  ScalarType displacement[ SpaceDimension ];

#ifdef RECURSIVEVERSION3
  ScalarType * mu;
    //displacement[ j ] = 0.0;
    mu = this->m_CoefficientImages[ 0 ]->GetBufferPointer() + totalOffsetToSupportIndex * SpaceDimension;

  /** Call recursive interpolate function, vector version. */
  RecursiveBSplineTransformImplementation3< SpaceDimension, SpaceDimension, SplineOrder, TScalar , true>
    ::TransformPoint2( displacement, mu, bsplineOffsetTable, weightsArray1D );
#else
  ScalarType * mu[ SpaceDimension ];
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    //displacement[ j ] = 0.0;
    mu[ j ] = this->m_CoefficientImages[ j ]->GetBufferPointer() + totalOffsetToSupportIndex;
  }

  /** Call recursive interpolate function, vector version. */
  RecursiveBSplineTransformImplementation2< SpaceDimension, SpaceDimension, SplineOrder, TScalar >
    ::TransformPoint2( displacement, mu, bsplineOffsetTable, weightsArray1D );
#endif
  // The output point is the start point + displacement.
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    outputPoint[ j ] = displacement[ j ] + point[ j ];
  }

  return outputPoint;
#endif // end of select between call to transformPoints or duplicate code
}

template< typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
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
#ifdef RECURSIVEVERSION3
  ScalarType * bufferPointer =  this->m_CoefficientImages[ 0 ]->GetBufferPointer() ;
#endif

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

	  
#ifdef RECURSIVEVERSION4
			  typedef vec<ScalarType, SpaceDimension> vecPointType;
			  typedef vecptr< ScalarType * , SpaceDimension> vecPointerType;
			  vecPointerType mu( bufferPointer + totalOffsetToSupportIndex[ blockIdx ] * SpaceDimension );
			  //vecPointerType prefetch_mu( bufferPointer + totalOffsetToSupportIndex[ blockIdx +1 ] * SpaceDimension );
			  vecPointType displacement = RecursiveBSplineTransformImplementation4< vecPointType, SpaceDimension, SplineOrder, vecPointerType , false >
						::TransformPoint2( mu, bsplineOffsetTable, weightsPtr );//, prefetch_mu);
			  displacement += vecPointType( & pointListIn[ pointIdxOuter + blockIdx ][0] );
			  displacement.store( &pointListOut[ pointIdxOuter + blockIdx ][0] );

#elif defined RECURSIVEVERSION3
			  ScalarType displacement[ SpaceDimension ];
			  ScalarType * mu = bufferPointer + totalOffsetToSupportIndex[ blockIdx ] * SpaceDimension;

			  /** Call recursive interpolate function, vector version. */
			  if (bsplineOffsetTable[SpaceDimension] > 100000) { // only if bspline coefficients are expected to get out of the cache do prefetching. 
			    RecursiveBSplineTransformImplementation3< SpaceDimension, SpaceDimension, SplineOrder, TScalar , true >
						::TransformPoint2( displacement, mu, bsplineOffsetTable, weightsPtr );
			  } else {
			    RecursiveBSplineTransformImplementation3< SpaceDimension, SpaceDimension, SplineOrder, TScalar , false >
						::TransformPoint2( displacement, mu, bsplineOffsetTable, weightsPtr );
			  }
#else
			  ScalarType displacement[ SpaceDimension ];
			  ScalarType * mu[ SpaceDimension ];
			  for( unsigned int j = 0; j < SpaceDimension; ++j )
			  {
				//displacement[ j ] = 0.0;
				mu[ j ] = this->m_CoefficientImages[ j ]->GetBufferPointer() + totalOffsetToSupportIndex[ blockIdx ];
			  }

			  /** Call recursive interpolate function, vector version. */
			  RecursiveBSplineTransformImplementation2< SpaceDimension, SpaceDimension, SplineOrder, TScalar >
				::TransformPoint2( displacement, mu, bsplineOffsetTable, weightsPtr );
#endif
#ifndef RECURSIVEVERSION4
			  // The output point is the start point + displacement.
			  for( unsigned int j = 0; j < SpaceDimension; ++j )
			  {
				pointListOut[ pointIdxOuter + blockIdx ][j] = displacement[j] + pointListIn[ pointIdxOuter + blockIdx ][j];
			  }
#endif
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
 * ********************* TransformPointOld ****************************
 */

template< typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
typename RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::OutputPointType
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::TransformPointOld( const InputPointType & point ) const
{
  /** Allocate memory on the stack: */
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  const unsigned int numberOfIndices = RecursiveBSplineWeightFunctionType::NumberOfIndices;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  typename ParameterIndexArrayType::ValueType indicesArray[ numberOfIndices ];
  WeightsType             weights( weightsArray, numberOfWeights, false );
  ParameterIndexArrayType indices( indicesArray, numberOfIndices, false );

  OutputPointType         outputPoint;
  bool                    inside;

  this->TransformPoint( point, outputPoint, weights, indices, inside );
  return outputPoint;

} // end TransformPointOld()


/**
 * ********************* TransformPoint ****************************
 */

template< typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
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

  // Allocation of memory
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
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * offsetTable[ n ]
#ifdef RECURSIVEVERSION3  // when compiled with recursiveversion3, the parameter order is permuted; requiring multiplication of steps with SpaceDimension
																						* SpaceDimension
#endif
																									      ;
    }
  }

  // Call recursive interpolate function
  outputPoint.Fill( NumericTraits<TScalar>::Zero );
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
#ifdef RECURSIVEVERSION3 // when compiled with recursiveversion3, the parameter order is permuted; starting xyz all at CoefficientImage[0], but with offset j.
	  const TScalar *basePointer = this->m_CoefficientImages[ 0 ]->GetBufferPointer() + j;
#else
    const TScalar *basePointer = this->m_CoefficientImages[ j ]->GetBufferPointer();
#endif
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
  } // end for

} // end TransformPoint()


/**
 * ********************* GetJacobian ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursiveBSplineTransform< TScalar, NDimensions, VSplineOrder >
::GetJacobian( const InputPointType & ipp, JacobianType & jacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
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

#if 1
  /** Recursively compute the first numberOfIndices entries of the Jacobian.
   * They are directly written in the Jacobian matrix memory block.
   * The pointer has changed after this function call.
   */
  ParametersValueType * jacobianPointer = jacobian.data_block();
  RecursiveBSplineTransformImplementation< SpaceDimension, SplineOrder, TScalar >
    ::GetJacobian( jacobianPointer, weightsArray1D, 1.0 );

  /** Copy the Jacobian values to the other dimensions. */
  jacobianPointer = jacobian.data_block();
  for( unsigned int d = 1; d < SpaceDimension; ++d )
  {
    unsigned long offset = d * SpaceDimension * numberOfIndices + d * numberOfIndices;
    std::copy( jacobianPointer, jacobianPointer + numberOfIndices, jacobianPointer + offset );
  }
#else
  /** Recursively compute the first numberOfIndices entries of the Jacobian.
   * They are directly written in the Jacobian matrix memory block.
   * The pointer has changed after this function call.
   *
   * This version is actually a tiny bit slower than the above which uses
   * a very efficient memcopy.
   */
  ParametersValueType * jacobianPointer = jacobian.data_block();
  RecursiveBSplineTransformImplementation2< SpaceDimension, SpaceDimension, SplineOrder, TScalar >
    ::GetJacobian( jacobianPointer, weightsArray1D, 1.0 );
#endif

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );
  this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportRegion );

} // end GetJacobian()


/**
 * ********************* EvaluateJacobianAndImageGradientProduct ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
  RecursiveBSplineTransform< TScalar, NDimensions, VSplineOrder >
::EvaluateJacobianWithImageGradientProduct(
  const InputPointType & ipp,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType & imageJacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
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

  /** Recursively compute the inner product of the Jacobian and the moving image gradient.
   * The pointer has changed after this function call.
   */
  //ParametersValueType migArray[ SpaceDimension ];
  double migArray[ SpaceDimension ];//InternalFloatType
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    migArray[ j ] = movingImageGradient[ j ];
  }
  //const ParametersValueType * movingImageGradientPointer = movingImageGradient.GetDataPointer();
  ParametersValueType * imageJacobianPointer = imageJacobian.data_block();
  RecursiveBSplineTransformImplementation2< SpaceDimension, SpaceDimension, SplineOrder, TScalar >
    //::EvaluateJacobianWithImageGradientProduct( imageJacobianPointer, movingImageGradientPointer, weightsArray1D, 1.0 );
    ::EvaluateJacobianWithImageGradientProduct( imageJacobianPointer, migArray, weightsArray1D, 1.0 );

  /** Setup support region needed for the nonZeroJacobianIndices. */
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportRegion );

} // end EvaluateJacobianWithImageGradientProduct()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursiveBSplineTransform< TScalar, NDimensions, VSplineOrder >
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

#ifdef RECURSIVEVERSION3
  /** Get handles to the mu's. */
  ScalarType * mu;
  mu = this->m_CoefficientImages[ 0 ]->GetBufferPointer() + totalOffsetToSupportIndex * SpaceDimension;

  /** Recursively compute the spatial Jacobian. */
  double spatialJacobian[ SpaceDimension * ( SpaceDimension + 1 ) ];//double
  RecursiveBSplineTransformImplementation3< SpaceDimension, SpaceDimension, SplineOrder, TScalar , true>
    ::GetSpatialJacobian( spatialJacobian, mu, bsplineOffsetTable, weightsPointer, derivativeWeightsPointer );
#else
  /** Get handles to the mu's. */
  ScalarType * mu[ SpaceDimension ];
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    mu[ j ] = this->m_CoefficientImages[ j ]->GetBufferPointer() + totalOffsetToSupportIndex;
  }

  /** Recursively compute the spatial Jacobian. */
  double spatialJacobian[ SpaceDimension * ( SpaceDimension + 1 ) ];//double
  RecursiveBSplineTransformImplementation2< SpaceDimension, SpaceDimension, SplineOrder, TScalar >
    ::GetSpatialJacobian( spatialJacobian, mu, bsplineOffsetTable, weightsPointer, derivativeWeightsPointer );
#endif
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
} // end GetSpatialJacobian()



/**
 * ********************* GetSpatialHessian ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursiveBSplineTransform< TScalar, NDimensions, VSplineOrder >
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
#ifdef RECURSIVEVERSION3
  /** Get handles to the mu's. */
  ScalarType * mu;
  mu = this->m_CoefficientImages[ 0 ]->GetBufferPointer() + totalOffsetToSupportIndex * SpaceDimension;

  /** Recursively compute the spatial Jacobian. */
  double spatialHessian[ SpaceDimension * ( SpaceDimension + 1 ) * ( SpaceDimension + 2 ) / 2 ];
  RecursiveBSplineTransformImplementation3< SpaceDimension, SpaceDimension, SplineOrder, TScalar , true>
    ::GetSpatialHessian( spatialHessian, mu, bsplineOffsetTable,
    weightsPointer, derivativeWeightsPointer, hessianWeightsPointer );
#else
  /** Get handles to the mu's. */
  ScalarType * mu[ SpaceDimension ];
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    mu[ j ] = this->m_CoefficientImages[ j ]->GetBufferPointer() + totalOffsetToSupportIndex;
  }

  /** Recursively compute the spatial Jacobian. */
  double spatialHessian[ SpaceDimension * ( SpaceDimension + 1 ) * ( SpaceDimension + 2 ) / 2 ];
  RecursiveBSplineTransformImplementation2< SpaceDimension, SpaceDimension, SplineOrder, TScalar >
    ::GetSpatialHessian( spatialHessian, mu, bsplineOffsetTable,
    weightsPointer, derivativeWeightsPointer, hessianWeightsPointer );
#endif
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
RecursiveBSplineTransform< TScalar, NDimensions, VSplineOrder >
::ComputeNonZeroJacobianIndices(
  NonZeroJacobianIndicesType & nonZeroJacobianIndices,
  const RegionType & supportRegion ) const
{
  /** Initialize some helper variables. */
  const unsigned long parametersPerDim = this->GetNumberOfParametersPerDimension();
  nonZeroJacobianIndices.resize( this->GetNumberOfNonZeroJacobianIndices() );

  /** Compute total offset at start index. */
  IndexType startIndex = supportRegion.GetIndex();
  const OffsetValueType * gridOffsetTable = this->m_CoefficientImages[ 0 ]->GetOffsetTable();
  OffsetValueType totalOffsetToSupportIndex = 0;
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    totalOffsetToSupportIndex += startIndex[ j ] * gridOffsetTable[ j ];
  }

  /** Call the recursive implementation. */
  unsigned int c = 0;
  unsigned long currentIndex = totalOffsetToSupportIndex;
  RecursiveBSplineTransformImplementation2< SpaceDimension, SpaceDimension, SplineOrder, TScalar >
    ::ComputeNonZeroJacobianIndices( &nonZeroJacobianIndices[0],
    parametersPerDim, currentIndex, gridOffsetTable, c );

} // end ComputeNonZeroJacobianIndices()


} // end namespace itk


#endif
