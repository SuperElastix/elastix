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
#ifndef __itkRecursiveBSplineTransformImplementation_h
#define __itkRecursiveBSplineTransformImplementation_h

#include "itkRecursiveBSplineInterpolationWeightFunction.h"

#ifdef RECURSIVEVERSION4
#include "c:\Users\Dirk\EclipseSVNWorkspace\MATLAB\Tools\supportingfiles\emm_vec.hxx"
#endif

namespace itk
{
/** \class RecursiveBSplineTransformImplementation
 *
 * \brief This helper class contains the actual implementation of the
 * recursive B-spline transform
 *
 * These classes contain static inline functions for performance.
 *
 * \ingroup ITKTransform
 */

template< unsigned int SpaceDimension, unsigned int SplineOrder, class TScalar >
class RecursiveBSplineTransformImplementation
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  typedef TScalar ScalarType;
  typedef double InternalFloatType;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** TransformPoint recursive implementation. */
  static inline InternalFloatType TransformPoint(
    const ScalarType * mu, const long * steps, const double * weights1D,
    const ScalarType * coefBasePointer, Array<unsigned long> & indices, unsigned int & c )
  {
    ScalarType coord = 0.0;
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      const ScalarType * tmp_mu = mu + steps[ k + HelperConstVariable ];
      coord += RecursiveBSplineTransformImplementation< SpaceDimension - 1, SplineOrder, TScalar >
        ::TransformPoint( tmp_mu, steps, weights1D, coefBasePointer, indices, c ) * weights1D[ k + HelperConstVariable ];
    }
    return coord;
  } // end InterpolateTransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void GetJacobian(
    ScalarType * & jacobians, const double * weights1D , double value )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation< SpaceDimension - 1, SplineOrder, TScalar >
        ::GetJacobian( jacobians, weights1D, value * weights1D[ k + HelperConstVariable ] );
    }
  } // end GetJacobian()


}; // end class

/** \class RecursiveBSplineTransformImplementation
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< unsigned int SplineOrder, class TScalar >
class RecursiveBSplineTransformImplementation< 0, SplineOrder, TScalar >
{
public:

  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  typedef TScalar ScalarType;
  typedef double InternalFloatType;

  /** TransformPoint recursive implementation. */
  static inline InternalFloatType TransformPoint(
    const ScalarType * mu,
    const long * steps,
    const double * weights1D,
    const TScalar *coefBasePointer,
    Array<unsigned long> & indices,
    unsigned int & c )
  {
    indices[ c ] = mu - coefBasePointer;
    ++c;
    return *mu;
  } // end TransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void GetJacobian(
    ScalarType * & jacobians, const double * weights1D, double value )
  {
    *jacobians = value;// potential conversion
    ++jacobians;
  } // end GetJacobian()


}; // end class



/** \class RecursiveBSplineTransformImplementation2
 *
 * \brief This helper class contains the actual implementation of the
 * recursive B-spline transform
 *
 * Compared to the RecursiveBSplineTransformImplementation class, this
 * class works as a vector operator, and is therefore also templated
 * over the OutputDimension.
 *
 * \ingroup ITKTransform
 */

template< unsigned int OutputDimension, unsigned int SpaceDimension, unsigned int SplineOrder, class TScalar >
class RecursiveBSplineTransformImplementation2
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  typedef TScalar ScalarType;
  typedef double InternalFloatType;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** Typedef to know the number of indices at compile time. */
  typedef itk::RecursiveBSplineInterpolationWeightFunction<
    ScalarType, OutputDimension, SplineOrder > RecursiveBSplineWeightFunctionType;
  itkStaticConstMacro( BSplineNumberOfIndices, unsigned int,
    RecursiveBSplineWeightFunctionType::NumberOfIndices );

  typedef ScalarType *  OutputPointType;
  typedef ScalarType ** CoefficientPointerVectorType;

  /** TransformPoint recursive implementation. */
  static inline void TransformPoint(
    OutputPointType opp,
    const CoefficientPointerVectorType mu, const OffsetValueType * steps, const double * weights1D )
  {
    ScalarType * tmp_mu[ OutputDimension ];
    ScalarType tmp_opp[ OutputDimension ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        tmp_opp[ j ] = 0.0;
        tmp_mu[ j ] = mu[ j ] + steps[ k + HelperConstVariable ];
      }

      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::TransformPoint( tmp_opp, tmp_mu, steps, weights1D );
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        opp[ j ] += tmp_opp[ j ] * weights1D[ k + HelperConstVariable ];
      }
    }
  } // end TransformPoint()


  /** TransformPoint recursive implementation. */
  static inline void TransformPoint2(
    OutputPointType opp, const CoefficientPointerVectorType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
    ScalarType * tmp_mu[ OutputDimension ];
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      tmp_mu[ j ] = mu[ j ];
    }

    /** Create a temporary sj and initialize the original. */
    ScalarType tmp_opp[ OutputDimension ];
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      opp[ j ] = 0.0;
    }

    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::TransformPoint2( tmp_opp, tmp_mu, gridOffsetTable, weights1D );

      // Multiply by the weights
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        opp[ j ] += tmp_opp[ j ] * weights1D[ k + HelperConstVariable ];

        // move to the next mu
        tmp_mu[ j ] += bot;
      }
    }
  } // end TransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void GetJacobian(
    ScalarType * & jacobians, const double * weights1D, double value )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetJacobian( jacobians, weights1D, value * weights1D[ k + HelperConstVariable ] );
    }
  } // end GetJacobian()


  /** EvaluateJacobianWithImageGradientProduct recursive implementation. */
  static inline void EvaluateJacobianWithImageGradientProduct(
    ScalarType * & imageJacobian, const InternalFloatType * movingImageGradient,
    const double * weights1D, double value )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::EvaluateJacobianWithImageGradientProduct( imageJacobian, movingImageGradient, weights1D,
          value * weights1D[ k + HelperConstVariable ] );
    }
  } // end EvaluateJacobianWithImageGradientProduct()


  /** ComputeNonZeroJacobianIndices recursive implementation. */
  static inline void ComputeNonZeroJacobianIndices(
    unsigned long * nzji,
    unsigned long parametersPerDim,
    unsigned long currentIndex,
    const OffsetValueType * gridOffsetTable,
    unsigned int & c )
  {
    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::ComputeNonZeroJacobianIndices( nzji, parametersPerDim, currentIndex, gridOffsetTable, c );
      currentIndex += bot;
    }
  } // end ComputeNonZeroJacobianIndices()


  /** GetSpatialJacobian recursive implementation.
   * As an (almost) free by-product this function delivers the displacement,
   * i.e. the TransformPoint() function.
   */
  static inline void GetSpatialJacobian(
    InternalFloatType * sj,
    const CoefficientPointerVectorType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
    ScalarType * tmp_mu[ OutputDimension ];
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      tmp_mu[ j ] = mu[ j ];
    }

    /** Create a temporary sj and initialize the original. */
    InternalFloatType tmp_sj[ OutputDimension * SpaceDimension ];
    for( unsigned int n = 0; n < OutputDimension * ( SpaceDimension + 1 ); ++n )
     {
       sj[ n ] = 0.0;
     }

    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetSpatialJacobian( tmp_sj, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D );

      // Multiply by the weights
      for( unsigned int n = 0; n < OutputDimension * SpaceDimension; ++n )
      {
        sj[ n ] += tmp_sj[ n ] * weights1D[ k + HelperConstVariable ];
      }

      // Multiply by the derivative weights
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        sj[ OutputDimension * SpaceDimension + j ]
          += tmp_sj[ j ] * derivativeWeights1D[ k + HelperConstVariable ];

        // move to the next mu
        tmp_mu[ j ] += bot;
      }
    }
  } // end GetSpatialJacobian()


  /** GetSpatialHessian recursive implementation.
   * As an (almost) free by-product this function delivers the displacement,
   * i.e. the TransformPoint() function, as well as the SpatialJacobian.
   */
  static inline void GetSpatialHessian(
    InternalFloatType * sh,
    const CoefficientPointerVectorType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D )   // 2nd derivative of B-spline
  {
    const unsigned int helperDim1 = OutputDimension * SpaceDimension * ( SpaceDimension + 1 ) / 2;
    const unsigned int helperDim2 = OutputDimension * ( SpaceDimension + 1 ) * ( SpaceDimension + 2 ) / 2;

    /** Make a copy of the pointers to mu. The pointer will move later. */
    ScalarType * tmp_mu[ OutputDimension ];
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      tmp_mu[ j ] = mu[ j ];
    }

    /** Create a temporary sh and initialize the original. */
    InternalFloatType tmp_sh[ helperDim1 ];
    for( unsigned int n = 0; n < helperDim2; ++n )
    {
      sh[ n ] = 0.0;
    }

    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetSpatialHessian( tmp_sh, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D , hessianWeights1D );

      // Multiply by the weights
      for( unsigned int n = 0; n < helperDim1; ++n )
      {
        sh[ n ] += tmp_sh[ n ] * weights1D[ k + HelperConstVariable ];
      }

      // Multiply by the derivative weights
      for( unsigned int n = 0; n < SpaceDimension; ++n )
      {
        for( unsigned int j = 0 ; j < OutputDimension; ++j )
        {
          sh[ OutputDimension * n + helperDim1 + j ]
            += tmp_sh[ OutputDimension * n * ( n + 1 ) / 2 + j ] * derivativeWeights1D[ k + HelperConstVariable ];
        }
      }

      // Multiply by the Hessian weights
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        sh[ helperDim2 - OutputDimension + j ]
          += tmp_sh[ j ] * hessianWeights1D[ k + HelperConstVariable ];

        // move to the next mu
        tmp_mu[ j ] += bot;
      }
    }
  } // end GetSpatialHessian()

}; // end class


/** \class RecursiveBSplineTransformImplementation2
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< unsigned int OutputDimension, unsigned int SplineOrder, class TScalar >
class RecursiveBSplineTransformImplementation2< OutputDimension, 0, SplineOrder, TScalar >
{
public:

  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  typedef TScalar ScalarType;
  typedef double InternalFloatType;

  /** Typedef to know the number of indices at compile time. */
  typedef itk::RecursiveBSplineInterpolationWeightFunction<
    TScalar, OutputDimension, SplineOrder > RecursiveBSplineWeightFunctionType;
  itkStaticConstMacro( BSplineNumberOfIndices, unsigned int,
    RecursiveBSplineWeightFunctionType::NumberOfIndices );

  typedef ScalarType *  OutputPointType;
  typedef ScalarType ** CoefficientPointerVectorType;

  /** TransformPoint recursive implementation. */
  static inline void TransformPoint(
    OutputPointType opp,
    const CoefficientPointerVectorType mu, const OffsetValueType * steps, const double * weights1D )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      opp[ j ] = *(mu[ j ]);
    }
  } // end TransformPoint()


  /** TransformPoint recursive implementation. */
  static inline void TransformPoint2(
    OutputPointType opp, const CoefficientPointerVectorType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      opp[ j ] = *(mu[ j ]);
    }
  } // end TransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void GetJacobian(
    ScalarType * & jacobians, const double * weights1D, double value )
  {
    unsigned long offset = 0;
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      offset = j * BSplineNumberOfIndices * ( OutputDimension + 1 );
      *(jacobians + offset) = value;
    }
    ++jacobians;
  } // end GetJacobian()


  /** EvaluateJacobianWithImageGradientProduct recursive implementation. */
  static inline void EvaluateJacobianWithImageGradientProduct(
    ScalarType * & imageJacobian, const InternalFloatType * movingImageGradient,
    const double * weights1D, double value )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      *(imageJacobian + j * BSplineNumberOfIndices) = value * movingImageGradient[ j ];
    }
    ++imageJacobian;
  } // end EvaluateJacobianWithImageGradientProduct()


  /** ComputeNonZeroJacobianIndices recursive implementation. */
  static inline void ComputeNonZeroJacobianIndices(
    unsigned long * nzji,
    unsigned long parametersPerDim,
    unsigned long currentIndex,
    const OffsetValueType * gridOffsetTable,
    unsigned int & c )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      nzji[ c + j * BSplineNumberOfIndices ] = currentIndex + j * parametersPerDim;
    }
    ++c;
  } // end ComputeNonZeroJacobianIndices()


  /** GetSpatialJacobian recursive implementation. */
  static inline void GetSpatialJacobian(
    InternalFloatType * sj,
    const CoefficientPointerVectorType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      sj[ j ] = *(mu[ j ]);
    }
  } // end GetSpatialJacobian()


  /** GetSpatialHessian recursive implementation. */
  static inline void GetSpatialHessian(
    InternalFloatType * sh,
    const CoefficientPointerVectorType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D,
    const double * hessianWeights1D )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      sh[ j ] = *(mu[ j ]);
    }
  } // end GetSpatialHessian()

}; // end class





/** \class RecursiveBSplineTransformImplementation3
 *
 * \brief This helper class contains the actual implementation of the
 * recursive B-spline transform
 *
 * Compared to the RecursiveBSplineTransformImplementation class, this
 * class works as a vector operator, and is therefore also templated
 * over the OutputDimension.
 * Compared to RecursiveBSplineTransformImplementation2, the OutputDimension 
 * coefficients in a voxel are now adjacent in memory instead of being individually referenced through mu.
 * 
 *
 * \ingroup ITKTransform
 */

template< unsigned int OutputDimension, unsigned int SpaceDimension, unsigned int SplineOrder, class TScalar, bool doPrefetch >
class RecursiveBSplineTransformImplementation3
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  typedef TScalar ScalarType;
  typedef double InternalFloatType;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** Typedef to know the number of indices at compile time. */
  typedef itk::RecursiveBSplineInterpolationWeightFunction<
    ScalarType, OutputDimension, SplineOrder > RecursiveBSplineWeightFunctionType;
  itkStaticConstMacro( BSplineNumberOfIndices, unsigned int,
    RecursiveBSplineWeightFunctionType::NumberOfIndices );

  typedef ScalarType *  OutputPointType;
  typedef ScalarType * CoefficientPointerType;

  /** TransformPoint recursive implementation. */
  static inline void TransformPoint(
    OutputPointType opp,
    const CoefficientPointerType mu, const OffsetValueType * steps, const double * weights1D )
  {
    ScalarType * tmp_mu;
    ScalarType tmp_opp[ OutputDimension ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        tmp_opp[ j ] = 0.0;
        
      }
	  tmp_mu = mu + steps[ k + HelperConstVariable ];
      RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar , doPrefetch >
        ::TransformPoint( tmp_opp, tmp_mu, steps, weights1D );
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        opp[ j ] += tmp_opp[ j ] * weights1D[ k + HelperConstVariable ];
      }
    }
  } // end TransformPoint()


  /** TransformPoint recursive implementation. */
  static inline void TransformPoint2(
    OutputPointType opp, const CoefficientPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
	  if ( doPrefetch && SpaceDimension==2 ) {
		 _mm_prefetch( (const char *) mu , _MM_HINT_T0);//_MM_HINT_NTA
		 _mm_prefetch( ((const char *) (mu))  + 64 , _MM_HINT_T0);// one cacheline ahead
		 _mm_prefetch( ((const char *) (mu + OutputDimension*(SplineOrder+1)-1))   , _MM_HINT_T0); // last element needed.
	  }
    ScalarType * tmp_mu;
    tmp_mu = mu;

    /** Create a temporary sj and initialize the original. */
    ScalarType tmp_opp[ OutputDimension ];
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      opp[ j ] = 0.0;
    }

    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      if (doPrefetch && (SpaceDimension==2)  && (k<SplineOrder)) { // with test for SplineOrder seems to have better performance (due to less unnececary caching). Branch seems to be predicted perfectly (does not seem to have a penalty)
		  _mm_prefetch( (const char *) (tmp_mu+bot*OutputDimension) , _MM_HINT_T0); //_MM_HINT_NTA
		  _mm_prefetch( ((const char *) (tmp_mu+bot*OutputDimension))  +64 , _MM_HINT_T0); // one cacheline ahead
		  _mm_prefetch( ((const char *) (tmp_mu+bot*OutputDimension + OutputDimension*(SplineOrder+1)-1))   , _MM_HINT_T0); // last element needed.
	  }
	  RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::TransformPoint2( tmp_opp, tmp_mu, gridOffsetTable, weights1D );

      // Multiply by the weights
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        opp[ j ] += tmp_opp[ j ] * weights1D[ k + HelperConstVariable ];

      }
      // move to the next mu
      tmp_mu += bot*OutputDimension;
    }
  } // end TransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void GetJacobian(
    ScalarType * & jacobians, const double * weights1D, double value )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::GetJacobian( jacobians, weights1D, value * weights1D[ k + HelperConstVariable ] );
    }
  } // end GetJacobian()


  /** EvaluateJacobianWithImageGradientProduct recursive implementation. */
  static inline void EvaluateJacobianWithImageGradientProduct(
    ScalarType * & imageJacobian, const InternalFloatType * movingImageGradient,
    const double * weights1D, double value )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::EvaluateJacobianWithImageGradientProduct( imageJacobian, movingImageGradient, weights1D,
          value * weights1D[ k + HelperConstVariable ] );
    }
  } // end EvaluateJacobianWithImageGradientProduct()


  /** ComputeNonZeroJacobianIndices recursive implementation. */
  static inline void ComputeNonZeroJacobianIndices(
    unsigned long * nzji,
    unsigned long parametersPerDim,
    unsigned long currentIndex,
    const OffsetValueType * gridOffsetTable,
    unsigned int & c )
  {
    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::ComputeNonZeroJacobianIndices( nzji, parametersPerDim, currentIndex, gridOffsetTable, c );
      currentIndex += bot;
    }
  } // end ComputeNonZeroJacobianIndices()


  /** GetSpatialJacobian recursive implementation.
   * As an (almost) free by-product this function delivers the displacement,
   * i.e. the TransformPoint() function.
   */
  static inline void GetSpatialJacobian(
    InternalFloatType * sj,
    const CoefficientPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
    if ( SpaceDimension==2 ) {
		// prefetch in dimension 2 since in the first dimension the points are next to each other in memory. 
		// Prefetch as early as possible in this function
		 _mm_prefetch( (const char *) mu , _MM_HINT_T0);//_MM_HINT_NTA
		 _mm_prefetch( ((const char *) (mu))  + 64 , _MM_HINT_T0);
	}
	ScalarType * tmp_mu;
    tmp_mu = mu;

    /** Create a temporary sj and initialize the original. */
    InternalFloatType tmp_sj[ OutputDimension * SpaceDimension ];
    for( unsigned int n = 0; n < OutputDimension * ( SpaceDimension + 1 ); ++n )
     {
       sj[ n ] = 0.0;
     }

    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
	  if (doPrefetch && (SpaceDimension==2) && (k<SplineOrder)) {
		  // prefetch in dimension 2 since in the first dimension the points are next to each other in memory. 
		  // prefetch next line to allow sufficient time to actually fetch that data. 
		  // assume that the k<SplineOrder test is either optimized away or perfectly predicted by the CPU's branch predictor (since loop length typically is <=4) 
		  _mm_prefetch( (const char *) (tmp_mu+bot*OutputDimension) , _MM_HINT_T0); //_MM_HINT_NTA
		  _mm_prefetch( ((const char *) (tmp_mu+bot*OutputDimension))  +64 , _MM_HINT_T0);
	  }
	  RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::GetSpatialJacobian( tmp_sj, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D );

      // Multiply by the weights
      for( unsigned int n = 0; n < OutputDimension * SpaceDimension; ++n )
      {
        sj[ n ] += tmp_sj[ n ] * weights1D[ k + HelperConstVariable ];
      }

      // Multiply by the derivative weights
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        sj[ OutputDimension * SpaceDimension + j ]
          += tmp_sj[ j ] * derivativeWeights1D[ k + HelperConstVariable ];

      }
      // move to the next mu
      tmp_mu += bot*OutputDimension;
    }
  } // end GetSpatialJacobian()


  /** GetSpatialHessian recursive implementation.
   * As an (almost) free by-product this function delivers the displacement,
   * i.e. the TransformPoint() function, as well as the SpatialJacobian.
   */
  static inline void GetSpatialHessian(
    InternalFloatType * sh,
    const CoefficientPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D )   // 2nd derivative of B-spline
  {
   if ( SpaceDimension==2 ) {
		// prefetch in dimension 2 since in the first dimension the points are next to each other in memory. 
		// Prefetch as early as possible in this function
		 _mm_prefetch( (const char *) mu , _MM_HINT_T0);//_MM_HINT_NTA
		 _mm_prefetch( ((const char *) (mu))  + 64 , _MM_HINT_T0);
	}
    const unsigned int helperDim1 = OutputDimension * SpaceDimension * ( SpaceDimension + 1 ) / 2;
    const unsigned int helperDim2 = OutputDimension * ( SpaceDimension + 1 ) * ( SpaceDimension + 2 ) / 2;

    /** Make a copy of the pointers to mu. The pointer will move later. */
    ScalarType * tmp_mu;
    tmp_mu = mu;

    /** Create a temporary sh and initialize the original. */
    InternalFloatType tmp_sh[ helperDim1 ];
    for( unsigned int n = 0; n < helperDim2; ++n )
    {
      sh[ n ] = 0.0;
    }

    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      if (doPrefetch && (SpaceDimension==2) && (k<SplineOrder)) {
		  // prefetch in dimension 2 since in the first dimension the points are next to each other in memory. 
		  // prefetch next line to allow sufficient time to actually fetch that data. 
		  // assume that the k<SplineOrder test is either optimized away or perfectly predicted by the CPU's branch predictor (since loop length typically is <=4) 
		  _mm_prefetch( (const char *) (tmp_mu+bot*OutputDimension) , _MM_HINT_T0); //_MM_HINT_NTA
		  _mm_prefetch( ((const char *) (tmp_mu+bot*OutputDimension))  +64 , _MM_HINT_T0);
	  }
	  RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::GetSpatialHessian( tmp_sh, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D , hessianWeights1D );

      // Multiply by the weights
      for( unsigned int n = 0; n < helperDim1; ++n )
      {
        sh[ n ] += tmp_sh[ n ] * weights1D[ k + HelperConstVariable ];
      }

      // Multiply by the derivative weights
      for( unsigned int n = 0; n < SpaceDimension; ++n )
      {
        for( unsigned int j = 0 ; j < OutputDimension; ++j )
        {
          sh[ OutputDimension * n + helperDim1 + j ]
            += tmp_sh[ OutputDimension * n * ( n + 1 ) / 2 + j ] * derivativeWeights1D[ k + HelperConstVariable ];
        }
      }

      // Multiply by the Hessian weights
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        sh[ helperDim2 - OutputDimension + j ]
          += tmp_sh[ j ] * hessianWeights1D[ k + HelperConstVariable ];
      }

      // move to the next mu
      tmp_mu += bot*OutputDimension;
    }
  } // end GetSpatialHessian()

}; // end class


/** \class RecursiveBSplineTransformImplementation3
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< unsigned int OutputDimension, unsigned int SplineOrder, class TScalar, bool doPrefetch >
class RecursiveBSplineTransformImplementation3< OutputDimension, 0, SplineOrder, TScalar, doPrefetch >
{
public:

  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  typedef TScalar ScalarType;
  typedef double InternalFloatType;

  /** Typedef to know the number of indices at compile time. */
  typedef itk::RecursiveBSplineInterpolationWeightFunction<
    TScalar, OutputDimension, SplineOrder > RecursiveBSplineWeightFunctionType;
  itkStaticConstMacro( BSplineNumberOfIndices, unsigned int,
    RecursiveBSplineWeightFunctionType::NumberOfIndices );

  typedef ScalarType *  OutputPointType;
  typedef ScalarType * CoefficientPointerType;

  /** TransformPoint recursive implementation. */
  static inline void TransformPoint(
    OutputPointType opp,
    const CoefficientPointerType mu, const OffsetValueType * steps, const double * weights1D )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      opp[ j ] = mu[ j ];
    }
  } // end TransformPoint()


  /** TransformPoint recursive implementation. */
  static inline void TransformPoint2(
    OutputPointType opp, const CoefficientPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      opp[ j ] = mu[ j ];
    }
  } // end TransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void GetJacobian(
    ScalarType * & jacobians, const double * weights1D, double value )
  {
    unsigned long offset = 0;
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      offset = j * BSplineNumberOfIndices * ( OutputDimension + 1 );
      *(jacobians + offset) = value;
    }
    ++jacobians;
  } // end GetJacobian()


  /** EvaluateJacobianWithImageGradientProduct recursive implementation. */
  static inline void EvaluateJacobianWithImageGradientProduct(
    ScalarType * & imageJacobian, const InternalFloatType * movingImageGradient,
    const double * weights1D, double value )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      *(imageJacobian + j * BSplineNumberOfIndices) = value * movingImageGradient[ j ];
    }
    ++imageJacobian;
  } // end EvaluateJacobianWithImageGradientProduct()


  /** ComputeNonZeroJacobianIndices recursive implementation. */
  static inline void ComputeNonZeroJacobianIndices(
    unsigned long * nzji,
    unsigned long parametersPerDim,
    unsigned long currentIndex,
    const OffsetValueType * gridOffsetTable,
    unsigned int & c )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      nzji[ c + j * BSplineNumberOfIndices ] = currentIndex + j * parametersPerDim;
    }
    ++c;
  } // end ComputeNonZeroJacobianIndices()


  /** GetSpatialJacobian recursive implementation. */
  static inline void GetSpatialJacobian(
    InternalFloatType * sj,
    const CoefficientPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      sj[ j ] = mu[ j ];
    }
  } // end GetSpatialJacobian()


  /** GetSpatialHessian recursive implementation. */
  static inline void GetSpatialHessian(
    InternalFloatType * sh,
    const CoefficientPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D,
    const double * hessianWeights1D )
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      sh[ j ] = mu[ j ];
    }
  } // end GetSpatialHessian()

}; // end class



#ifdef RECURSIVEVERSION3_OPTIMIZED_SSE2
/** \class RecursiveBSplineTransformImplementation3
 *
 * \brief Define the end case for SpaceDimension = 1 for double precision.
 *  Contains heavy optimizations. IMPORTANT: assumes gridOffsetTable[0]==1
 *
 * NOTE this version demonstrates that it inconvenient to put all RecursiveBSplineTransformImplementation methods in a single class.
 * For some methods we don't do (/can do/want to do) optimizations, but now we still have to copy all code. 
 * Therefore, I (D Poot) think it actually would be better to split RecursiveBSplineTransformImplementation either per method or 
 * into groups of similar methods:
 *  - TransformPoint, GetSpatialJacobian, GetSpatialHessian
 *  - GetJacobian, ComputeNonZeroJacobianIndices, (EvaluateJacobianWithImageGradientProduct?)
 * ( - Multiply with spatial jacobian, multiply with spatial hessian)
 */

template<bool doPrefetch > class RecursiveBSplineTransformImplementation3< 3, 1, 3, double, doPrefetch>
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  typedef double TScalar;
  typedef TScalar ScalarType;
  typedef double InternalFloatType;
  static const unsigned int OutputDimension = 3;
  static const unsigned int SpaceDimension  = 1;
  static const unsigned int SplineOrder = 3;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** Typedef to know the number of indices at compile time. */
  typedef itk::RecursiveBSplineInterpolationWeightFunction<
    ScalarType, OutputDimension, SplineOrder > RecursiveBSplineWeightFunctionType;
  itkStaticConstMacro( BSplineNumberOfIndices, unsigned int,
    RecursiveBSplineWeightFunctionType::NumberOfIndices );

  typedef ScalarType *  OutputPointType;
  typedef ScalarType * CoefficientPointerType;

  /** TransformPoint recursive implementation. */
  static inline void TransformPoint(
    OutputPointType opp,
    const CoefficientPointerType mu, const OffsetValueType * steps, const double * weights1D )
  {
    ScalarType * tmp_mu;
    ScalarType tmp_opp[ OutputDimension ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        tmp_opp[ j ] = 0.0;
        
      }
	  tmp_mu = mu + steps[ k + HelperConstVariable ];
      RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::TransformPoint( tmp_opp, tmp_mu, steps, weights1D );
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        opp[ j ] += tmp_opp[ j ] * weights1D[ k + HelperConstVariable ];
      }
    }
  } // end TransformPoint()


  /** TransformPoint recursive implementation. */
  static inline void TransformPoint2(
    OutputPointType opp, const CoefficientPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    typedef __m128d vT;

	vT weights01 = _mm_loadu_pd( & weights1D[ 0 + HelperConstVariable ] );
	vT weights23 = _mm_loadu_pd( & weights1D[ 2 + HelperConstVariable ] );

	vT mu01 = _mm_loadu_pd( mu     );
	vT mu23 = _mm_loadu_pd( mu + 2 );
	vT mu45 = _mm_loadu_pd( mu + 4 );
	vT mu67 = _mm_loadu_pd( mu + 6 );
	vT mu89 = _mm_loadu_pd( mu + 8 );
	vT muAB = _mm_loadu_pd( mu +10 );

	vT res20 = _mm_add_pd( _mm_mul_pd(weights01,mu23) , _mm_mul_pd(weights23,mu89) );
	vT weights0 = _mm_unpacklo_pd(weights01, weights01);
	vT weights1 = _mm_unpackhi_pd(weights01, weights01);
	vT weights2 = _mm_unpacklo_pd(weights23, weights23);
	vT weights3 = _mm_unpackhi_pd(weights23, weights23);
	vT res01    = _mm_add_pd( _mm_mul_pd( weights0 , mu01), _mm_mul_pd( weights2, mu67 ) ); 
	vT res12    = _mm_add_pd( _mm_mul_pd( weights1 , mu45), _mm_mul_pd( weights3, muAB ) );
	res01 = _mm_add_pd( res01, _mm_shuffle_pd( res20, res12, _MM_SHUFFLE2(0, 1) ) );
	_mm_storeu_pd( opp , res01);
	opp[2] = _mm_cvtsd_f64( _mm_add_sd( res20, _mm_unpackhi_pd( res12, res12 ) ) );
  } // end TransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void GetJacobian(
    ScalarType * & jacobians, const double * weights1D, double value )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::GetJacobian( jacobians, weights1D, value * weights1D[ k + HelperConstVariable ] );
    }
  } // end GetJacobian()


  /** EvaluateJacobianWithImageGradientProduct recursive implementation. */
  static inline void EvaluateJacobianWithImageGradientProduct(
    ScalarType * & imageJacobian, const InternalFloatType * movingImageGradient,
    const double * weights1D, double value )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::EvaluateJacobianWithImageGradientProduct( imageJacobian, movingImageGradient, weights1D,
          value * weights1D[ k + HelperConstVariable ] );
    }
  } // end EvaluateJacobianWithImageGradientProduct()


  /** ComputeNonZeroJacobianIndices recursive implementation. */
  static inline void ComputeNonZeroJacobianIndices(
    unsigned long * nzji,
    unsigned long parametersPerDim,
    unsigned long currentIndex,
    const OffsetValueType * gridOffsetTable,
    unsigned int & c )
  {
    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation3< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar, doPrefetch >
        ::ComputeNonZeroJacobianIndices( nzji, parametersPerDim, currentIndex, gridOffsetTable, c );
      currentIndex += bot;
    }
  } // end ComputeNonZeroJacobianIndices()


  /** GetSpatialJacobian recursive implementation.
   * As an (almost) free by-product this function delivers the displacement,
   * i.e. the TransformPoint() function.
   */
  static inline void GetSpatialJacobian(
    InternalFloatType * sj,
    const CoefficientPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
   typedef __m128d vT;

	vT weights01 = _mm_loadu_pd( & weights1D[ 0 + HelperConstVariable ] );
	vT weights23 = _mm_loadu_pd( & weights1D[ 2 + HelperConstVariable ] );

	vT mu01 = _mm_loadu_pd( mu     );
	vT mu23 = _mm_loadu_pd( mu + 2 );
	vT mu45 = _mm_loadu_pd( mu + 4 );
	vT mu67 = _mm_loadu_pd( mu + 6 );
	vT mu89 = _mm_loadu_pd( mu + 8 );
	vT muAB = _mm_loadu_pd( mu +10 );

	vT res20 = _mm_add_pd( _mm_mul_pd(weights01,mu23) , _mm_mul_pd(weights23,mu89) );
	vT weights0 = _mm_unpacklo_pd(weights01, weights01);
	vT weights1 = _mm_unpackhi_pd(weights01, weights01);
	vT weights2 = _mm_unpacklo_pd(weights23, weights23);
	vT weights3 = _mm_unpackhi_pd(weights23, weights23);
	vT res01    = _mm_add_pd( _mm_mul_pd( weights0 , mu01), _mm_mul_pd( weights2, mu67 ) ); 
	vT res12    = _mm_add_pd( _mm_mul_pd( weights1 , mu45), _mm_mul_pd( weights3, muAB ) );
	res01 = _mm_add_pd( res01, _mm_shuffle_pd( res20, res12, _MM_SHUFFLE2(0, 1) ) );
	_mm_storeu_pd( sj , res01);
	
	vT Dweights01 = _mm_loadu_pd( & derivativeWeights1D[ 0 + HelperConstVariable ] );
	vT Dweights23 = _mm_loadu_pd( & derivativeWeights1D[ 2 + HelperConstVariable ] );
	vT Dres20 = _mm_add_pd( _mm_mul_pd(Dweights01,mu23) , _mm_mul_pd(Dweights23,mu89) );
	vT Dweights0 = _mm_unpacklo_pd(Dweights01, Dweights01);
	vT Dweights1 = _mm_unpackhi_pd(Dweights01, Dweights01);
	vT Dweights2 = _mm_unpacklo_pd(Dweights23, Dweights23);
	vT Dweights3 = _mm_unpackhi_pd(Dweights23, Dweights23);
	vT Dres01    = _mm_add_pd( _mm_mul_pd( Dweights0 , mu01), _mm_mul_pd( Dweights2, mu67 ) ); 
	vT Dres12    = _mm_add_pd( _mm_mul_pd( Dweights1 , mu45), _mm_mul_pd( Dweights3, muAB ) );

	Dres12 = _mm_add_pd( Dres12, _mm_shuffle_pd( Dres01, Dres20, _MM_SHUFFLE2(0, 1) ) );
	vT res2Dres0 = _mm_add_pd( _mm_unpacklo_pd( res20, Dres01), _mm_unpackhi_pd( res12, Dres20));
	_mm_storeu_pd( sj + 2 , res2Dres0);
	_mm_storeu_pd( sj + 4 , Dres12);
	
  } // end GetSpatialJacobian()


  /** GetSpatialHessian recursive implementation.
   * As an (almost) free by-product this function delivers the displacement,
   * i.e. the TransformPoint() function, as well as the SpatialJacobian.
   */
  static inline void GetSpatialHessian(
    InternalFloatType * sh,
    const CoefficientPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D )   // 2nd derivative of B-spline
  {
  typedef __m128d vT;

	vT weights01 = _mm_loadu_pd( & weights1D[ 0 + HelperConstVariable ] );
	vT weights23 = _mm_loadu_pd( & weights1D[ 2 + HelperConstVariable ] );

	vT mu01 = _mm_loadu_pd( mu     );
	vT mu23 = _mm_loadu_pd( mu + 2 );
	vT mu45 = _mm_loadu_pd( mu + 4 );
	vT mu67 = _mm_loadu_pd( mu + 6 );
	vT mu89 = _mm_loadu_pd( mu + 8 );
	vT muAB = _mm_loadu_pd( mu +10 );

	vT res20 = _mm_add_pd( _mm_mul_pd(weights01,mu23) , _mm_mul_pd(weights23,mu89) );
	vT weights0 = _mm_unpacklo_pd(weights01, weights01);
	vT weights1 = _mm_unpackhi_pd(weights01, weights01);
	vT weights2 = _mm_unpacklo_pd(weights23, weights23);
	vT weights3 = _mm_unpackhi_pd(weights23, weights23);
	vT res01    = _mm_add_pd( _mm_mul_pd( weights0 , mu01), _mm_mul_pd( weights2, mu67 ) ); 
	vT res12    = _mm_add_pd( _mm_mul_pd( weights1 , mu45), _mm_mul_pd( weights3, muAB ) );
	res01 = _mm_add_pd( res01, _mm_shuffle_pd( res20, res12, _MM_SHUFFLE2(0, 1) ) );
	_mm_storeu_pd( sh , res01);
	
	vT Dweights01 = _mm_loadu_pd( & derivativeWeights1D[ 0 + HelperConstVariable ] );
	vT Dweights23 = _mm_loadu_pd( & derivativeWeights1D[ 2 + HelperConstVariable ] );
	vT Dres20 = _mm_add_pd( _mm_mul_pd(Dweights01,mu23) , _mm_mul_pd(Dweights23,mu89) );
	vT Dweights0 = _mm_unpacklo_pd(Dweights01, Dweights01);
	vT Dweights1 = _mm_unpackhi_pd(Dweights01, Dweights01);
	vT Dweights2 = _mm_unpacklo_pd(Dweights23, Dweights23);
	vT Dweights3 = _mm_unpackhi_pd(Dweights23, Dweights23);
	vT Dres01    = _mm_add_pd( _mm_mul_pd( Dweights0 , mu01), _mm_mul_pd( Dweights2, mu67 ) ); 
	vT Dres12    = _mm_add_pd( _mm_mul_pd( Dweights1 , mu45), _mm_mul_pd( Dweights3, muAB ) );

	Dres12 = _mm_add_pd( Dres12, _mm_shuffle_pd( Dres01, Dres20, _MM_SHUFFLE2(0, 1) ) );
	vT res2Dres0 = _mm_add_pd( _mm_unpacklo_pd( res20, Dres01), _mm_unpackhi_pd( res12, Dres20));
	_mm_storeu_pd( sh + 2 , res2Dres0);
	_mm_storeu_pd( sh + 4 , Dres12);

	vT Hweights01 = _mm_loadu_pd( & hessianWeights1D[ 0 + HelperConstVariable ] );
	vT Hweights23 = _mm_loadu_pd( & hessianWeights1D[ 2 + HelperConstVariable ] );
	vT Hres20 = _mm_add_pd( _mm_mul_pd(Hweights01,mu23) , _mm_mul_pd(Hweights23,mu89) );
	vT Hweights0 = _mm_unpacklo_pd(Hweights01, Hweights01);
	vT Hweights1 = _mm_unpackhi_pd(Hweights01, Hweights01);
	vT Hweights2 = _mm_unpacklo_pd(Hweights23, Hweights23);
	vT Hweights3 = _mm_unpackhi_pd(Hweights23, Hweights23);
	vT Hres01    = _mm_add_pd( _mm_mul_pd( Hweights0 , mu01), _mm_mul_pd( Hweights2, mu67 ) ); 
	vT Hres12    = _mm_add_pd( _mm_mul_pd( Hweights1 , mu45), _mm_mul_pd( Hweights3, muAB ) );

	Hres01 = _mm_add_pd( Hres01, _mm_shuffle_pd( Hres20, Hres12, _MM_SHUFFLE2(0, 1) ) );
	_mm_storeu_pd( sh + 6 , Hres01);
	sh[8] = _mm_cvtsd_f64( _mm_add_sd( Hres20, _mm_unpackhi_pd( Hres12, Hres12 ) ) );

	
  } // end GetSpatialHessian()

}; // end class

#endif



/** \class RecursiveBSplineTransformImplementation4
 *
 * \brief This helper class contains the actual implementation of the
 * recursive B-spline transform
 *
 * Compared to the RecursiveBSplineTransformImplementation class, this
 * class works as a vector operator, and is therefore also templated
 * over the OutputDimension.
 * Compared to RecursiveBSplineTransformImplementation2, the OutputDimension 
 * coefficients in a voxel are now adjacent in memory instead of being individually referenced through mu.
 * Compared to RecursiveBSplineTransformImplementation3, now a SSE optimized vector output is returned in TransformPoint (return by value).
 * SSE2 optimized vector functions are used to optimize efficiency.
 *
 * \ingroup ITKTransform
 */

template< class OutputPointType, unsigned int SpaceDimension, unsigned int SplineOrder, class InputPointerType, bool doPrefetch >
class RecursiveBSplineTransformImplementation4
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );


  /** TransformPoint recursive implementation. */
  static inline OutputPointType TransformPoint2(
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D ) // , const InputPointerType prefetch_mu)
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
    InputPointerType tmp_mu = mu;
	/*InputPointerType tmp_prefetch_mu;
	if (doPrefetch)
		tmp_prefetch_mu = prefetch_mu;*/

    /** Create a temporary sj and initialize the original. */
	OutputPointType opp(0.0);
    
    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
	  opp += RecursiveBSplineTransformImplementation4< OutputPointType, SpaceDimension - 1, SplineOrder, InputPointerType, doPrefetch >
		::TransformPoint2( tmp_mu, gridOffsetTable, weights1D ) * weights1D[ k + HelperConstVariable ];
        //::TransformPoint2( tmp_mu, gridOffsetTable, weights1D , tmp_prefetch_mu ) * weights1D[ k + HelperConstVariable ];

      // move to the next mu
      tmp_mu += bot;
	  /*if (doPrefetch)
		tmp_prefetch_mu +=bot;*/
    }
	return opp;
  } // end TransformPoint()


}; // end class


/** \class RecursiveBSplineTransformImplementation4
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< class OutputPointType, unsigned int SplineOrder, class InputPointerType, bool doPrefetch >
class RecursiveBSplineTransformImplementation4< OutputPointType, 0, SplineOrder, InputPointerType, doPrefetch >
{
public:

  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */

  /** TransformPoint recursive implementation. */
  static inline OutputPointType TransformPoint2(
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D ) //, const InputPointerType prefetch_mu)
  {
	 /*if (doPrefetch)
		 _mm_prefetch( (const char *) prefetch_mu , _MM_HINT_T2 ) ;*/
     return *mu;
  } // end TransformPoint()



}; // end class



#ifdef RECURSIVEVERSION3_OPTIMIZED_SSE2
/** \class RecursiveBSplineTransformImplementation3
 *
 * \brief Define the end case for SpaceDimension = 1 for double precision.
 *  Contains heavy optimizations. IMPORTANT: assumes gridOffsetTable[0]==1
 *
 * NOTE this version demonstrates that it inconvenient to put all RecursiveBSplineTransformImplementation methods in a single class.
 * For some methods we don't do (/can do/want to do) optimizations, but now we still have to copy all code. 
 * Therefore, I (D Poot) think it actually would be better to split RecursiveBSplineTransformImplementation either per method or 
 * into groups of similar methods:
 *  - TransformPoint, GetSpatialJacobian, GetSpatialHessian
 *  - GetJacobian, ComputeNonZeroJacobianIndices, (EvaluateJacobianWithImageGradientProduct?)
 * ( - Multiply with spatial jacobian, multiply with spatial hessian)
 */

template<class OutputPointType, class InputPointerType, bool doPrefetch > class RecursiveBSplineTransformImplementation4< OutputPointType, 1, 3,  InputPointerType, doPrefetch>
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  static const unsigned int SpaceDimension = 1;
  static const unsigned int SplineOrder =3;
  static const unsigned int CACHE_LINE_SIZE = 64;
  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** TransformPoint recursive implementation. */
  static inline OutputPointType TransformPoint2(
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D ) // , const InputPointerType prefetch_mu)
  {
    OutputPointType weights0( weights1D[ 0 + HelperConstVariable ] );
	OutputPointType weights1( weights1D[ 1 + HelperConstVariable ] );
	OutputPointType weights2( weights1D[ 2 + HelperConstVariable ] );
	OutputPointType weights3( weights1D[ 3 + HelperConstVariable ] );
	/*if (doPrefetch) {
		 _mm_prefetch( ( (const char *) prefetch_mu ) + 0 * CACHE_LINE_SIZE, _MM_HINT_T2 ) ; // first byte used.
		 _mm_prefetch( ( (const char *) prefetch_mu ) + 1 * CACHE_LINE_SIZE, _MM_HINT_T2 ) ;
		 //if ( ( (const char *) prefetch_mu ) + 2 * CACHE_LINE_SIZE < lastByteUsed) // we need some compile time test to determine how many cache lines to prefetch.
		 //_mm_prefetch( ( (const char *) prefetch_mu ) + 2 * CACHE_LINE_SIZE, _MM_HINT_T2 ) ;
		 const char * lastByteUsed = ( (const char *) (prefetch_mu+4) ) - 1;
		 _mm_prefetch( lastByteUsed, _MM_HINT_T2 ) ; // last byte used.
	}*/
	return ( (*(mu+0)) * weights0 + (*(mu+1)  * weights1)) + ( (*(mu+2)) * weights2 + (*(mu+3)  * weights3));
  } // end TransformPoint()



}; // end class

/*  Surprisingly, the very specialized case below is a bit slower than the version above.

template< bool doPrefetch > class RecursiveBSplineTransformImplementation4< vec<double, 3>, 1, 3,  vecptr< double * , 3> , doPrefetch>
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   * /
  static const unsigned int SpaceDimension = 1;
  static const unsigned int SplineOrder =3;
  typedef vec<double, 3> OutputPointType ;
  typedef vecptr<  double * , 3> InputPointerType;

  /** Helper constant variable. * /
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** TransformPoint recursive implementation. * /
  static inline OutputPointType TransformPoint2(
    const InputPointerType vecmu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    typedef __m128d vT;
	const double * mu = vecmu.getRawPointer();

	vT weights01 = _mm_loadu_pd( & weights1D[ 0 + HelperConstVariable ] );
	vT weights23 = _mm_loadu_pd( & weights1D[ 2 + HelperConstVariable ] );

	vT mu01 = _mm_loadu_pd( mu     );
	vT mu23 = _mm_loadu_pd( mu + 2 );
	vT mu45 = _mm_loadu_pd( mu + 4 );
	vT mu67 = _mm_loadu_pd( mu + 6 );
	vT mu89 = _mm_loadu_pd( mu + 8 );
	vT muAB = _mm_loadu_pd( mu +10 );

	vT res20 = _mm_add_pd( _mm_mul_pd(weights01,mu23) , _mm_mul_pd(weights23,mu89) );
	vT weights0 = _mm_unpacklo_pd(weights01, weights01);
	vT weights1 = _mm_unpackhi_pd(weights01, weights01);
	vT weights2 = _mm_unpacklo_pd(weights23, weights23);
	vT weights3 = _mm_unpackhi_pd(weights23, weights23);
	vT res01    = _mm_add_pd( _mm_mul_pd( weights0 , mu01), _mm_mul_pd( weights2, mu67 ) ); 
	vT res12    = _mm_add_pd( _mm_mul_pd( weights1 , mu45), _mm_mul_pd( weights3, muAB ) );
	res01 = _mm_add_pd( res01, _mm_shuffle_pd( res20, res12, _MM_SHUFFLE2(0, 1) ) );
	vT res2 =  _mm_add_sd( res20, _mm_unpackhi_pd( res12, res12 ) ) ;
	return vec< double, 3>( res01, res2 );
  } // end TransformPoint()



}; // end class
*/

#endif



} // end namespace itk

#endif /* __itkRecursiveBSplineTransformImplementation_h */
