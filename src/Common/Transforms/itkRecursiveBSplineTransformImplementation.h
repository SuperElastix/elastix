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

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** TransformPoint recursive implementation. */
  static inline ScalarType TransformPoint(
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

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

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
    ScalarType * tmp_mu[ OutputDimension ];
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      tmp_mu[ j ] = mu[ j ];
    }

    ScalarType tmp_opp[ OutputDimension ];
    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        tmp_opp[ j ] = 0.0;
      }

      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::TransformPoint2( tmp_opp, tmp_mu, gridOffsetTable, weights1D );
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        opp[ j ] += tmp_opp[ j ] * weights1D[ k + HelperConstVariable ];
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
    ScalarType * & imageJacobian, const ScalarType * movingImageGradient,
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
    ScalarType * sj,// ook doubles
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
    ScalarType tmp_sj[ OutputDimension * SpaceDimension ];
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
	  for( unsigned int n = 0; n < SpaceDimension*OutputDimension; ++n )
	  {
		  sj[  n ]+= tmp_sj[ n  ] * weights1D[ k + HelperConstVariable ];
	  }
      // Multiply by the derivative weights
      for( unsigned int j = 0; j < OutputDimension; ++j )
	  {
        sj[ j + OutputDimension * SpaceDimension ]
          += tmp_sj[ j ] * derivativeWeights1D[ k + HelperConstVariable ];

        // move to the next mu
        tmp_mu[ j ] += bot;
      }
    }
  } // end GetSpatialJacobian()

  /** GetSpatialHessian recursive implementation.
   * As an (almost) free by-product this function delivers the displacement,
   * i.e. the TransformPoint() function as well as the Jacobion
   */
  static inline void GetSpatialHessian(
    ScalarType * sj,// ook doubles
    const CoefficientPointerVectorType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,				// normal b-spline
    const double * derivativeWeights1D,		// 1st derivative of b-spline
    const double * hessianWeights1D)		// 2nd derivative of b-spline.
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
    ScalarType * tmp_mu[ OutputDimension ];
    for( unsigned int j = 0; j < OutputDimension; ++j ) {
      tmp_mu[ j ] = mu[ j ];
    }

    /** Create a temporary sj and initialize the original. */
    ScalarType tmp_sj[ OutputDimension * SpaceDimension * (SpaceDimension+1)/2 ];
    for( unsigned int n = 0; n < OutputDimension * ( SpaceDimension + 1 )*( SpaceDimension + 2 )/2; ++n ) {
       sj[ n ] = 0.0;
    }

    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetSpatialHessian( tmp_sj, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D , hessianWeights1D);

      // Multiply by the weights  (value and 'old' part of gradient and hessian)
	  for( unsigned int n = 0; n < SpaceDimension*(SpaceDimension+1)/2*OutputDimension; ++n ) {
		  sj[  n ] += tmp_sj[ n  ] * weights1D[ k + HelperConstVariable ];
	  }
      // Multiply by the derivative weights (new element of gradient and new part of hessian, excluding diagonal)
	  for( unsigned int n = 0; n < SpaceDimension; ++n ) {
		  for ( unsigned int j = 0 ; j < OutputDimension; ++j ) {
	  		  sj[  n *OutputDimension + j +  SpaceDimension*(SpaceDimension+1)/2*OutputDimension ]
	  		       += tmp_sj[ n*(n+1)/2* OutputDimension + j  ] * derivativeWeights1D[ k + HelperConstVariable ];
	  	  }
	  }
      for( unsigned int j = 0; j < OutputDimension; ++j )
	  {
        sj[ j + (SpaceDimension+1)*(SpaceDimension+2)/2-1)*OutputDimension ]
          += tmp_sj[ j ] * hessianWeights1D[ k + HelperConstVariable ];

        // move to the next mu
        tmp_mu[ j ] += bot;
      }
    }
  } // end GetSpatialHessian()

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

  /** TransformPoint recursive implementation. */
  static inline TScalar TransformPoint(
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
    *jacobians = value;
    ++jacobians;
  } // end GetJacobian()


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
    ScalarType * & imageJacobian, const ScalarType * movingImageGradient, const double * weights1D, double value )
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
    ScalarType * sj,
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
    ScalarType * sj,
    const CoefficientPointerVectorType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D,
    const double * hessianWeights1D)
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      sj[ j ] = *(mu[ j ]);
    }
  } // end GetSpatialHessian()

}; // end class


} // end namespace itk

#endif /* __itkRecursiveBSplineTransformImplementation_h */
