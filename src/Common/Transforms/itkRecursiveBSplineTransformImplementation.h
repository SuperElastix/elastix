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
 * recursive B-spline transform.
 *
 * These classes contain static inline functions for performance.
 *
 * Note: More optimized code can be found in itkRecursiveBSplineImplementation.h
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

  /** TransformPoint recursive implementation.
   * This function is only used for evaluation. A faster version is provided by
   * RecursiveBSplineTransformImplementation2::TransformPoint2
   */
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
  } // end TransformPoint()


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
 * Note: More optimized code can be found in itkRecursiveBSplineImplementation.h
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

  /** TransformPoint recursive implementation.
   * This function is only used for evaluation. It needs pre-computed steps.
   * TransformPoint2 computes them on-the-fly, and is overall faster.
   */
  static inline void TransformPoint(
    OutputPointType opp,
    const CoefficientPointerVectorType mu, const OffsetValueType * steps, const double * weights1D )
  {
    ScalarType * tmp_mu[ OutputDimension ];
    ScalarType tmp_opp[ OutputDimension ];

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      /** Initialize the temporary opp and mu. */
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        tmp_opp[ j ] = 0.0;
        tmp_mu[ j ] = mu[ j ] + steps[ k + HelperConstVariable ];
      }

      /** Recurse. */
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::TransformPoint( tmp_opp, tmp_mu, steps, weights1D );

      /** Accumulate the weights. */
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

    /** Create a temporary opp and initialize the original. */
    ScalarType tmp_opp[ OutputDimension ];
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      opp[ j ] = 0.0;
    }

    OffsetValueType bot = gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      /** Recurse. */
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::TransformPoint2( tmp_opp, tmp_mu, gridOffsetTable, weights1D );

      /** Accumulate the weights. */
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
      /** Recurse. */
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
      /** Recurse. */
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
      /** Recurse. */
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
    const double * weights1D,            // normal B-spline weights
    const double * derivativeWeights1D ) // 1st derivative of B-spline
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
      /** Recurse. */
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetSpatialJacobian( tmp_sj, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D );

      /** Accumulate the weights part. */
      for( unsigned int n = 0; n < OutputDimension * SpaceDimension; ++n )
      {
        sj[ n ] += tmp_sj[ n ] * weights1D[ k + HelperConstVariable ];
      }

      /** Accumulate the derivative weights part. */
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
   *
   * Specifically, sh is the output argument. It should be allocated with a size
   * OutputDimension * ( SpaceDimension + 1 ) * ( SpaceDimension + 2 ) / 2.
   * sh should point to allocated memory, but this function initializes sh.
   *
   * Upon return sh contains the spatial Hessian, spatial Jacobian and transformpoint. With
   * Hk = [ transformPoint     spatialJacobian'
   *        spatialJacobian    spatialHessian   ] .
   * (Hk specifies all info of dimension (element) k (< OutputDimension) of the point
   * and spatialJacobian is a vector of the derivative of this point with respect to the dimensions.)
   * The i,j (both < SpaceDimension) element of Hk is stored in:
   * i<=j : sh[ k +  OutputDimension * (i + j*(j+1)/2 ) ]
   * i>=j : sh[ k +  OutputDimension * (j + i*(i+1)/2 ) ]
   *
   * Note that we store only one of the symmetric halves of Hk.
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
      /** Recurse. */
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetSpatialHessian( tmp_sh, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D, hessianWeights1D );

      /** Accumulate the weights part. */
      for( unsigned int n = 0; n < helperDim1; ++n )
      {
        sh[ n ] += tmp_sh[ n ] * weights1D[ k + HelperConstVariable ];
      }

      /** Accumulate the derivative weights part. */
      for( unsigned int n = 0; n < SpaceDimension; ++n )
      {
        for( unsigned int j = 0 ; j < OutputDimension; ++j )
        {
          sh[ OutputDimension * n + helperDim1 + j ]
            += tmp_sh[ OutputDimension * n * ( n + 1 ) / 2 + j ] * derivativeWeights1D[ k + HelperConstVariable ];
        }
      }

      /** Accumulate the Hessian weights part. */
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        sh[ helperDim2 - OutputDimension + j ]
          += tmp_sh[ j ] * hessianWeights1D[ k + HelperConstVariable ];

        // move to the next mu
        tmp_mu[ j ] += bot;
      }
    }
  } // end GetSpatialHessian()


  /** GetJacobianOfSpatialJacobian recursive implementation. */
  static inline void GetJacobianOfSpatialJacobian(
    InternalFloatType * & jsj_out,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    InternalFloatType * jsj )
  {
    const unsigned int helperDim = OutputDimension - SpaceDimension + 1;

    /** Create a temporary jsj. Here, an additional element is needed for the Jacobian. */
    InternalFloatType tmp_jsj[ helperDim + 1 ];

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      /** Initialize the weights part of the temporary jsj. */
      for( unsigned int n = 0; n < helperDim; ++n )
      {
        tmp_jsj[ n ] = jsj[ n ] * weights1D[ k + HelperConstVariable ];
      }

      /** Initialize the derivative weights part. */
      tmp_jsj[ helperDim ] = jsj[ 0 ] * derivativeWeights1D[ k + HelperConstVariable ];

      /** Recurse. */
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetJacobianOfSpatialJacobian( jsj_out, weights1D, derivativeWeights1D, tmp_jsj );
    }
  } // end GetJacobianOfSpatialJacobian()


  /** GetJacobianOfSpatialJacobian recursive implementation. */
  static inline void GetJacobianOfSpatialJacobian(
    InternalFloatType * & jsj_out,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * directionCosines,
    InternalFloatType * jsj )
  {
    const unsigned int helperDim = OutputDimension - SpaceDimension + 1;

    /** Create a temporary jsj. Here, an additional element is needed for the Jacobian. */
    InternalFloatType tmp_jsj[ helperDim + 1 ];

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      /** Initialize the weights part of the temporary jsj. */
      for( unsigned int n = 0; n < helperDim; ++n )
      {
        tmp_jsj[ n ] = jsj[ n ] * weights1D[ k + HelperConstVariable ];
      }

      /** Initialize the derivative weights part. */
      tmp_jsj[ helperDim ] = jsj[ 0 ] * derivativeWeights1D[ k + HelperConstVariable ];

      /** Recurse. */
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetJacobianOfSpatialJacobian( jsj_out, weights1D, derivativeWeights1D, directionCosines, tmp_jsj );
    }
  } // end GetJacobianOfSpatialJacobian()


  /** GetJacobianOfSpatialHessian recursive implementation. */
  static inline void GetJacobianOfSpatialHessian(
    InternalFloatType * & jsh_out,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D,    // 2nd derivative of B-spline
    InternalFloatType * jsh )
  {
    const unsigned int helperDim   = OutputDimension - SpaceDimension;
    const unsigned int helperDimW  = ( helperDim + 1 ) * ( helperDim + 2 ) / 2;
    const unsigned int helperDimDW = helperDim + 1;

    /** Create a temporary jsh. */
    InternalFloatType tmp_jsh[ helperDimW + helperDimDW + 1 ];

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      /** Store some weights. */
      const double  w = weights1D[ k + HelperConstVariable ];
      const double dw = derivativeWeights1D[ k + HelperConstVariable ];
      const double hw = hessianWeights1D[ k + HelperConstVariable ];

      /** Initialize the weights part of the temporary jsh. */
      for( unsigned int n = 0; n < helperDimW; ++n )
      {
        //tmp_jsh[ n ] = jsh[ n ] * weights1D[ k + HelperConstVariable ];
        tmp_jsh[ n ] = jsh[ n ] * w;
      }

      /** Initialize the derivative weights part. */
      for( unsigned int n = 0; n < helperDimDW; ++n )
      {
        unsigned int nn = n * ( n + 1 ) / 2;
        tmp_jsh[ n + helperDimW ] = jsh[ nn ] * dw;
      }

      /** Initialize the Hessian weights part. */
      //tmp_jsh[ helperDimW + helperDimDW ] = jsh[ 0 ] * hessianWeights1D[ k + HelperConstVariable ];
      tmp_jsh[ helperDimW + helperDimDW ] = jsh[ 0 ] * hw;

      /** Recurse. */
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetJacobianOfSpatialHessian( jsh_out, weights1D, derivativeWeights1D, hessianWeights1D, tmp_jsh );
    }
  } // end GetJacobianOfSpatialHessian()


  /** GetJacobianOfSpatialHessian recursive implementation. */
  static inline void GetJacobianOfSpatialHessian(
    InternalFloatType * & jsh_out,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D,    // 2nd derivative of B-spline
    const double * directionCosines,
    InternalFloatType * jsh )
  {
    const unsigned int helperDim   = OutputDimension - SpaceDimension;
    const unsigned int helperDimW  = ( helperDim + 1 ) * ( helperDim + 2 ) / 2;
    const unsigned int helperDimDW = helperDim + 1;

    /** Create a temporary jsh. */
    InternalFloatType tmp_jsh[ helperDimW + helperDimDW + 1 ];

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      /** Store some weights. */
      const double  w = weights1D[ k + HelperConstVariable ];
      const double dw = derivativeWeights1D[ k + HelperConstVariable ];
      const double hw = hessianWeights1D[ k + HelperConstVariable ];

      /** Initialize the weights part of the temporary jsh. */
      for( unsigned int n = 0; n < helperDimW; ++n )
      {
        //tmp_jsh[ n ] = jsh[ n ] * weights1D[ k + HelperConstVariable ];
        tmp_jsh[ n ] = jsh[ n ] * w;
      }

      /** Initialize the derivative weights part. */
      for( unsigned int n = 0; n < helperDimDW; ++n )
      {
        unsigned int nn = n * ( n + 1 ) / 2;
        //tmp_jsh[ n + helperDimW ] = jsh[ nn ] * derivativeWeights1D[ k + HelperConstVariable ];
        tmp_jsh[ n + helperDimW ] = jsh[ nn ] * dw;
      }

      /** Initialize the Hessian weights part. */
      //tmp_jsh[ helperDimW + helperDimDW ] = jsh[ 0 ] * hessianWeights1D[ k + HelperConstVariable ];
      tmp_jsh[ helperDimW + helperDimDW ] = jsh[ 0 ] * hw;

      /** Recurse. */
      RecursiveBSplineTransformImplementation2< OutputDimension, SpaceDimension - 1, SplineOrder, TScalar >
        ::GetJacobianOfSpatialHessian( jsh_out, weights1D, derivativeWeights1D, hessianWeights1D, 
          directionCosines, tmp_jsh );
    }
  } // end GetJacobianOfSpatialHessian()


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
    const double * weights1D,            // normal B-spline weights
    const double * derivativeWeights1D ) // 1st derivative of B-spline
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
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D )   // 2nd derivative of B-spline
  {
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      sh[ j ] = *(mu[ j ]);
    }
  } // end GetSpatialHessian()


  /** GetJacobianOfSpatialJacobian recursive implementation. */
  static inline void GetJacobianOfSpatialJacobian(
    InternalFloatType * & jsj_out,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    InternalFloatType * jsj )
  {
    /** Copy the correct elements to the output.
     * Note that the first element jsj[0] is the normal Jacobian. We ignore it for now.
     * Also note that the received order is [dz, dy, dx] and that we return [dx, dy, dz].
     */
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      //*jsj_out = jsj[ j + 1 ];// flipped order
      *jsj_out = jsj[ OutputDimension - j ];
      ++jsj_out;
    }
  } // end GetJacobianOfSpatialJacobian()


  /** GetJacobianOfSpatialJacobian recursive implementation. */
  static inline void GetJacobianOfSpatialJacobian(
    InternalFloatType * & jsj_out,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * directionCosines,
    InternalFloatType * jsj )
  {
    /** Copy the correct elements to the output.
     * Note that the first element jsj[0] is the normal Jacobian. We ignore it for now.
     * Also note that the received order is [dz, dy, dx] and that we return [dx, dy, dz].
     * Below we use the fact that only one row of the jsj matrix is filled.
     */
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      //*jsj_out = jsj[ OutputDimension - j ] * directionCosines[ ( OutputDimension + 1 ) * j ]; // if diagonal only
      *jsj_out = jsj[ OutputDimension ] * directionCosines[ j ];
      for( unsigned int k = 1; k < OutputDimension; ++k )
      {
        *jsj_out += jsj[ OutputDimension - k ] * directionCosines[ k * OutputDimension + j ];
      }
      ++jsj_out;
    }
  } // end GetJacobianOfSpatialJacobian()


  /** GetJacobianOfSpatialHessian recursive implementation. */
  static inline void GetJacobianOfSpatialHessian(
    InternalFloatType * & jsh_out,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D,    // 2nd derivative of B-spline
    InternalFloatType * jsh )
  {
    /** Copy the correct elements to the output.
     * Currently only the upper triangle part. Note that the order of derivatives is swapped
     * (last dimension first in jsh):
     * i and j are the matrix indices into the outputted Hessian matrix (of which we only store the upper triangular part).
     * Note that jsh contains the upper triangular part of a matrix of size (OutputDimension+1) x (OutputDimension+1)
     */
	// First two optimized version (they also visualize a bit what happens)
	//
    if( OutputDimension == 3 )
    {
      jsh_out[ 0 ] = jsh[ 9 ];    jsh_out[ 1 ] = jsh[ 8 ];    jsh_out[ 3 ] = jsh[ 7 ];
                                  jsh_out[ 2 ] = jsh[ 5 ];    jsh_out[ 4 ] = jsh[ 4 ];
                                                              jsh_out[ 5 ] = jsh[ 2 ];
	  jsh_out += OutputDimension * ( OutputDimension + 1 ) / 2;
    }
    else if( OutputDimension == 2 )
    {
      jsh_out[ 0 ] = jsh[ 5 ];    jsh_out[ 1 ] = jsh[ 4 ];
                                  jsh_out[ 2 ] = jsh[ 2 ];
	  jsh_out += OutputDimension * ( OutputDimension + 1 ) / 2;
    }
    else // this is general (if a compiler sufficiently unrolls loops it should end up at the same as the above optimized cases. Unfortunately, compilers don't seem to be that smart)
      for( unsigned int j = 0; j < OutputDimension; ++j )
	  {
        for( unsigned int i = 0 ; i <= j; ++i )
		{
          *jsh_out = jsh[ ( OutputDimension - j ) + ( OutputDimension - i ) * ( OutputDimension - i + 1 ) / 2 ];
          ++jsh_out;
		}
      }
  } // end GetJacobianOfSpatialHessian()


  /** GetJacobianOfSpatialHessian recursive implementation. */
  static inline void GetJacobianOfSpatialHessian(
    InternalFloatType * & jsh_out,
    const double * weights1D,           // normal B-spline weights
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D,    // 2nd derivative of B-spline
    const double * directionCosines,
    InternalFloatType * jsh )
  {
    double jsh_tmp[ OutputDimension * OutputDimension ];
    double matrixProduct[ OutputDimension * OutputDimension ];

    /** Copy the correct elements to the intermediate matrix.
     * Note that in contrast to the other function, here we create the full matrix.
     */
    // First two much faster manually unrolled cases (NOTE: compile time if condition)
    if( OutputDimension == 3 )
    {
      jsh_tmp[ 0 ] = jsh[ 9 ];        jsh_tmp[ 1 ] = jsh[ 8 ];        jsh_tmp[ 2 ] = jsh[ 7 ];
      jsh_tmp[ 3 ] = jsh_tmp[ 1 ];    jsh_tmp[ 4 ] = jsh[ 5 ];        jsh_tmp[ 5 ] = jsh[ 4 ];
      jsh_tmp[ 6 ] = jsh_tmp[ 2 ];    jsh_tmp[ 7 ] = jsh_tmp[ 5 ];    jsh_tmp[ 8 ] = jsh[ 2 ];
    }
    else if( OutputDimension == 2 )
    {
      jsh_tmp[ 0 ] = jsh[ 5 ];        jsh_tmp[ 1 ] = jsh[ 4 ];
      jsh_tmp[ 2 ] = jsh_tmp[ 1 ];    jsh_tmp[ 3 ] = jsh[ 2 ];
    }
    else // this is general:
      for( unsigned int j = 0; j < OutputDimension; ++j )
      {
        for( unsigned int i = 0 ; i <= j; ++i )
    	{
    	  jsh_tmp[ j * OutputDimension + i ] = jsh[ ( OutputDimension - j ) + ( OutputDimension - i ) * ( OutputDimension - i + 1 ) / 2 ];
    	  if( i != j ) jsh_tmp[ i * OutputDimension + j ] = jsh_tmp[ j * OutputDimension + i ];
    	}
      }


    // multiply directionCosines^t * H
    for( unsigned int i = 0; i < OutputDimension; ++i ) // row
    {
      for( unsigned int j = 0; j < OutputDimension; ++j ) // column
      {
        double accum = directionCosines[ i ] * jsh_tmp[ j ];
        for( unsigned int k = 1; k < OutputDimension; ++k )
        {
          accum += directionCosines[ k * OutputDimension + i ] * jsh_tmp[ k * OutputDimension + j ];
        }
        matrixProduct[ i * OutputDimension + j ] = accum;
      }
    }

    // multiply matrixProduct * directionCosines
    for( unsigned int i = 0; i < OutputDimension; ++i ) // row
    {
      for( unsigned int j = 0; j < OutputDimension; ++j ) // column
      {
        double accum = matrixProduct[ i * OutputDimension ] * directionCosines[ j ];
        for( unsigned int k = 1; k < OutputDimension; ++k )
        {
          accum += matrixProduct[ i * OutputDimension + k ] * directionCosines[ k * OutputDimension + j ];
        }
        jsh_out[ i * OutputDimension + j ] = accum;
      }
    }

    /** Jump to the next non-empty matrix, skipping the zero matrices. */
    jsh_out += OutputDimension * OutputDimension * OutputDimension;

  } // end GetJacobianOfSpatialHessian()


}; // end class


} // end namespace itk

#endif /* __itkRecursiveBSplineTransformImplementation_h */
