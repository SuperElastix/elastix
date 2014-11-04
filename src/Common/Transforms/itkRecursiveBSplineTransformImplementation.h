/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
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
  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** TransformPoint recursive implementation. */
  static inline TScalar TransformPoint(
    const TScalar * mu, const long * steps, const double * weights1D,
    const TScalar * coefBasePointer, Array<unsigned long> & indices, unsigned int & c )
  {
    TScalar coord = 0.0;
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      const TScalar * tmp_mu = mu + steps[ k + HelperConstVariable ];
      coord += RecursiveBSplineTransformImplementation< SpaceDimension - 1, SplineOrder, TScalar >
        ::TransformPoint( tmp_mu, steps, weights1D, coefBasePointer, indices, c ) * weights1D[ k + HelperConstVariable ];
    }
    return coord;
  } // end InterpolateTransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void GetJacobian(
    TScalar * & jacobians, const double * weights1D , double value )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineTransformImplementation< SpaceDimension - 1, SplineOrder, TScalar >
        ::GetJacobian( jacobians, weights1D, value * weights1D[ k + HelperConstVariable ] );
    }
  } // end GetJacobian()


  /** SpatialJacobian recursive implementation. */
  static inline void GetSpatialJacobian(
    TScalar derivativeAndValue[],
    const TScalar * mu,
    const long * steps,
    const double * weights,
    const double * derivativeWeights )
  {
    /** derivativeAndValue length must be at least dim + 1. */
    TScalar derivativeAndValueNext[ SpaceDimension + 1 ];
    for( unsigned int n = 0; n <= SpaceDimension; ++n )
    {
      derivativeAndValue[ n ] = 0.0;
    }

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      const TScalar * tmp_mu = mu + steps[ k + HelperConstVariable ];

      RecursiveBSplineTransformImplementation< SpaceDimension - 1, SplineOrder, TScalar >
        ::GetSpatialJacobian( derivativeAndValueNext, tmp_mu, steps, weights, derivativeWeights );
      for( unsigned int n = 0; n < SpaceDimension; ++n )
      {
        derivativeAndValue[ n ] += derivativeAndValueNext[ n ] * weights[ k + HelperConstVariable ];
      }
      derivativeAndValue[ SpaceDimension ]
        += derivativeAndValueNext[ 0 ] * derivativeWeights[ k + HelperConstVariable ];
    }
  } // end SpatialJacobian()

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
  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** Typedef to know the number of indices at compile time. */
  typedef itk::RecursiveBSplineInterpolationWeightFunction<
    TScalar, OutputDimension, SplineOrder > RecursiveBSplineWeightFunctionType;
  itkStaticConstMacro( BSplineNumberOfIndices, unsigned int,
    RecursiveBSplineWeightFunctionType::NumberOfIndices );

  typedef TScalar *  OutputPointType;
  typedef TScalar ** CoefficientPointerVectorType;

  /** TransformPoint recursive implementation. */
  static inline void TransformPoint(
    OutputPointType opp,
    const CoefficientPointerVectorType mu, const OffsetValueType * steps, const double * weights1D )
  {
    TScalar * tmp_mu[ OutputDimension ];
    TScalar tmp_opp[ OutputDimension ];
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
    TScalar * tmp_mu[ OutputDimension ];
    for( unsigned int j = 0; j < OutputDimension; ++j )
    {
      tmp_mu[ j ] = mu[ j ];
    }

    TScalar tmp_opp[ OutputDimension ];
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


}; // end class


/** \class RecursiveBSplineTransformImplementation
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< unsigned int SplineOrder, class TScalar >
class RecursiveBSplineTransformImplementation< 0, SplineOrder, TScalar >
{
public:

  /** TransformPoint recursive implementation. */
  static inline TScalar TransformPoint(
    const TScalar * mu,
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
  // outerdim template parameter nodig
  static inline void GetJacobian(
    TScalar * & jacobians, const double * weights1D, double value )
  {
    *jacobians = value;//*dm/dx
	  //for( i = outerdim){ jacobians[64*i ] = value * dmdx[i]}
  	++jacobians;
  } // end GetJacobian()


  /** SpatialJacobian recursive implementation. */
  static inline void GetSpatialJacobian(
    TScalar derivativeAndValue[],
    const TScalar * mu,
    const long * steps,
    const double * weights,
    const double * derivativeWeights )
  {
    derivativeAndValue[ 0 ] = *mu;
  } // end GetSpatialJacobian()

}; // end class


/** \class RecursiveBSplineTransformImplementation2
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< unsigned int OutputDimension, unsigned int SplineOrder, class TScalar >
class RecursiveBSplineTransformImplementation2< OutputDimension, 0, SplineOrder, TScalar >
{
public:

  /** Typedef to know the number of indices at compile time. */
  typedef itk::RecursiveBSplineInterpolationWeightFunction<
    TScalar, OutputDimension, SplineOrder > RecursiveBSplineWeightFunctionType;
  itkStaticConstMacro( BSplineNumberOfIndices, unsigned int,
    RecursiveBSplineWeightFunctionType::NumberOfIndices );

  typedef TScalar *  OutputPointType;
  typedef TScalar ** CoefficientPointerVectorType;

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

}; // end class


} // end namespace itk

#endif /* __itkRecursiveBSplineTransformImplementation_h */
