/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkCorrespondingPointsEuclideanDistancePointMetric_hxx
#define __itkCorrespondingPointsEuclideanDistancePointMetric_hxx

#include "itkCorrespondingPointsEuclideanDistancePointMetric.h"


namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
CorrespondingPointsEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::CorrespondingPointsEuclideanDistancePointMetric()
{
} // end Constructor


/**
 * ******************* GetValue *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
typename CorrespondingPointsEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>::MeasureType
CorrespondingPointsEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::GetValue( const TransformParametersType & parameters ) const
{
  /** Sanity checks. */
  FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();
  if ( !fixedPointSet )
  {
    itkExceptionMacro( << "Fixed point set has not been assigned" );
  }

  MovingPointSetConstPointer movingPointSet = this->GetMovingPointSet();
  if ( !movingPointSet )
  {
    itkExceptionMacro( << "Moving point set has not been assigned" );
  }

  /** Initialize some variables. */
  this->m_NumberOfPointsCounted = 0;
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  InputPointType movingPoint;
  OutputPointType fixedPoint, mappedPoint;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Create iterators. */
  PointIterator pointItFixed = fixedPointSet->GetPoints()->Begin();
  PointIterator pointItMoving = movingPointSet->GetPoints()->Begin();
  PointIterator pointEnd = fixedPointSet->GetPoints()->End();

  /** Loop over the corresponding points. */
  while ( pointItFixed != pointEnd )
  {
    /** Get the current corresponding points. */
    fixedPoint = pointItFixed.Value();
    movingPoint = pointItMoving.Value();

    /** Transform point and check if it is inside the B-spline support region. */
    //bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
    mappedPoint = this->m_Transform->TransformPoint( fixedPoint );

    /** Check if point is inside mask. */
    bool sampleOk = true;
    if ( sampleOk )
    {
      //sampleOk = this->IsInsideMovingMask( mappedPoint );
      if ( this->m_MovingImageMask.IsNotNull() )
      {
        sampleOk = this->m_MovingImageMask->IsInside( mappedPoint );
      }
    }

    if ( sampleOk )
    {
      this->m_NumberOfPointsCounted++;

      VnlVectorType diffPoint = ( movingPoint - mappedPoint ).GetVnlVector();
      measure += diffPoint.magnitude();

    } // end if sampleOk

    ++pointItFixed;
    ++pointItMoving;

  } // end loop over all corresponding points

  return measure / this->m_NumberOfPointsCounted;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
CorrespondingPointsEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::GetDerivative( const TransformParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
CorrespondingPointsEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::GetValueAndDerivative(const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Sanity checks. */
  FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();
  if ( !fixedPointSet )
  {
    itkExceptionMacro( << "Fixed point set has not been assigned" );
  }

  MovingPointSetConstPointer movingPointSet = this->GetMovingPointSet();
  if ( !movingPointSet )
  {
    itkExceptionMacro( << "Moving point set has not been assigned" );
  }

  /** Initialize some variables */
  this->m_NumberOfPointsCounted = 0;
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
  NonZeroJacobianIndicesType nzji(
    this->m_Transform->GetNumberOfNonZeroJacobianIndices() );
  TransformJacobianType jacobian;

  InputPointType movingPoint;
  OutputPointType fixedPoint, mappedPoint;

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Create iterators. */
  PointIterator pointItFixed = fixedPointSet->GetPoints()->Begin();
  PointIterator pointItMoving = movingPointSet->GetPoints()->Begin();
  PointIterator pointEnd = fixedPointSet->GetPoints()->End();

  /** Loop over the corresponding points. */
  while ( pointItFixed != pointEnd )
  {
    /** Get the current corresponding points. */
    fixedPoint = pointItFixed.Value();
    movingPoint = pointItMoving.Value();

    /** Transform point and check if it is inside the B-spline support region. */
    //bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
    mappedPoint = this->m_Transform->TransformPoint( fixedPoint );

    /** Check if point is inside mask. */
    bool sampleOk = true;
    if ( sampleOk )
    {
      //sampleOk = this->IsInsideMovingMask( mappedPoint );
      if ( this->m_MovingImageMask.IsNotNull() )
      {
        sampleOk = this->m_MovingImageMask->IsInside( mappedPoint );
      }
    }

    if ( sampleOk )
    {
      this->m_NumberOfPointsCounted++;

      /** Get the TransformJacobian dT/dmu. */
      //this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
      this->m_Transform->GetJacobian( fixedPoint, jacobian, nzji );

      VnlVectorType diffPoint = ( movingPoint - mappedPoint ).GetVnlVector();
      MeasureType distance = diffPoint.magnitude();
      measure += distance;

      /** Calculate the contributions to the derivatives with respect to each parameter. */
      if ( distance > vcl_numeric_limits< MeasureType >::epsilon() )
      {
        VnlVectorType diff_2 = diffPoint / distance;
        if ( nzji.size() == this->GetNumberOfParameters() )
        {
          /** Loop over all Jacobians. */
          derivative -= diff_2 * jacobian;
        }
        else
        {
          /** Only pick the nonzero Jacobians. */
          for ( unsigned int i = 0; i < nzji.size(); ++i )
          {
            const unsigned int index = nzji[ i ];
            VnlVectorType column = jacobian.get_column( i );
            derivative[ index ] -= dot_product( diff_2, column );
          }
        }
      } // end if distance != 0

    } // end if sampleOk

    ++pointItFixed;
    ++pointItMoving;

  } // end loop over all corresponding points

  /** Check if enough samples were valid. */
//   this->CheckNumberOfSamples(
//     fixedPointSet->GetNumberOfPoints(), this->m_NumberOfPointsCounted );

  /** Copy the measure to value. */
  value = measure;
  if ( this->m_NumberOfPointsCounted > 0 )
  {
    derivative /= this->m_NumberOfPointsCounted;
    value = measure / this->m_NumberOfPointsCounted;
  }

} // end GetValueAndDerivative()


} // end namespace itk


#endif // end #ifndef __itkCorrespondingPointsEuclideanDistancePointMetric_hxx
