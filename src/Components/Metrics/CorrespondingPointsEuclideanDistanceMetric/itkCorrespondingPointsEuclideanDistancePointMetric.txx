/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkCorrespondingPointsEuclideanDistancePointMetric.txx,v $
  Language:  C++
  Date:      $Date: 2008-12-17 18:52:03 $
  Version:   $Revision: 1.6 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCorrespondingPointsEuclideanDistancePointMetric_txx
#define __itkCorrespondingPointsEuclideanDistancePointMetric_txx

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
      MeasureType distance = diffPoint.magnitude();
//      MeasureType distance = diffPoint.GetVnlVector().squared_magnitude();
//       if ( !this->m_ComputeSquaredDistance )
//       {
//         distance = vcl_sqrt( distance );
//       }

      measure += distance;

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

      /** Get the TransformJacobian dT/dmu. */
      //this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
      this->m_Transform->GetJacobian( fixedPoint, jacobian, nzji );

      VnlVectorType diffPoint = ( movingPoint - mappedPoint ).GetVnlVector();
      MeasureType distance = diffPoint.magnitude();
      //MeasureType distance = diffPoint.squared_magnitude();
//       if ( !this->m_ComputeSquaredDistance )
//       {
//         distance = vcl_sqrt( distance );
//       }
      measure += distance;

      /** Calculate the contributions to the derivatives with respect to each parameter. */
      const MeasureType diff_2 = distance * 2.0;
      if ( nzji.size() == this->GetNumberOfParameters() )
      {
        /** Loop over all Jacobians. */
        VnlVectorType vec = diffPoint * jacobian;
        derivative -= diff_2 * vec;
      }
      else
      {
        /** Only pick the nonzero Jacobians. */
        for ( unsigned int i = 0; i < nzji.size(); ++i )
        {
          const unsigned int index = nzji[ i ];
          VnlVectorType column = jacobian.get_column( i );
          derivative[ index ] -= diff_2 * dot_product( diffPoint, column );
        }
      }

    } // end if sampleOk

    ++pointItFixed;
    ++pointItMoving;

  } // end loop over all corresponding points

  /** Copy the measure to value. */
  derivative /= this->m_NumberOfPointsCounted;
  value = measure / this->m_NumberOfPointsCounted;

} // end GetValueAndDerivative()


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
CorrespondingPointsEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
//
//   if ( this->m_ComputeSquaredDistance )
//   {
//     os << indent << "m_ComputeSquaredDistance: True"<< std::endl;
//   }
//   else
//   {
//     os << indent << "m_ComputeSquaredDistance: False"<< std::endl;
//   }
} // end PrintSelf()


} // end namespace itk


#endif // end #ifndef __itkCorrespondingPointsEuclideanDistancePointMetric_txx
