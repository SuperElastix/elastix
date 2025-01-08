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
#ifndef itkCorrespondingPointsEuclideanDistancePointMetric_hxx
#define itkCorrespondingPointsEuclideanDistancePointMetric_hxx

#include "itkCorrespondingPointsEuclideanDistancePointMetric.h"

namespace itk
{

/**
 * ******************* GetValue *******************
 */

template <typename TFixedPointSet, typename TMovingPointSet>
auto
CorrespondingPointsEuclideanDistancePointMetric<TFixedPointSet, TMovingPointSet>::GetValue(
  const ParametersType & parameters) const -> MeasureType
{
  const auto & fixedPoints = this->Superclass::GetFixedPoints();
  const auto & movingPoints = this->Superclass::GetMovingPoints();

  /** Initialize some variables. */
  Superclass::m_NumberOfPointsCounted = 0;
  MeasureType measure{};

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters(parameters);

  /** Create iterator. */
  auto pointItMoving = movingPoints.cbegin();

  /** Loop over the corresponding points. */
  for (const OutputPointType & fixedPoint : fixedPoints)
  {
    /** Get the current corresponding moving point. */
    const InputPointType & movingPoint = *pointItMoving;

    /** Transform point and check if it is inside the B-spline support region. */
    // bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
    const OutputPointType mappedPoint = Superclass::m_Transform->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    if ((Superclass::m_MovingImageMask == nullptr) || Superclass::m_MovingImageMask->IsInsideInWorldSpace(mappedPoint))
    {
      Superclass::m_NumberOfPointsCounted++;
      const auto diffVector = movingPoint - mappedPoint;
      measure += diffVector.GetNorm();

    } // end if sampleOk

    ++pointItMoving;

  } // end loop over all corresponding points

  return measure / Superclass::m_NumberOfPointsCounted;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <typename TFixedPointSet, typename TMovingPointSet>
void
CorrespondingPointsEuclideanDistancePointMetric<TFixedPointSet, TMovingPointSet>::GetDerivative(
  const ParametersType & parameters,
  DerivativeType &       derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue{};
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <typename TFixedPointSet, typename TMovingPointSet>
void
CorrespondingPointsEuclideanDistancePointMetric<TFixedPointSet, TMovingPointSet>::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  using VnlVectorType = vnl_vector<typename OutputPointType::CoordinateType>;

  const auto & fixedPoints = this->Superclass::GetFixedPoints();
  const auto & movingPoints = this->Superclass::GetMovingPoints();

  /** Initialize some variables */
  Superclass::m_NumberOfPointsCounted = 0;
  MeasureType measure{};
  derivative.set_size(this->GetNumberOfParameters());
  derivative.Fill(0.0);
  NonZeroJacobianIndicesType nzji(Superclass::m_Transform->GetNumberOfNonZeroJacobianIndices());
  TransformJacobianType      jacobian;

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
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Create iterator. */
  auto pointItMoving = movingPoints.cbegin();

  /** Loop over the corresponding points. */
  for (const OutputPointType & fixedPoint : fixedPoints)
  {
    /** Get the current corresponding moving point. */
    const InputPointType & movingPoint = *pointItMoving;

    /** Transform point and check if it is inside the B-spline support region. */
    // bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
    const OutputPointType mappedPoint = Superclass::m_Transform->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    if ((Superclass::m_MovingImageMask == nullptr) || Superclass::m_MovingImageMask->IsInsideInWorldSpace(mappedPoint))
    {
      Superclass::m_NumberOfPointsCounted++;

      /** Get the TransformJacobian dT/dmu. */
      // this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
      Superclass::m_Transform->GetJacobian(fixedPoint, jacobian, nzji);

      const auto  diffVector = movingPoint - mappedPoint;
      MeasureType distance = diffVector.GetNorm();
      measure += distance;

      /** Calculate the contributions to the derivatives with respect to each parameter. */
      if (distance > std::numeric_limits<MeasureType>::epsilon())
      {
        VnlVectorType diff_2 = (diffVector / distance).GetVnlVector();
        if (nzji.size() == this->GetNumberOfParameters())
        {
          /** Loop over all Jacobians. */
          derivative -= diff_2 * jacobian;
        }
        else
        {
          /** Only pick the nonzero Jacobians. */
          for (unsigned int i = 0; i < nzji.size(); ++i)
          {
            const unsigned int index = nzji[i];
            VnlVectorType      column = jacobian.get_column(i);
            derivative[index] -= dot_product(diff_2, column);
          }
        }
      } // end if distance != 0

    } // end if sampleOk

    ++pointItMoving;

  } // end loop over all corresponding points

  /** Check if enough samples were valid. */
  //   this->CheckNumberOfSamples(
  //     fixedPointSet->GetNumberOfPoints(), Superclass::m_NumberOfPointsCounted );

  /** Copy the measure to value. */
  value = measure;
  if (Superclass::m_NumberOfPointsCounted > 0)
  {
    derivative /= Superclass::m_NumberOfPointsCounted;
    value = measure / Superclass::m_NumberOfPointsCounted;
  }

} // end GetValueAndDerivative()


} // end namespace itk

#endif // end #ifndef itkCorrespondingPointsEuclideanDistancePointMetric_hxx
