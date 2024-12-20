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
#ifndef itkCorrespondingPointsEuclideanDistancePointMetric_h
#define itkCorrespondingPointsEuclideanDistancePointMetric_h

#include "itkSingleValuedPointSetToPointSetMetric.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"

namespace itk
{

/** \class CorrespondingPointsEuclideanDistancePointMetric
 * \brief Computes the Euclidean distance between a moving point-set
 *  and a fixed point-set.
 *  Correspondence is needed.
 *
 *
 * \ingroup RegistrationMetrics
 */

template <typename TFixedPointSet, typename TMovingPointSet>
class ITK_TEMPLATE_EXPORT CorrespondingPointsEuclideanDistancePointMetric
  : public SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CorrespondingPointsEuclideanDistancePointMetric);

  /** Standard class typedefs. */
  using Self = CorrespondingPointsEuclideanDistancePointMetric;
  using Superclass = SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CorrespondingPointsEuclideanDistancePointMetric);

  /** Types transferred from the base class */
  using typename Superclass::ParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::FixedPointSetType;
  using typename Superclass::MovingPointSetType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /**  Get the value for single valued optimizers. */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const ParametersType & parameters, DerivativeType & Derivative) const override;

  /**  Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          Value,
                        DerivativeType &       Derivative) const override;

protected:
  CorrespondingPointsEuclideanDistancePointMetric() = default;
  ~CorrespondingPointsEuclideanDistancePointMetric() override = default;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCorrespondingPointsEuclideanDistancePointMetric.hxx"
#endif

#endif
