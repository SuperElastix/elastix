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

template <class TFixedPointSet, class TMovingPointSet>
class ITK_TEMPLATE_EXPORT CorrespondingPointsEuclideanDistancePointMetric
  : public SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>
{
public:
  /** Standard class typedefs. */
  typedef CorrespondingPointsEuclideanDistancePointMetric                       Self;
  typedef SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet> Superclass;
  typedef SmartPointer<Self>                                                    Pointer;
  typedef SmartPointer<const Self>                                              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CorrespondingPointsEuclideanDistancePointMetric, SingleValuedPointSetToPointSetMetric);

  /** Types transferred from the base class */
  typedef typename Superclass::TransformType           TransformType;
  typedef typename Superclass::TransformPointer        TransformPointer;
  typedef typename Superclass::TransformParametersType TransformParametersType;
  typedef typename Superclass::TransformJacobianType   TransformJacobianType;

  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::DerivativeValueType        DerivativeValueType;
  typedef typename Superclass::FixedPointSetType          FixedPointSetType;
  typedef typename Superclass::MovingPointSetType         MovingPointSetType;
  typedef typename Superclass::FixedPointSetConstPointer  FixedPointSetConstPointer;
  typedef typename Superclass::MovingPointSetConstPointer MovingPointSetConstPointer;

  typedef typename Superclass::PointIterator     PointIterator;
  typedef typename Superclass::PointDataIterator PointDataIterator;

  typedef typename Superclass::InputPointType    InputPointType;
  typedef typename Superclass::OutputPointType   OutputPointType;
  typedef typename OutputPointType::CoordRepType CoordRepType;
  typedef vnl_vector<CoordRepType>               VnlVectorType;

  typedef typename Superclass::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  /**  Get the value for single valued optimizers. */
  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & Derivative) const override;

  /**  Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const override;

protected:
  CorrespondingPointsEuclideanDistancePointMetric();
  ~CorrespondingPointsEuclideanDistancePointMetric() override = default;

private:
  CorrespondingPointsEuclideanDistancePointMetric(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCorrespondingPointsEuclideanDistancePointMetric.hxx"
#endif

#endif
