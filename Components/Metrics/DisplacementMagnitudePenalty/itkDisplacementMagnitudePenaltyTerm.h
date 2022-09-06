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
#ifndef itkDisplacementMagnitudePenaltyTerm_h
#define itkDisplacementMagnitudePenaltyTerm_h

#include "itkTransformPenaltyTerm.h"

namespace itk
{

/**
 * \class DisplacementMagnitudePenaltyTerm
 * \brief A cost function that calculates \f$||T(x)-x||^2\f$.
 *
 * \ingroup Metrics
 */

template <class TFixedImage, class TScalarType>
class ITK_TEMPLATE_EXPORT DisplacementMagnitudePenaltyTerm : public TransformPenaltyTerm<TFixedImage, TScalarType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DisplacementMagnitudePenaltyTerm);

  /** Standard ITK stuff. */
  using Self = DisplacementMagnitudePenaltyTerm;
  using Superclass = TransformPenaltyTerm<TFixedImage, TScalarType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DisplacementMagnitudePenaltyTerm, TransformPenaltyTerm);

  /** Typedefs inherited from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImagePointer;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImagePointer;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::RealType;
  using typename Superclass::GradientPixelType;
  using typename Superclass::GradientImageType;
  using typename Superclass::GradientImagePointer;
  using typename Superclass::GradientImageFilterType;
  using typename Superclass::GradientImageFilterPointer;
  using typename Superclass::FixedImageMaskType;
  using typename Superclass::FixedImageMaskPointer;
  using typename Superclass::MovingImageMaskType;
  using typename Superclass::MovingImageMaskPointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;
  using typename Superclass::ParametersType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::ScalarType;

  /** Typedefs from the AdvancedTransform. */
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** Define the dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** Get the penalty term value.
   * \f[ Value = 1/N sum_x ||T(x) - x||^2 \f]
   */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /** Get the penalty term derivative.
   * Simply calls GetValueAndDerivative and returns the derivative. */
  void
  GetDerivative(const ParametersType & parameters, DerivativeType & derivative) const override;

  /** Get the penalty term value and derivative.
   * \f[ Value = C(\mu) = 1/N sum_x ||T_{\mu}(x) - x||^2 \f]
   * \f[ Derivative = \frac{\partial C}{\partial\mu} = 2/N sum_x (T_{\mu}(x)-x)' \frac{\partial T}{\partial \mu} \f]
   */
  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          value,
                        DerivativeType &       derivative) const override;

protected:
  /** Typedefs for indices and points. */
  using typename Superclass::FixedImageIndexType;
  using typename Superclass::FixedImageIndexValueType;
  using typename Superclass::MovingImageIndexType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageContinuousIndexType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /** The constructor. */
  DisplacementMagnitudePenaltyTerm();

  /** The destructor. */
  ~DisplacementMagnitudePenaltyTerm() override = default;

  /** PrintSelf. *
  void PrintSelf( std::ostream& os, Indent indent ) const;*/
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkDisplacementMagnitudePenaltyTerm.hxx"
#endif

#endif // #ifndef itkDisplacementMagnitudePenaltyTerm_h
