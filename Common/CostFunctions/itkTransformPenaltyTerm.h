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
#ifndef itkTransformPenaltyTerm_h
#define itkTransformPenaltyTerm_h

#include "itkAdvancedImageToImageMetric.h"

// Needed for checking for B-spline for faster implementation
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkAdvancedCombinationTransform.h"

namespace itk
{
/**
 * \class TransformPenaltyTerm
 * \brief A cost function that calculates a penalty term
 * on a transformation.
 *
 * We decided to make it an itk::ImageToImageMetric, since possibly
 * all the stuff in there is also needed for penalty terms.
 *
 * A transformation penalty terms has some extra demands on the transform.
 * Therefore, the transformation is required to be of itk::AdvancedTransform
 * type.
 *
 * \ingroup Metrics
 */

template <class TFixedImage, class TScalarType = double>
class ITK_TEMPLATE_EXPORT TransformPenaltyTerm : public AdvancedImageToImageMetric<TFixedImage, TFixedImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TransformPenaltyTerm);

  /** Standard ITK stuff. */
  using Self = TransformPenaltyTerm;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TFixedImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformPenaltyTerm, AdvancedImageToImageMetric);

  /** Typedef's inherited from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImagePointer;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImagePointer;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  // these not: use advanced transform below
  // using typename Superclass::TransformType;
  // using typename Superclass::TransformPointer;
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
  using typename Superclass::ThreaderType;
  using typename Superclass::ThreadInfoType;

  /** Typedef's for the B-spline transform. */
  using typename Superclass::CombinationTransformType;
  using typename Superclass::BSplineOrder1TransformType;
  using typename Superclass::BSplineOrder1TransformPointer;
  using typename Superclass::BSplineOrder2TransformType;
  using typename Superclass::BSplineOrder2TransformPointer;
  using typename Superclass::BSplineOrder3TransformType;
  using typename Superclass::BSplineOrder3TransformPointer;

  /** Template parameters. FixedImageType has already been taken from superclass. */
  using ScalarType = TScalarType; // \todo: not really meaningful name.

  /** Typedefs from the AdvancedTransform. */
  using TransformType = typename Superclass::AdvancedTransformType;
  using SpatialJacobianType = typename TransformType::SpatialJacobianType;
  using JacobianOfSpatialJacobianType = typename TransformType::JacobianOfSpatialJacobianType;
  using SpatialHessianType = typename TransformType::SpatialHessianType;
  using JacobianOfSpatialHessianType = typename TransformType::JacobianOfSpatialHessianType;
  using InternalMatrixType = typename TransformType::InternalMatrixType;

  /** Define the dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

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
  TransformPenaltyTerm() = default;

  /** The destructor. */
  ~TransformPenaltyTerm() override = default;

  /** A function to check if the transform is B-spline, for speedup. */
  virtual bool
  CheckForBSplineTransform2(BSplineOrder3TransformPointer & bspline) const;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTransformPenaltyTerm.hxx"
#endif

#endif // #ifndef itkTransformPenaltyTerm_h
