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
#ifndef itkNormalizedGradientCorrelationImageToImageMetric_h
#define itkNormalizedGradientCorrelationImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"
#include "itkSobelOperator.h"
#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkPoint.h"
#include "itkCastImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkOptimizer.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedRayCastInterpolateImageFunction.h"

namespace itk
{

/**
 * \class NormalizedGradientCorrelationImageToImageMetric
 * \brief An metric based on the itk::NormalizedGradientCorrelationImageToImageMetric.
 *
 *
 * \ingroup Metrics
 *
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT NormalizedGradientCorrelationImageToImageMetric
  : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(NormalizedGradientCorrelationImageToImageMetric);

  /** Standard class typedefs. */
  using Self = NormalizedGradientCorrelationImageToImageMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NormalizedGradientCorrelationImageToImageMetric, AdvancedImageToImageMetric);

/** Types transferred from the base class */
/** Work around a Visual Studio .NET bug */
#if defined(_MSC_VER) && (_MSC_VER == 1300)
  using RealType = double;
#else
  using typename Superclass::RealType;
#endif

  using typename Superclass::TransformType;
  using ScalarType = typename TransformType::ScalarType;
  using typename Superclass::TransformPointer;
  using TransformConstPointer = typename TransformType::ConstPointer;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::InterpolatorType;
  using InterpolatorPointer = typename InterpolatorType::Pointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImageRegionType;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::MovingImagePointer;
  using FixedImagePixelType = typename TFixedImage::PixelType;
  using MovedImagePixelType = typename TMovingImage::PixelType;
  using ScalesType = typename Optimizer::ScalesType;

  itkStaticConstMacro(FixedImageDimension, unsigned int, TFixedImage::ImageDimension);

  /** Types for transforming the moving image */
  using CombinationTransformType = typename itk::AdvancedCombinationTransform<ScalarType, FixedImageDimension>;
  using CombinationTransformPointer = typename CombinationTransformType::Pointer;
  using TransformedMovingImageType = itk::Image<FixedImagePixelType, Self::FixedImageDimension>;
  using MaskImageType = itk::Image<unsigned char, Self::FixedImageDimension>;
  using MaskImageTypePointer = typename MaskImageType::Pointer;
  using TransformMovingImageFilterType = itk::ResampleImageFilter<MovingImageType, TransformedMovingImageType>;
  using TransformMovingImageFilterPointer = typename TransformMovingImageFilterType::Pointer;
  using RayCastInterpolatorType = typename itk::AdvancedRayCastInterpolateImageFunction<MovingImageType, ScalarType>;
  using RayCastInterpolatorPointer = typename RayCastInterpolatorType::Pointer;

  /** Sobel filters to compute the gradients of the Fixed Image */
  using FixedGradientImageType = itk::Image<RealType, Self::FixedImageDimension>;
  using CastFixedImageFilterType = itk::CastImageFilter<FixedImageType, FixedGradientImageType>;
  using CastFixedImageFilterPointer = typename CastFixedImageFilterType::Pointer;
  using FixedGradientPixelType = typename FixedGradientImageType::PixelType;

  /** Sobel filters to compute the gradients of the Moved Image */
  itkStaticConstMacro(MovedImageDimension, unsigned int, MovingImageType::ImageDimension);
  using MovedGradientImageType = itk::Image<RealType, Self::MovedImageDimension>;
  using CastMovedImageFilterType = itk::CastImageFilter<TransformedMovingImageType, MovedGradientImageType>;
  using CastMovedImageFilterPointer = typename CastMovedImageFilterType::Pointer;
  using MovedGradientPixelType = typename MovedGradientImageType::PixelType;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & derivative) const override;

  /**  Get the value for single valued optimizers. */
  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /**  Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                derivative) const override;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   */
  void
  Initialize() override;

  /** Write gradient images to a files for debugging purposes. */
  void
  WriteGradientImagesToFiles() const;

  /** Set/Get Scales  */
  itkSetMacro(Scales, ScalesType);
  itkGetConstReferenceMacro(Scales, ScalesType);

  /** Set/Get the value of Delta used for computing derivatives by finite
   * differences in the GetDerivative() method.
   */
  itkSetMacro(DerivativeDelta, double);
  itkGetConstReferenceMacro(DerivativeDelta, double);

  /** Set the parameters defining the Transform. */
  void
  SetTransformParameters(const TransformParametersType & parameters) const;

protected:
  NormalizedGradientCorrelationImageToImageMetric();
  ~NormalizedGradientCorrelationImageToImageMetric() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Compute the mean of the fixed and moved image gradients. */
  void
  ComputeMeanMovedGradient() const;

  void
  ComputeMeanFixedGradient() const;

  /** Compute the similarity measure  */
  MeasureType
  ComputeMeasure(const TransformParametersType & parameters) const;

  using FixedSobelFilter = NeighborhoodOperatorImageFilter<FixedGradientImageType, FixedGradientImageType>;
  using MovedSobelFilter = NeighborhoodOperatorImageFilter<MovedGradientImageType, MovedGradientImageType>;

private:
  ScalesType                  m_Scales;
  double                      m_DerivativeDelta;
  CombinationTransformPointer m_CombinationTransform;

  /** The mean of the moving image gradients. */
  mutable MovedGradientPixelType m_MeanMovedGradient[MovedImageDimension];

  /** The mean of the fixed image gradients. */
  mutable FixedGradientPixelType m_MeanFixedGradient[FixedImageDimension];

  /** The filter for transforming the moving images. */
  TransformMovingImageFilterPointer m_TransformMovingImageFilter;

  /** The Sobel gradients of the fixed image */
  CastFixedImageFilterPointer m_CastFixedImageFilter;

  SobelOperator<FixedGradientPixelType, Self::FixedImageDimension> m_FixedSobelOperators[FixedImageDimension];

  typename FixedSobelFilter::Pointer m_FixedSobelFilters[Self::FixedImageDimension];

  ZeroFluxNeumannBoundaryCondition<MovedGradientImageType> m_MovedBoundCond;
  ZeroFluxNeumannBoundaryCondition<FixedGradientImageType> m_FixedBoundCond;

  /** The Sobel gradients of the moving image */
  CastMovedImageFilterPointer                                      m_CastMovedImageFilter;
  SobelOperator<MovedGradientPixelType, Self::MovedImageDimension> m_MovedSobelOperators[MovedImageDimension];

  typename MovedSobelFilter::Pointer m_MovedSobelFilters[Self::MovedImageDimension];
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkNormalizedGradientCorrelationImageToImageMetric.hxx"
#endif

#endif
