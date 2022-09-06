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
#ifndef itkImageToImageMetricWithFeatures_h
#define itkImageToImageMetricWithFeatures_h

#include "itkAdvancedImageToImageMetric.h"
#include "itkInterpolateImageFunction.h"

namespace itk
{

/** \class ImageToImageMetricWithFeatures
 * \brief Computes similarity between regions of two images.
 *
 * This base class adds functionality that makes it possible
 * to use fixed and moving image features.
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedImage,
          class TMovingImage,
          class TFixedFeatureImage = TFixedImage,
          class TMovingFeatureImage = TMovingImage>
class ITK_TEMPLATE_EXPORT ImageToImageMetricWithFeatures : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageToImageMetricWithFeatures);

  /** Standard class typedefs. */
  using Self = ImageToImageMetricWithFeatures;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageToImageMetricWithFeatures, AdvancedImageToImageMetric);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
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
  using typename Superclass::ParametersType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::MovingImageRegionType;
  using typename Superclass::ImageSamplerType;
  using typename Superclass::ImageSamplerPointer;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::InternalMaskPixelType;
  using typename Superclass::InternalMovingImageMaskType;
  using typename Superclass::MovingImageMaskInterpolatorType;
  using typename Superclass::FixedImageLimiterType;
  using typename Superclass::MovingImageLimiterType;
  using typename Superclass::FixedImageLimiterOutputType;
  using typename Superclass::MovingImageLimiterOutputType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedefs for the feature images. */
  using FixedFeatureImageType = TFixedFeatureImage;
  using FixedFeatureImagePointer = typename FixedFeatureImageType::Pointer;
  using MovingFeatureImageType = TMovingFeatureImage;
  using MovingFeatureImagePointer = typename MovingFeatureImageType::Pointer;
  using FixedFeatureImageVectorType = std::vector<FixedFeatureImagePointer>;
  using MovingFeatureImageVectorType = std::vector<MovingFeatureImagePointer>;

  /** Typedefs for the feature images interpolators. */
  using FixedFeatureInterpolatorType = InterpolateImageFunction<FixedFeatureImageType, double>;
  using MovingFeatureInterpolatorType = InterpolateImageFunction<MovingFeatureImageType, double>;
  using FixedFeatureInterpolatorPointer = typename FixedFeatureInterpolatorType::Pointer;
  using MovingFeatureInterpolatorPointer = typename MovingFeatureInterpolatorType::Pointer;
  using FixedFeatureInterpolatorVectorType = std::vector<FixedFeatureInterpolatorPointer>;
  using MovingFeatureInterpolatorVectorType = std::vector<MovingFeatureInterpolatorPointer>;

  /** Set the number of fixed feature images. */
  void
  SetNumberOfFixedFeatureImages(unsigned int arg);

  /** Get the number of fixed feature images. */
  itkGetConstMacro(NumberOfFixedFeatureImages, unsigned int);

  /** Functions to set the fixed feature images. */
  void
  SetFixedFeatureImage(unsigned int i, FixedFeatureImageType * im);

  void
  SetFixedFeatureImage(FixedFeatureImageType * im)
  {
    this->SetFixedFeatureImage(0, im);
  }


  /** Functions to get the fixed feature images. */
  const FixedFeatureImageType *
  GetFixedFeatureImage(unsigned int i) const;

  const FixedFeatureImageType *
  GetFixedFeatureImage() const
  {
    return this->GetFixedFeatureImage(0);
  }


  /** Functions to set the fixed feature interpolators. */
  void
  SetFixedFeatureInterpolator(unsigned int i, FixedFeatureInterpolatorType * interpolator);

  void
  SetFixedFeatureInterpolator(FixedFeatureInterpolatorType * interpolator)
  {
    this->SetFixedFeatureInterpolator(0, interpolator);
  }


  /** Functions to get the fixed feature interpolators. */
  const FixedFeatureInterpolatorType *
  GetFixedFeatureInterpolator(unsigned int i) const;

  const FixedFeatureInterpolatorType *
  GetFixedFeatureInterpolator() const
  {
    return this->GetFixedFeatureInterpolator(0);
  }


  /** Set the number of moving feature images. */
  void
  SetNumberOfMovingFeatureImages(unsigned int arg);

  /** Get the number of moving feature images. */
  itkGetConstMacro(NumberOfMovingFeatureImages, unsigned int);

  /** Functions to set the moving feature images. */
  void
  SetMovingFeatureImage(unsigned int i, MovingFeatureImageType * im);

  void
  SetMovingFeatureImage(MovingFeatureImageType * im)
  {
    this->SetMovingFeatureImage(0, im);
  }


  /** Functions to get the moving feature images. */
  const MovingFeatureImageType *
  GetMovingFeatureImage(unsigned int i) const;

  const MovingFeatureImageType *
  GetMovingFeatureImage() const
  {
    return this->GetMovingFeatureImage(0);
  }


  /** Functions to set the moving feature interpolators. */
  void
  SetMovingFeatureInterpolator(unsigned int i, MovingFeatureInterpolatorType * interpolator);

  void
  SetMovingFeatureInterpolator(MovingFeatureInterpolatorType * interpolator)
  {
    this->SetMovingFeatureInterpolator(0, interpolator);
  }


  /** Functions to get the moving feature interpolators. */
  const MovingFeatureInterpolatorType *
  GetMovingFeatureInterpolator(unsigned int i) const;

  const MovingFeatureInterpolatorType *
  GetMovingFeatureInterpolator() const
  {
    return this->GetMovingFeatureInterpolator(0);
  }


  /** Initialize the metric. */
  virtual void
  Initialize();

protected:
  ImageToImageMetricWithFeatures();
  virtual ~ImageToImageMetricWithFeatures() {}
  void
  PrintSelf(std::ostream & os, Indent indent) const;

  using typename Superclass::BSplineInterpolatorType;
  using BSplineInterpolatorPointer = typename BSplineInterpolatorType::Pointer;
  using BSplineFeatureInterpolatorVectorType = std::vector<BSplineInterpolatorPointer>;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageDerivativeType;
  using typename Superclass::MovingImageContinuousIndexType;

  /** Member variables. */
  unsigned int                        m_NumberOfFixedFeatureImages;
  unsigned int                        m_NumberOfMovingFeatureImages;
  FixedFeatureImageVectorType         m_FixedFeatureImages;
  MovingFeatureImageVectorType        m_MovingFeatureImages;
  FixedFeatureInterpolatorVectorType  m_FixedFeatureInterpolators;
  MovingFeatureInterpolatorVectorType m_MovingFeatureInterpolators;

  std::vector<bool>                    m_FeatureInterpolatorsIsBSpline;
  BSplineFeatureInterpolatorVectorType m_MovingFeatureBSplineInterpolators;

  /** Initialize variables for image derivative computation; this
   * method is called by Initialize.
   */
  virtual void
  CheckForBSplineFeatureInterpolators();
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageToImageMetricWithFeatures.hxx"
#endif

#endif // end #ifndef itkImageToImageMetricWithFeatures_h
