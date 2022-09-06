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
#ifndef itkAdvancedMeanSquaresImageToImageMetric_h
#define itkAdvancedMeanSquaresImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"

#include "itkSmoothingRecursiveGaussianImageFilter.h"   // needed for SelfHessian
#include "itkImageGridSampler.h"                        // needed for SelfHessian
#include "itkNearestNeighborInterpolateImageFunction.h" // needed for SelfHessian

namespace itk
{

/** \class AdvancedMeanSquaresImageToImageMetric
 * \brief Compute Mean square difference between two images, based on AdvancedImageToImageMetric...
 *
 * This Class is templated over the type of the fixed and moving
 * images to be compared.
 *
 * This metric computes the sum of squared differenced between pixels in
 * the moving image and pixels in the fixed image. The spatial correspondance
 * between both images is established through a Transform. Pixel values are
 * taken from the Moving image. Their positions are mapped to the Fixed image
 * and result in general in non-grid position on it. Values at these non-grid
 * position of the Fixed image are interpolated using a user-selected Interpolator.
 *
 * This implementation of the MeanSquareDifference is based on the
 * AdvancedImageToImageMetric, which means that:
 * \li It uses the ImageSampler-framework
 * \li It makes use of the compact support of B-splines, in case of B-spline transforms.
 * \li Image derivatives are computed using either the B-spline interpolator's implementation
 * or by nearest neighbor interpolation of a precomputed central difference image.
 * \li A minimum number of samples that should map within the moving image (mask) can be specified.
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT AdvancedMeanSquaresImageToImageMetric
  : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedMeanSquaresImageToImageMetric);

  /** Standard class typedefs. */
  using Self = AdvancedMeanSquaresImageToImageMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedMeanSquaresImageToImageMetric, AdvancedImageToImageMetric);

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
  using typename Superclass::NumberOfParametersType;
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
  using typename Superclass::MovingImageRegionType;
  using typename Superclass::ImageSamplerType;
  using typename Superclass::ImageSamplerPointer;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::FixedImageLimiterType;
  using typename Superclass::MovingImageLimiterType;
  using typename Superclass::FixedImageLimiterOutputType;
  using typename Superclass::MovingImageLimiterOutputType;
  using typename Superclass::MovingImageDerivativeScalesType;
  using typename Superclass::HessianValueType;
  using typename Superclass::HessianType;
  using typename Superclass::ThreaderType;
  using typename Superclass::ThreadInfoType;

  using typename Superclass::FixedImageMaskSpatialObject2Type;
  using typename Superclass::MovingImageMaskSpatialObject2Type;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Get the value for single valued optimizers. */
  virtual MeasureType
  GetValueSingleThreaded(const TransformParametersType & parameters) const;

  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & derivative) const override;

  /** Get value and derivative. */
  void
  GetValueAndDerivativeSingleThreaded(const TransformParametersType & parameters,
                                      MeasureType &                   value,
                                      DerivativeType &                derivative) const;

  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   value,
                        DerivativeType &                derivative) const override;

  /** Experimental feature: compute SelfHessian */
  void
  GetSelfHessian(const TransformParametersType & parameters, HessianType & H) const override;

  /** Default: 1.0 mm */
  itkSetMacro(SelfHessianSmoothingSigma, double);
  itkGetConstMacro(SelfHessianSmoothingSigma, double);

  /** Default: 1.0 mm */
  itkSetMacro(SelfHessianNoiseRange, double);
  itkGetConstMacro(SelfHessianNoiseRange, double);

  /** Default: 100000 */
  itkSetMacro(NumberOfSamplesForSelfHessian, unsigned int);
  itkGetConstMacro(NumberOfSamplesForSelfHessian, unsigned int);

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation
   * \li Estimate the normalization factor, if asked for.  */
  void
  Initialize() override;

  /** Set/Get whether to normalize the mean squares measure.
   * This divides the MeanSquares by a factor (range/10)^2,
   * where range represents the maximum gray value range of the
   * images. Based on the ad hoc assumption that range/10 is the
   * maximum average difference that will be observed.
   * Dividing by range^2 sounds less ad hoc, but will yield
   * very small values. */
  itkSetMacro(UseNormalization, bool);
  itkGetConstMacro(UseNormalization, bool);

  /** If the compiler supports OpenMP, this flag specifies whether
   * or not to use it. For this metric we have an OpenMP variant for
   * GetValueAndDerivative(). It is also used at other places.
   * Note that MS Visual Studio and gcc support OpenMP.
   */
  itkSetMacro(UseOpenMP, bool);

protected:
  AdvancedMeanSquaresImageToImageMetric();
  ~AdvancedMeanSquaresImageToImageMetric() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  using typename Superclass::FixedImageIndexType;
  using typename Superclass::FixedImageIndexValueType;
  using typename Superclass::MovingImageIndexType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageContinuousIndexType;
  using typename Superclass::BSplineInterpolatorType;
  using typename Superclass::CentralDifferenceGradientFilterType;
  using typename Superclass::MovingImageDerivativeType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /** Protected typedefs for SelfHessian */
  using SmootherType = SmoothingRecursiveGaussianImageFilter<FixedImageType, FixedImageType>;
  using FixedImageInterpolatorType = BSplineInterpolateImageFunction<FixedImageType, CoordinateRepresentationType>;
  using DummyFixedImageInterpolatorType =
    NearestNeighborInterpolateImageFunction<FixedImageType, CoordinateRepresentationType>;
  using SelfHessianSamplerType = ImageGridSampler<FixedImageType>;

  double m_NormalizationFactor;

  /** Compute a pixel's contribution to the measure and derivatives;
   * Called by GetValueAndDerivative(). */
  void
  UpdateValueAndDerivativeTerms(const RealType                     fixedImageValue,
                                const RealType                     movingImageValue,
                                const DerivativeType &             imageJacobian,
                                const NonZeroJacobianIndicesType & nzji,
                                MeasureType &                      measure,
                                DerivativeType &                   deriv) const;

  /** Compute a pixel's contribution to the SelfHessian;
   * Called by GetSelfHessian(). */
  void
  UpdateSelfHessianTerms(const DerivativeType &             imageJacobian,
                         const NonZeroJacobianIndicesType & nzji,
                         HessianType &                      H) const;

  /** Get value for each thread. */
  inline void
  ThreadedGetValue(ThreadIdType threadID) override;

  /** Gather the values from all threads. */
  inline void
  AfterThreadedGetValue(MeasureType & value) const override;

  /** Get value and derivatives for each thread. */
  inline void
  ThreadedGetValueAndDerivative(ThreadIdType threadID) override;

  /** Gather the values and derivatives from all threads. */
  inline void
  AfterThreadedGetValueAndDerivative(MeasureType & value, DerivativeType & derivative) const override;

private:
  bool         m_UseNormalization;
  double       m_SelfHessianSmoothingSigma;
  double       m_SelfHessianNoiseRange;
  unsigned int m_NumberOfSamplesForSelfHessian;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedMeanSquaresImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkAdvancedMeanSquaresImageToImageMetric_h
