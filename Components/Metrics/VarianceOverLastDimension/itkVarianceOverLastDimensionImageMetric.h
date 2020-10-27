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

#ifndef __itkVarianceOverLastDimensionImageMetric_h
#define __itkVarianceOverLastDimensionImageMetric_h

#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkImageRandomCoordinateSampler.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkAdvancedImageToImageMetric.h"

namespace itk
{

/** \class VarianceOverLastDimensionImageMetric
 * \brief Compute the sum of variances over the slowest varying dimension in the moving image.
 *
 * This metric is based on the AdvancedImageToImageMetric.
 * It is templated over the type of the fixed and moving images to be compared.
 *
 * This metric computes the sum of variances over the slowest varying dimension in
 * the moving image. The spatial positions of the moving image are established
 * through a Transform. Pixel values are taken from the Moving image.
 *
 * This implementation is based on the AdvancedImageToImageMetric, which means that:
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
class VarianceOverLastDimensionImageMetric : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef VarianceOverLastDimensionImageMetric                  Self;
  typedef AdvancedImageToImageMetric<TFixedImage, TMovingImage> Superclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;

  typedef typename Superclass::FixedImageRegionType FixedImageRegionType;
  typedef typename FixedImageRegionType::SizeType   FixedImageSizeType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VarianceOverLastDimensionImageMetric, AdvancedImageToImageMetric);

  /** Set functions. */
  itkSetMacro(SampleLastDimensionRandomly, bool);
  itkSetMacro(NumSamplesLastDimension, unsigned int);
  itkSetMacro(NumAdditionalSamplesFixed, unsigned int);
  itkSetMacro(ReducedDimensionIndex, unsigned int);
  itkSetMacro(SubtractMean, bool);
  itkSetMacro(GridSize, FixedImageSizeType);
  itkSetMacro(TransformIsStackTransform, bool);

  /** Get functions. */
  itkGetConstMacro(SampleLastDimensionRandomly, bool);
  itkGetConstMacro(NumSamplesLastDimension, int);

  /** Typedefs from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType    CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType                 MovingImageType;
  typedef typename Superclass::MovingImagePixelType            MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer         MovingImageConstPointer;
  typedef typename Superclass::FixedImageType                  FixedImageType;
  typedef typename Superclass::FixedImageConstPointer          FixedImageConstPointer;
  typedef typename Superclass::TransformType                   TransformType;
  typedef typename Superclass::TransformPointer                TransformPointer;
  typedef typename Superclass::InputPointType                  InputPointType;
  typedef typename Superclass::OutputPointType                 OutputPointType;
  typedef typename Superclass::TransformParametersType         TransformParametersType;
  typedef typename Superclass::TransformJacobianType           TransformJacobianType;
  typedef typename Superclass::InterpolatorType                InterpolatorType;
  typedef typename Superclass::InterpolatorPointer             InterpolatorPointer;
  typedef typename Superclass::RealType                        RealType;
  typedef typename Superclass::GradientPixelType               GradientPixelType;
  typedef typename Superclass::GradientImageType               GradientImageType;
  typedef typename Superclass::GradientImagePointer            GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType         GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer      GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType              FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer           FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType             MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer          MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                     MeasureType;
  typedef typename Superclass::DerivativeType                  DerivativeType;
  typedef typename Superclass::ParametersType                  ParametersType;
  typedef typename Superclass::FixedImagePixelType             FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType           MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType                ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer             ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType        ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer     ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType           FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType          MovingImageLimiterType;
  typedef typename Superclass::FixedImageLimiterOutputType     FixedImageLimiterOutputType;
  typedef typename Superclass::MovingImageLimiterOutputType    MovingImageLimiterOutputType;
  typedef typename Superclass::MovingImageDerivativeScalesType MovingImageDerivativeScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Get the value for single valued optimizers. */
  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & derivative) const override;

  /** Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const override;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation.   */
  void
  Initialize(void) override;

protected:
  VarianceOverLastDimensionImageMetric();
  ~VarianceOverLastDimensionImageMetric() override {}
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  typedef typename Superclass::FixedImageIndexType      FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType     MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType      FixedImagePointType;
  typedef typename itk::ContinuousIndex<CoordinateRepresentationType, FixedImageDimension>
                                                                   FixedImageContinuousIndexType;
  typedef typename Superclass::MovingImagePointType                MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType      MovingImageContinuousIndexType;
  typedef typename Superclass::BSplineInterpolatorType             BSplineInterpolatorType;
  typedef typename Superclass::CentralDifferenceGradientFilterType CentralDifferenceGradientFilterType;
  typedef typename Superclass::MovingImageDerivativeType           MovingImageDerivativeType;
  typedef typename Superclass::NonZeroJacobianIndicesType          NonZeroJacobianIndicesType;

  /** Computes the innerproduct of transform Jacobian with moving image gradient.
   * The results are stored in imageJacobian, which is supposed
   * to have the right size (same length as Jacobian's number of columns). */
  void
  EvaluateTransformJacobianInnerProduct(const TransformJacobianType &     jacobian,
                                        const MovingImageDerivativeType & movingImageDerivative,
                                        DerivativeType &                  imageJacobian) const override;

private:
  VarianceOverLastDimensionImageMetric(const Self &); // purposely not implemented
  void
  operator=(const Self &); // purposely not implemented

  /** Sample n random numbers from 0..m and add them to the vector. */
  void
  SampleRandom(const int n, const int m, std::vector<int> & numbers) const;

  /** Variables to control random sampling in last dimension. */
  bool         m_SampleLastDimensionRandomly;
  unsigned int m_NumSamplesLastDimension;
  unsigned int m_NumAdditionalSamplesFixed;
  unsigned int m_ReducedDimensionIndex;

  /** Bool to determine if we want to subtract the mean derivate from the derivative elements. */
  bool m_SubtractMean;

  /** Initial variance in last dimension, used as normalization factor. */
  float m_InitialVariance;

  /** GridSize of B-spline transform. */
  FixedImageSizeType m_GridSize;

  /** Bool to indicate if the transform used is a stacktransform. Set by elx files. */
  bool m_TransformIsStackTransform;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVarianceOverLastDimensionImageMetric.hxx"
#endif

#endif // end #ifndef __itkVarianceOverLastDimensionImageMetric_h
