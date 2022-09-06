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
#ifndef itkSumSquaredTissueVolumeDifferenceImageToImageMetric_h
#define itkSumSquaredTissueVolumeDifferenceImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"

namespace itk
{

/** \class SumSquaredTissueVolumeDifferenceImageToImageMetric
 * \brief Compute sum of square tissue volume difference between two images
 *
 * This Class is templated over the type of the fixed and moving
 * images to be compared.
 *
 * This metrics implements a mass-preserving image similarity term, as described
 * by both Yin et al. and Gorbunova et al. Essentially, the similarity term is
 * equivalent to the sum of squared differences between pixels in the moving and
 * fixed images, except the intensity of the moving image is first scaled by the
 * determinant of the spatial Jacobian to correct for density effects on image
 * intensity. Gorbunova et al. provide the analytical gradient of the cost
 * function with respect to the transform parameters, which is implemented here.
 *
 * This implementation is based on the AdvancedImageToImageMetric, which means
 * that:
 * \li It uses the ImageSampler-framework
 * \li It makes use of the compact support of B-splines, in case of B-spline transforms.
 * \li Image derivatives are computed using either the B-spline interpolator's implementation
 * or by nearest neighbor interpolation of a precomputed central difference image.
 * \li A minimum number of samples that should map within the moving image (mask) can be specified.
 *
 * References:\n
 * [1] Yin, Y., Hoffman, E. A., & Lin, C. L. (2009).
 *     Mass preserving nonrigid registration of CT lung images using cubic B-spline.
 *     Medical physics, 36(9), 4213-4222.
 * [2] Gorbunova, V., Sporring, J., Lo, P., Loeve, M., Tiddens, H. A., Nielsen, M., Dirksen, A., de Bruijne, M. (2012).
 *     Mass preserving image registration for lung CT.
 *     Medical image analysis, 16(4), 786-795.
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT SumSquaredTissueVolumeDifferenceImageToImageMetric
  : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SumSquaredTissueVolumeDifferenceImageToImageMetric);

  /** Standard class typedefs. */
  using Self = SumSquaredTissueVolumeDifferenceImageToImageMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SumSquaredTissueVolumeDifferenceImageToImageMetric, AdvancedImageToImageMetric);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
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

  /** Typedefs from the AdvancedTransform. */
  using TransformType = typename Superclass::AdvancedTransformType;
  using SpatialJacobianType = typename TransformType::SpatialJacobianType;
  using JacobianOfSpatialJacobianType = typename TransformType::JacobianOfSpatialJacobianType;
  using SpatialHessianType = typename TransformType::SpatialHessianType;
  using JacobianOfSpatialHessianType = typename TransformType::JacobianOfSpatialHessianType;
  using InternalMatrixType = typename TransformType::InternalMatrixType;

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

  /** Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const override;

  /** Get value and derivatives single-threaded */
  void
  GetValueAndDerivativeSingleThreaded(const TransformParametersType & parameters,
                                      MeasureType &                   measure,
                                      DerivativeType &                derivative) const;

  /** Set/get the air intensity value */
  itkSetMacro(AirValue, RealType);
  itkGetConstMacro(AirValue, RealType);

  /** Set/get the tissue intensity value */
  itkSetMacro(TissueValue, RealType);
  itkGetConstMacro(TissueValue, RealType);

protected:
  SumSquaredTissueVolumeDifferenceImageToImageMetric();
  ~SumSquaredTissueVolumeDifferenceImageToImageMetric() override = default;

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

  /** Computes the inner product of transform Jacobian with moving image gradient.
   * The results are stored in imageJacobian, which is supposed
   * to have the right size (same length as Jacobian's number of columns). */
  void
  EvaluateTransformJacobianInnerProduct(const TransformJacobianType &     jacobian,
                                        const MovingImageDerivativeType & movingImageDerivative,
                                        DerivativeType &                  imageJacobian) const override;

  /** Compute a pixel's contribution to the measure and derivatives;
   * Called by GetValueAndDerivative(). */
  void
  UpdateValueAndDerivativeTerms(const RealType                     fixedImageValue,
                                const RealType                     movingImageValue,
                                const DerivativeType &             imageJacobian,
                                const NonZeroJacobianIndicesType & nzji,
                                const RealType                     spatialJacobianDeterminant,
                                const DerivativeType &             jacobianOfSpatialJacobianDeterminant,
                                MeasureType &                      measure,
                                DerivativeType &                   deriv) const;

  /** Compute the inverse SpatialJacobian to support calculation of the metric gradient.
   * Note that this function does not calculate the true inverse, but instead calculates
   * the inverse SpatialJacobian multiplied by the determinant of the SpatialJacobian, to
   * avoid redundant use of the determinant.
   * This function returns false if the SpatialJacobianDeterminant is zero.
   */
  bool
  EvaluateInverseSpatialJacobian(const SpatialJacobianType & spatialJacobian,
                                 const RealType              spatialJacobianDeterminant,
                                 SpatialJacobianType &       inverseSpatialJacobian) const;

  /** Compute the dot product of the inverse SpatialJacobian with the
   * Jacobian of SpatialJacobian.  The results are stored in
   * jacobianOfSpatialJacobianDeterminant, which has a length equal to
   * the number of transform parameters times the length of the spatialJacobian.
   */
  void
  EvaluateJacobianOfSpatialJacobianDeterminantInnerProduct(
    const JacobianOfSpatialJacobianType & jacobianOfSpatialJacobian,
    const SpatialJacobianType &           inverseSpatialJacobian,
    DerivativeType &                      jacobianOfSpatialJacobianDeterminant) const;

  /** Get value for each thread. */
  inline void
  ThreadedGetValue(ThreadIdType threadID) override;

  /** Gather the values from all threads. */
  inline void
  AfterThreadedGetValue(MeasureType & value) const override;

  /** Get value and derivatives for each thread. */
  inline void
  ThreadedGetValueAndDerivative(ThreadIdType threadId) override;

  /** Gather the values and derivatives from all threads */
  inline void
  AfterThreadedGetValueAndDerivative(MeasureType & measure, DerivativeType & derivative) const override;

private:
  /** Intensity value to use for air.  Default is -1000 */
  RealType m_AirValue;

  /** Intensity value to use for tissue.  Default is 55 */
  RealType m_TissueValue;

}; // end class SumSquaredTissueVolumeDifferenceImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkSumSquaredTissueVolumeDifferenceImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkSumSquaredTissueVolumeDifferenceImageToImageMetric_h
