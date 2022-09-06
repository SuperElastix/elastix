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
#ifndef itkAdvancedNormalizedCorrelationImageToImageMetric_h
#define itkAdvancedNormalizedCorrelationImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"
#include <vector>

namespace itk
{
/** \class AdvancedNormalizedCorrelationImageToImageMetric
 * \brief Computes normalized correlation between two images, based on AdvancedImageToImageMetric...
 *
 * This metric computes the correlation between pixels in the fixed image
 * and pixels in the moving image. The spatial correspondence between
 * fixed and moving image is established through a Transform. Pixel values are
 * taken from the fixed image, their positions are mapped to the moving
 * image and result in general in non-grid position on it. Values at these
 * non-grid position of the moving image are interpolated using a user-selected
 * Interpolator. The correlation is normalized by the autocorrelations of both
 * the fixed and moving images.
 *
 * This implementation of the NormalizedCorrelation is based on the
 * AdvancedImageToImageMetric, which means that:
 * \li It uses the ImageSampler-framework
 * \li It makes use of the compact support of B-splines, in case of B-spline transforms.
 * \li Image derivatives are computed using either the B-spline interpolator's implementation
 * or by nearest neighbor interpolation of a precomputed central difference image.
 * \li A minimum number of samples that should map within the moving image (mask) can be specified.
 *
 * The normalized correlation NC is defined as:
 *
 * \f[
 * \mathrm{NC} = \frac{\sum_x f(x) * m(x+u(x,p))}{\sqrt{ \sum_x f(x)^2 * \sum_x m(x+u(x,p))^2}}
 *    = \frac{\mathtt{sfm}}{\sqrt{\mathtt{sff} * \mathtt{smm}}}
 * \f]
 *
 * where x a voxel in the fixed image f, m the moving image, u(x,p) the
 * deformation of x depending on the transform parameters p. sfm, sff and smm
 * is notation used in the source code. The derivative of NC to p equals:
 * \f[
 *   \frac{\partial \mathrm{NC}}{\partial p} = \frac{\partial \mathrm{NC}}{\partial m}
 *     \frac{\partial m}{\partial x} \frac{\partial x}{\partial p}
 *     = \frac{\partial \mathrm{NC}}{\partial m} * \mathtt{gradient} * \mathtt{jacobian},
 * \f]
 * where gradient is the derivative of the moving image m to x, and where Jacobian is the
 * derivative of the transformation to its parameters. gradient * Jacobian is called the differential.
 * This yields for the derivative:
 *
 * \f[
 *   \frac{\partial \mathrm{NC}}{\partial p}
 *     = \frac{\sum_x[ f(x) * \mathtt{differential} ] - ( \mathtt{sfm} / \mathtt{smm} )
 *     * \sum_x[ m(x+u(x,p)) * \mathtt{differential} ]}{\sqrt{\mathtt{sff} * \mathtt{smm}}}
 * \f]
 *
 * This class has an option to subtract the sample mean from the sample values
 * in the cross correlation formula. This typically results in narrower valleys
 * in the cost function NC. The default value is false. If SubtractMean is true,
 * the NC is defined as:
 *
 * \f[
 * \mathrm{NC} = \frac{\sum_x ( f(x) - \mathtt{Af} ) * ( m(x+u(x,p)) - \mathtt{Am})}
 *     {\sqrt{\sum_x (f(x) - \mathtt{Af})^2 * \sum_x (m(x+u(x,p)) - \mathtt{Am})^2}}
 *    = \frac{\mathtt{sfm} - \mathtt{sf} * \mathtt{sm} / N}
 *   {\sqrt{(\mathtt{sff} - \mathtt{sf} * \mathtt{sf} / N) * (\mathtt{smm} - \mathtt{sm} *\mathtt{sm} / N)}},
 * \f]
 *
 * where Af and Am are the average of f and m, respectively.
 *
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT AdvancedNormalizedCorrelationImageToImageMetric
  : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedNormalizedCorrelationImageToImageMetric);

  /** Standard class typedefs. */
  using Self = AdvancedNormalizedCorrelationImageToImageMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedNormalizedCorrelationImageToImageMetric, AdvancedImageToImageMetric);

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
  using typename Superclass::ThreaderType;
  using typename Superclass::ThreadInfoType;

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
  GetValueAndDerivativeSingleThreaded(const TransformParametersType & parameters,
                                      MeasureType &                   value,
                                      DerivativeType &                derivative) const;

  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   value,
                        DerivativeType &                derivative) const override;

  /** Set/Get SubtractMean boolean. If true, the sample mean is subtracted
   * from the sample values in the cross-correlation formula and
   * typically results in narrower valleys in the cost function.
   * Default value is false.
   */
  itkSetMacro(SubtractMean, bool);
  itkGetConstReferenceMacro(SubtractMean, bool);
  itkBooleanMacro(SubtractMean);

protected:
  AdvancedNormalizedCorrelationImageToImageMetric();
  ~AdvancedNormalizedCorrelationImageToImageMetric() override = default;

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

  /** Compute a pixel's contribution to the derivative terms;
   * Called by GetValueAndDerivative().
   */
  void
  UpdateDerivativeTerms(const RealType &                   fixedImageValue,
                        const RealType &                   movingImageValue,
                        const DerivativeType &             imageJacobian,
                        const NonZeroJacobianIndicesType & nzji,
                        DerivativeType &                   derivativeF,
                        DerivativeType &                   derivativeM,
                        DerivativeType &                   differential) const;

  /** Initialize some multi-threading related parameters.
   * Overrides function in AdvancedImageToImageMetric, because
   * here we use other parameters.
   */
  void
  InitializeThreadingParameters() const override;

  /** Get value and derivatives for each thread. */
  inline void
  ThreadedGetValueAndDerivative(ThreadIdType threadID) override;

  /** Gather the values and derivatives from all threads */
  inline void
  AfterThreadedGetValueAndDerivative(MeasureType & value, DerivativeType & derivative) const override;

  /** AccumulateDerivatives threader callback function */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  AccumulateDerivativesThreaderCallback(void * arg);

private:
  mutable bool m_SubtractMean;

  using AccumulateType = typename NumericTraits<MeasureType>::AccumulateType;

  /** Helper structs that multi-threads the computation of
   * the metric derivative using ITK threads.
   */
  struct MultiThreaderAccumulateDerivativeType
  {
    AdvancedNormalizedCorrelationImageToImageMetric * st_Metric;

    AccumulateType        st_sf_N;
    AccumulateType        st_sm_N;
    AccumulateType        st_sfm_smm;
    RealType              st_InvertedDenominator;
    DerivativeValueType * st_DerivativePointer;
  };

  struct CorrelationGetValueAndDerivativePerThreadStruct
  {
    SizeValueType  st_NumberOfPixelsCounted;
    AccumulateType st_Sff;
    AccumulateType st_Smm;
    AccumulateType st_Sfm;
    AccumulateType st_Sf;
    AccumulateType st_Sm;
    DerivativeType st_DerivativeF;
    DerivativeType st_DerivativeM;
    DerivativeType st_Differential;
  };
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT,
               CorrelationGetValueAndDerivativePerThreadStruct,
               PaddedCorrelationGetValueAndDerivativePerThreadStruct);
  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT,
                    PaddedCorrelationGetValueAndDerivativePerThreadStruct,
                    AlignedCorrelationGetValueAndDerivativePerThreadStruct);
  mutable std::vector<AlignedCorrelationGetValueAndDerivativePerThreadStruct>
    m_CorrelationGetValueAndDerivativePerThreadVariables;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedNormalizedCorrelationImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkAdvancedNormalizedCorrelationImageToImageMetric_h
