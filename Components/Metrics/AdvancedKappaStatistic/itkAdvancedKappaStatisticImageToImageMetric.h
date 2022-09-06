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
#ifndef itkAdvancedKappaStatisticImageToImageMetric_h
#define itkAdvancedKappaStatisticImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"
#include <vector>

namespace itk
{

/** \class AdvancedKappaStatisticImageToImageMetric
 * \brief Computes similarity between two objects to be registered
 *
 * This class is templated over the type of the fixed and moving
 * images to be compared.  The metric here is designed for matching
 * pixels in two images with the same exact value.  Only one value can
 * be considered (the default is 255) and can be specified with the
 * SetForegroundValue method.  In the computation of the metric, only
 * foreground pixels are considered.  The metric value is given
 * by 2*|A&B|/(|A|+|B|), where A is the foreground region in the moving
 * image, B is the foreground region in the fixed image, & is intersection,
 * and |.| indicates the area of the enclosed set.  The metric is
 * described in "Morphometric Analysis of White Matter Lesions in MR
 * Images: Method and Validation", A. P. Zijdenbos, B. M. Dawant, R. A.
 * Margolin, A. C. Palmer.
 *
 * This metric is especially useful when considering the similarity between
 * binary images.  Given the nature of binary images, a nearest neighbor
 * interpolator is the preferred interpolator.
 *
 * Metric values range from 0.0 (no foreground alignment) to 1.0
 * (perfect foreground alignment).  When dealing with optimizers that can
 * only minimize a metric, use the ComplementOn() method.
 *
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT AdvancedKappaStatisticImageToImageMetric
  : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedKappaStatisticImageToImageMetric);

  /** Standard class typedefs. */
  using Self = AdvancedKappaStatisticImageToImageMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedKappaStatisticImageToImageMetric, AdvancedImageToImageMetric);

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
  virtual void
  GetValueAndDerivativeSingleThreaded(const TransformParametersType & parameters,
                                      MeasureType &                   Value,
                                      DerivativeType &                Derivative) const;

  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const override;

  /** Computes the moving gradient image dM/dx. */
  void
  ComputeGradient() override;

  /** This method allows the user to set the foreground value. The default value is 1.0. */
  itkSetMacro(ForegroundValue, RealType);
  itkGetConstReferenceMacro(ForegroundValue, RealType);

  /** Select which kind of kappa to compute:
   * 1) compare with a foreground value
   * 2) compare if larger than zero
   */
  itkSetMacro(UseForegroundValue, bool);

  /** Set/Get whether this metric returns 2*|A&B|/(|A|+|B|)
   * (ComplementOff, the default) or 1.0 - 2*|A&B|/(|A|+|B|)
   * (ComplementOn). When using an optimizer that minimizes
   * metric values use ComplementOn().
   */
  itkSetMacro(Complement, bool);
  itkGetConstReferenceMacro(Complement, bool);
  itkBooleanMacro(Complement);

  /** Set the precision. */
  itkSetMacro(Epsilon, RealType);
  itkGetConstReferenceMacro(Epsilon, RealType);

protected:
  AdvancedKappaStatisticImageToImageMetric();
  ~AdvancedKappaStatisticImageToImageMetric() override = default;

  /** PrintSelf. */
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

  /** Compute a pixel's contribution to the measure and derivatives;
   * Called by GetValueAndDerivative().
   */
  void
  UpdateValueAndDerivativeTerms(const RealType &                   fixedImageValue,
                                const RealType &                   movingImageValue,
                                std::size_t &                      fixedForegroundArea,
                                std::size_t &                      movingForegroundArea,
                                std::size_t &                      intersection,
                                const DerivativeType &             imageJacobian,
                                const NonZeroJacobianIndicesType & nzji,
                                DerivativeType &                   sum1,
                                DerivativeType &                   sum2) const;

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
  bool     m_UseForegroundValue;
  RealType m_ForegroundValue;
  RealType m_Epsilon;
  bool     m_Complement;

  /** Threading related parameters. */

  /** Helper structs that multi-threads the computation of
   * the metric derivative using ITK threads.
   */
  struct MultiThreaderAccumulateDerivativeType
  {
    AdvancedKappaStatisticImageToImageMetric * st_Metric;

    MeasureType           st_Coefficient1;
    MeasureType           st_Coefficient2;
    DerivativeValueType * st_DerivativePointer;
  };

  struct KappaGetValueAndDerivativePerThreadStruct
  {
    SizeValueType  st_NumberOfPixelsCounted;
    SizeValueType  st_AreaSum;
    SizeValueType  st_AreaIntersection;
    DerivativeType st_DerivativeSum1;
    DerivativeType st_DerivativeSum2;
  };
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT,
               KappaGetValueAndDerivativePerThreadStruct,
               PaddedKappaGetValueAndDerivativePerThreadStruct);
  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT,
                    PaddedKappaGetValueAndDerivativePerThreadStruct,
                    AlignedKappaGetValueAndDerivativePerThreadStruct);
  mutable std::vector<AlignedKappaGetValueAndDerivativePerThreadStruct> m_KappaGetValueAndDerivativePerThreadVariables;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedKappaStatisticImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkAdvancedKappaStatisticImageToImageMetric_h
