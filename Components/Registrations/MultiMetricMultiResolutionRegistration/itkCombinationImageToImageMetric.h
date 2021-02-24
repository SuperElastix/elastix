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
#ifndef itkCombinationImageToImageMetric_h
#define itkCombinationImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"
#include "itkSingleValuedPointSetToPointSetMetric.h"

namespace itk
{

/** \class CombinationImageToImageMetric
 * \brief Combines multiple metrics.
 *
 * This metric is meant to be used in the
 * MultiMetricMultiResolutionImageRegistrationMethod.
 *
 * NB: while it may seem not logical that the SetInterpolator(arg)
 * sets the interpolator in all submetrics whereas the
 * GetInterpolator(void) returns GetInterpolator(0) it is logical.
 * If you set the interpolator the same in all metrics, you will
 * receive the correct interpolator with GetInterpolator(0).
 * If you set the interpolator differently in all metrics, the most
 * logical action is to return GetInterpolator(0) when GetInterpolator()
 * is invoked.
 *
 * Note: If you use Set{Transform,Interpolator etc}(0) or
 * Set{Transform,Interpolator}(), the member variables of the
 * superclass m_{Transform,Interpolator} are set as well.
 * So, it is not strictly necessary to reimplement the functions
 * Get{Transform,Interpolator}(), which return Get{Transform,Interpolator}(0),
 * which would result in the same as returning m_{Transform,Interpolator}
 * anyway. However, if Metric[0] is changed externally
 * the m_{Transform, Interpolator} are not up to date anymore. That's
 * why we chose to reimplement the Get{Transform,Interpolator}()
 * methods.
 *
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT CombinationImageToImageMetric : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef CombinationImageToImageMetric                         Self;
  typedef AdvancedImageToImageMetric<TFixedImage, TMovingImage> Superclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(CombinationImageToImageMetric, AdvancedImageToImageMetric);

  /** Define the New() method */
  itkNewMacro(Self);

  /** Constants for the image dimensions */
  itkStaticConstMacro(MovingImageDimension, unsigned int, TMovingImage::ImageDimension);
  itkStaticConstMacro(FixedImageDimension, unsigned int, TFixedImage::ImageDimension);

  /** Typedefs from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename Superclass::MovingImagePixelType         MovingImagePixelType;
  // typedef typename Superclass::MovingImagePointer         MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer MovingImageConstPointer;
  typedef typename Superclass::FixedImageType          FixedImageType;
  // typedef typename Superclass::FixedImagePointer          FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::AdvancedTransformType      TransformType;
  typedef typename TransformType::Pointer                 TransformPointer;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::InterpolatorType           InterpolatorType;
  typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::DerivativeValueType        DerivativeValueType;
  typedef typename Superclass::ParametersType             ParametersType;

  /** Some typedefs for computing the SelfHessian */
  typedef typename Superclass::HessianValueType HessianValueType;
  typedef typename Superclass::HessianType      HessianType;

  /**
  typedef typename Superclass::ImageSamplerType             ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer          ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType        FixedImageLimiterType;
  typedef typename Superclass::FixedImageLimiterOutputType  FixedImageLimiterOutputType;
  typedef typename Superclass::MovingImageLimiterType       MovingImageLimiterType;
  typedef typename Superclass::MovingImageLimiterOutputType MovingImageLimiterOutputType;
  typedef typename Superclass::ScalarType                   ScalarType;
  typedef typename Superclass::AdvancedTransformType        AdvancedTransformType;
  */

  /** Typedefs for the metrics. */
  typedef Superclass                                     ImageMetricType;
  typedef typename ImageMetricType::Pointer              ImageMetricPointer;
  typedef SingleValuedCostFunction                       SingleValuedCostFunctionType;
  typedef typename SingleValuedCostFunctionType::Pointer SingleValuedCostFunctionPointer;

  typedef typename FixedImageType::PixelType                               FixedImagePixelType;
  typedef typename MovingImageType::RegionType                             MovingImageRegionType;
  typedef FixedArray<double, itkGetStaticConstMacro(MovingImageDimension)> MovingImageDerivativeScalesType;

  /** Typedef for the PointSetMetric. */
  typedef PointSet<CoordinateRepresentationType,
                   TFixedImage::ImageDimension,
                   DefaultStaticMeshTraits<CoordinateRepresentationType,
                                           TFixedImage::ImageDimension,
                                           TFixedImage::ImageDimension,
                                           CoordinateRepresentationType,
                                           CoordinateRepresentationType,
                                           CoordinateRepresentationType>>
    FixedPointSetType;
  typedef PointSet<CoordinateRepresentationType,
                   TMovingImage::ImageDimension,
                   DefaultStaticMeshTraits<CoordinateRepresentationType,
                                           TMovingImage::ImageDimension,
                                           TMovingImage::ImageDimension,
                                           CoordinateRepresentationType,
                                           CoordinateRepresentationType,
                                           CoordinateRepresentationType>>
                                                                                      MovingPointSetType;
  typedef SingleValuedPointSetToPointSetMetric<FixedPointSetType, MovingPointSetType> PointSetMetricType;

  /** Typedefs for multi-threading. */
  typedef typename Superclass::ThreaderType   ThreaderType;
  typedef typename Superclass::ThreadInfoType ThreadInfoType;

  /**
   * Get and set the metrics and their weights.
   **/

  /** Set the number of metrics to combine. */
  void
  SetNumberOfMetrics(unsigned int count);

  /** Get the number of metrics to combine. */
  itkGetConstMacro(NumberOfMetrics, unsigned int);

  /** Set metric i. It may be a SingleValuedCostFunction, instead of
   * a ImageToImageMetric, but the first one should be an
   * ImageToImageMetric in all cases.
   */
  void
  SetMetric(SingleValuedCostFunctionType * metric, unsigned int pos);

  /** Get metric i. */
  SingleValuedCostFunctionType *
  GetMetric(unsigned int count) const;

  /** Set the weight for metric i. */
  void
  SetMetricWeight(double weight, unsigned int pos);

  /** Get the weight for metric i. */
  double
  GetMetricWeight(unsigned int pos) const;

  /** Set the relative weight for metric i. */
  void
  SetMetricRelativeWeight(double weight, unsigned int pos);

  /** Get the relative weight for metric i. */
  double
  GetMetricRelativeWeight(unsigned int pos) const;

  /** Set and Get the UseRelativeWeights variable. */
  itkSetMacro(UseRelativeWeights, bool);
  itkGetMacro(UseRelativeWeights, bool);

  /** Select which metrics are used.
   * This is useful in case you want to compute a certain measure, but not
   * actually use it during the registration.
   * By default all metrics that are set, are also used.
   */
  void
  SetUseMetric(const bool use, const unsigned int pos);

  /** Use all metrics. */
  void
  SetUseAllMetrics(void);

  /** Get if this metric is used. */
  bool
  GetUseMetric(const unsigned int pos) const;

  /** Get the last computed value for metric i. */
  MeasureType
  GetMetricValue(unsigned int pos) const;

  /** Get the last computed derivative for metric i. */
  const DerivativeType &
  GetMetricDerivative(unsigned int pos) const;

  /** Get the last computed derivative magnitude for metric i. */
  double
  GetMetricDerivativeMagnitude(unsigned int pos) const;

  /** Get the last computed computation time for metric i. */
  double
  GetMetricComputationTime(unsigned int pos) const;

  /**
   * Set/Get functions for the metric components
   */

  /** Pass the transform to all sub metrics.  */
  void
  SetTransform(TransformType * _arg) override;

  /** Pass a transform to a specific metric.
   * Only use this if you really know what you are doing.
   *
   * In fact, in general it makes no sense to specify a different
   * transform for every metric, because in the GetValue/GetDerivative
   * methods, the same set of parameters will be used in all cases.
   * Also, SetTransformParameters and GetNumberOfParameters may give
   * unpredictable results if you use this method. They only refer to
   * the first transform.
   */
  virtual void
  SetTransform(TransformType * _arg, unsigned int pos);

  /** Returns the transform set in a specific metric. If the submetric is a
   * singlevalued costfunction a zero pointer will be returned.
   */
  virtual const TransformType *
  GetTransform(unsigned int pos) const;

  /** Return Transform 0 */
  const TransformType *
  GetTransform(void) const override
  {
    return this->GetTransform(0);
  }


  /** Pass the interpolator to all sub metrics. */
  void
  SetInterpolator(InterpolatorType * _arg) override;

  /** Pass an interpolator to a specific metric */
  virtual void
  SetInterpolator(InterpolatorType * _arg, unsigned int pos);

  /** Returns the interpolator set in a specific metric. If the submetric is
   * a singlevalued costfunction a zero pointer will be returned.
   */
  virtual const InterpolatorType *
  GetInterpolator(unsigned int pos) const;

  /** Return Interpolator 0 */
  const InterpolatorType *
  GetInterpolator(void) const override
  {
    return this->GetInterpolator(0);
  }


  /** Pass the fixed image to all sub metrics. */
  void
  SetFixedImage(const FixedImageType * _arg) override;

  /** Pass a fixed image to a specific metric */
  virtual void
  SetFixedImage(const FixedImageType * _arg, unsigned int pos);

  /** Returns the fixedImage set in a specific metric. If the
   * submetric is a singlevalued costfunction a zero pointer will
   * be returned */
  virtual const FixedImageType *
  GetFixedImage(unsigned int pos) const;

  /** Return FixedImage 0 */
  const FixedImageType *
  GetFixedImage(void) const override
  {
    return this->GetFixedImage(0);
  }


  /** Pass the fixed image mask to all sub metrics. */
  void
  SetFixedImageMask(FixedImageMaskType * _arg) override;

  /** Pass a fixed image mask to a specific metric */
  virtual void
  SetFixedImageMask(FixedImageMaskType * _arg, unsigned int pos);

  /** Returns the fixedImageMask set in a specific metric. If the
   * submetric is a singlevalued costfunction a zero pointer will
   * be returned */
  virtual const FixedImageMaskType *
  GetFixedImageMask(unsigned int pos) const;

  /** Return FixedImageMask 0 */
  const FixedImageMaskType *
  GetFixedImageMask(void) const override
  {
    return this->GetFixedImageMask(0);
  }


  /** Pass the fixed image region to all sub metrics. */
  void
  SetFixedImageRegion(const FixedImageRegionType _arg) override;

  /** Pass a fixed image region to a specific metric. */
  virtual void
  SetFixedImageRegion(const FixedImageRegionType _arg, unsigned int pos);

  /** Returns the fixedImageRegion set in a specific metric. If the
   * submetric is a singlevalued costfunction a region with size zero will
   * be returned */
  virtual const FixedImageRegionType &
  GetFixedImageRegion(unsigned int pos) const;

  /** Return FixedImageRegion 0 */
  const FixedImageRegionType &
  GetFixedImageRegion(void) const override
  {
    return this->GetFixedImageRegion(0);
  }


  /** Pass the moving image to all sub metrics. */
  void
  SetMovingImage(const MovingImageType * _arg) override;

  /** Pass a moving image to a specific metric */
  virtual void
  SetMovingImage(const MovingImageType * _arg, unsigned int pos);

  /** Returns the movingImage set in a specific metric. If the
   * submetric is a singlevalued costfunction a zero pointer will
   * be returned */
  virtual const MovingImageType *
  GetMovingImage(unsigned int pos) const;

  /** Return MovingImage 0 */
  const MovingImageType *
  GetMovingImage(void) const override
  {
    return this->GetMovingImage(0);
  }


  /** Pass the moving image mask to all sub metrics. */
  void
  SetMovingImageMask(MovingImageMaskType * _arg) override;

  /** Pass a moving image mask to a specific metric */
  virtual void
  SetMovingImageMask(MovingImageMaskType * _arg, unsigned int pos);

  /** Returns the movingImageMask set in a specific metric. If the
   * submetric is a singlevalued costfunction a zero pointer will
   * be returned */
  virtual const MovingImageMaskType *
  GetMovingImageMask(unsigned int pos) const;

  /** Return MovingImageMask 0 */
  const MovingImageMaskType *
  GetMovingImageMask(void) const override
  {
    return this->GetMovingImageMask(0);
  }


  /** Get the number of pixels considered in the computation. Return the sum
   * of pixels counted by all metrics.
   */
  const SizeValueType &
  GetNumberOfPixelsCounted(void) const override;

  /** Pass initialization to all sub metrics. */
  void
  Initialize(void) override;

  /**
   * Combine all sub metrics by adding them.
   */

  /** The GetValue()-method. */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /** The GetDerivative()-method. */
  void
  GetDerivative(const ParametersType & parameters, DerivativeType & derivative) const override;

  /** The GetValueAndDerivative()-method. */
  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          value,
                        DerivativeType &       derivative) const override;

  /** Experimental feature: compute SelfHessian. */
  void
  GetSelfHessian(const TransformParametersType & parameters, HessianType & H) const override;

  /** Method to return the latest modified time of this object or any of its
   * cached ivars.
   */
  ModifiedTimeType
  GetMTime() const override;

protected:
  CombinationImageToImageMetric();
  ~CombinationImageToImageMetric() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Store the metrics and the corresponding weights. */
  unsigned int                                 m_NumberOfMetrics;
  std::vector<SingleValuedCostFunctionPointer> m_Metrics;
  std::vector<double>                          m_MetricWeights;
  std::vector<double>                          m_MetricRelativeWeights;
  bool                                         m_UseRelativeWeights;
  std::vector<bool>                            m_UseMetric;
  mutable std::vector<MeasureType>             m_MetricValues;
  mutable std::vector<DerivativeType>          m_MetricDerivatives;
  mutable std::vector<double>                  m_MetricDerivativesMagnitude;
  mutable std::vector<double>                  m_MetricComputationTime;

  /** Dummy image region and derivatives. */
  FixedImageRegionType m_NullFixedImageRegion;
  DerivativeType       m_NullDerivative;

private:
  CombinationImageToImageMetric(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  /** Initialize some multi-threading related parameters.
   * Overrides function in AdvancedImageToImageMetric, because
   * here we use other parameters.
   */
  void
  InitializeThreadingParameters(void) const override;

  /** Compute the current metric weight, given the user selected
   * strategy and derivative magnitude.
   */
  double
  GetFinalMetricWeight(unsigned int pos) const;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCombinationImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkCombinationImageToImageMetric_h
