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
#ifndef elxMetricBase_h
#define elxMetricBase_h

/** Needed for the macros. */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkAdvancedImageToImageMetric.h"
#include "itkImageGridSampler.h"
#include "itkPointSet.h"

namespace elastix
{

/**
 * \class MetricBase
 * \brief This class is the elastix base class for all  Metrics.
 *
 * This class contains the common functionality for all Metrics.
 *
 * The parameters used in this class are:
 * \parameter ShowExactMetricValue: Flag that can set to "true" or "false".
 *    If "true" the metric computes the exact metric value (computed on all
 *    voxels rather than on the set of spatial samples) and shows it each
 *    iteration. Must be given for each resolution. \n
 *    example: <tt>(ShowExactMetricValue "true" "true" "false")</tt> \n
 *    Default is "false" for all resolutions.
 * \parameter ExactMetricSampleGridSpacing: Set an integer downsampling rate for
 *    computing the "exact" metric. Only meaningful if set in combination with the
 *    ShowExactMetricValue set to "true". In some cases, it might be an overkill
 *    to really compute the exact metric with the ShowExactMetricValue.
 *    The metric computed on a downsampled image might already be accurate
 *    enough to draw conclusions about the rate of convergence for example.
 *    The downsampling rate must be given for each resolution, for each dimension.\n
 *    example: <tt>(ExactMetricSampleGridSpacing 1 1 2 2 )</tt> \n
 *    This example for a 2D registration of 2 resolutions sets the downsampling rate
 *    to 1 in the first resolution (so: use really all pixels), and to 2 in the
 *    second resolution. Default: 1 in each resolution and each dimension.
 * \parameter CheckNumberOfSamples: Whether the metric checks if at least
 *    a certain fraction (default 1/4) of the samples map inside the moving
 *    image. Can be given for each resolution or for all resolutions at once. \n
 *    example: <tt>(CheckNumberOfSamples "false" "true" "false")</tt> \n
 *    The default is true. In general it is wise to set this to true,
 *    since it detects if the registration is going really bad.
 * \parameter RequiredRatioOfValidSamples: Defines the fraction needed in
 *    CheckNumberOfSamples. \n
 *    example: <tt>(RequiredRatioOfValidSamples 0.1)</tt> \n
 *    The default is 0.25.
 *
 * \ingroup Metrics
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MetricBase : public BaseComponentSE<TElastix>
{
public:
  /** Standard ITK stuff. */
  typedef MetricBase                Self;
  typedef BaseComponentSE<TElastix> Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MetricBase, BaseComponentSE);

  /** Typedef's inherited from Elastix. */
  typedef typename Superclass::ElastixType          ElastixType;
  typedef typename Superclass::ElastixPointer       ElastixPointer;
  typedef typename Superclass::ConfigurationType    ConfigurationType;
  typedef typename Superclass::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass::RegistrationType     RegistrationType;
  typedef typename Superclass::RegistrationPointer  RegistrationPointer;

  /** Other typedef's. */
  typedef typename ElastixType::FixedImageType  FixedImageType;
  typedef typename FixedImageType::PointType    FixedPointType;
  typedef typename FixedPointType::ValueType    FixedPointValueType;
  typedef typename ElastixType::MovingImageType MovingImageType;
  typedef typename MovingImageType::PointType   MovingPointType;
  typedef typename MovingPointType::ValueType   MovingPointValueType;

  /** ITKBaseType. */
  typedef itk::SingleValuedCostFunction                                    ITKBaseType;
  typedef itk::AdvancedImageToImageMetric<FixedImageType, MovingImageType> AdvancedMetricType;
  typedef typename AdvancedMetricType::MovingImageDerivativeScalesType     MovingImageDerivativeScalesType;

  /** Get the dimension of the fixed image. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  /** Get the dimension of the moving image. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedefs for point sets. */
  typedef typename ITKBaseType::ParametersValueType CoordinateRepresentationType;
  typedef itk::PointSet<CoordinateRepresentationType,
                        FixedImageDimension,
                        itk::DefaultStaticMeshTraits<CoordinateRepresentationType,
                                                     FixedImageDimension,
                                                     FixedImageDimension,
                                                     CoordinateRepresentationType,
                                                     CoordinateRepresentationType,
                                                     CoordinateRepresentationType>>
    FixedPointSetType;
  typedef itk::PointSet<CoordinateRepresentationType,
                        MovingImageDimension,
                        itk::DefaultStaticMeshTraits<CoordinateRepresentationType,
                                                     MovingImageDimension,
                                                     MovingImageDimension,
                                                     CoordinateRepresentationType,
                                                     CoordinateRepresentationType,
                                                     CoordinateRepresentationType>>
    MovingPointSetType;

  /** Typedefs for sampler support. */
  typedef typename AdvancedMetricType::ImageSamplerType ImageSamplerBaseType;

  /** Return type of GetValue */
  typedef typename ITKBaseType::MeasureType MeasureType;

  /** Retrieves this object as ITKBaseType. */
  ITKBaseType *
  GetAsITKBaseType(void)
  {
    return &(this->GetSelf());
  }


  /** Retrieves this object as ITKBaseType, to use in const functions. */
  const ITKBaseType *
  GetAsITKBaseType(void) const
  {
    return &(this->GetSelf());
  }


  /** Execute stuff before each resolution:
   * \li Check if the exact metric value should be computed
   * (to monitor the progress of the registration).
   */
  void
  BeforeEachResolutionBase(void) override;

  /** Execute stuff after each iteration:
   * \li Optionally compute the exact metric value and plot it to screen.
   */
  void
  AfterEachIterationBase(void) override;

  /** Force the metric to base its computation on a new subset of image samples.
   * Not every metric may have implemented this.
   */
  virtual void
  SelectNewSamples(void);

  /** Returns whether the metric uses a sampler. When the metric is not of
   * AdvancedMetricType, the function returns false immediately.
   */
  virtual bool
  GetAdvancedMetricUseImageSampler(void) const;

  /** Method to set the image sampler. The image sampler is only used when
   * the metric is of type AdvancedMetricType, and has UseImageSampler set
   * to true. In other cases, the function does nothing.
   */
  virtual void
  SetAdvancedMetricImageSampler(ImageSamplerBaseType * sampler);

  /** Methods to get the image sampler. The image sampler is only used when
   * the metric is of type AdvancedMetricType, and has UseImageSampler set
   * to true. In other cases, the function returns 0.
   */
  virtual ImageSamplerBaseType *
  GetAdvancedMetricImageSampler(void) const;

  /** Get if the exact metric value is computed */
  virtual bool
  GetShowExactMetricValue(void) const
  {
    return this->m_ShowExactMetricValue;
  }

  /** Get the last computed exact metric value */
  virtual MeasureType
  GetCurrentExactMetricValue(void) const
  {
    return this->m_CurrentExactMetricValue;
  }

protected:
  /** The parameters type. */
  typedef typename ITKBaseType::ParametersType ParametersType;

  /** The full sampler used by the GetExactValue method. */
  typedef itk::ImageGridSampler<FixedImageType>                       ExactMetricImageSamplerType;
  typedef typename ExactMetricImageSamplerType::Pointer               ExactMetricImageSamplerPointer;
  typedef typename ExactMetricImageSamplerType::SampleGridSpacingType ExactMetricSampleGridSpacingType;

  /** The constructor. */
  MetricBase();
  /** The destructor. */
  ~MetricBase() override = default;

  /**  Get the exact value. Mutual information computed over all points.
   * It is meant in situations when you optimize using just a subset of pixels,
   * but are interested in the exact value of the metric.
   *
   * This method only works when the itkYourMetric inherits from
   * the AdvancedMetricType.
   * In other cases it returns 0. You may re-implement this method in
   * the elxYourMetric, if you like.
   */
  virtual MeasureType
  GetExactValue(const ParametersType & parameters);

  /** \todo the method GetExactDerivative could as well be added here. */

  bool                             m_ShowExactMetricValue;
  ExactMetricImageSamplerPointer   m_ExactMetricSampler;
  MeasureType                      m_CurrentExactMetricValue;
  ExactMetricSampleGridSpacingType m_ExactMetricSampleGridSpacing;
  unsigned int                     m_ExactMetricEachXNumberOfIterations;

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);

  /** The deleted copy constructor. */
  MetricBase(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMetricBase.hxx"
#endif

#endif // end #ifndef elxMetricBase_h
