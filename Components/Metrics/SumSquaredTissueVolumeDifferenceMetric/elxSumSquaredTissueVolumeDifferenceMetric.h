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
#ifndef elxSumSquaredTissueVolumeDifferenceMetric_h
#define elxSumSquaredTissueVolumeDifferenceMetric_h

#include "elxIncludes.h"
#include "itkSumSquaredTissueVolumeDifferenceImageToImageMetric.h"
//#include "itkTimeProbe.h"

namespace elastix
{
/**
 * \class SumSquaredTissueVolumeDifferenceMetric
 * \brief A metric based on the itk::SumSquaredTissueVolumeDifferenceImageToImageMetric.
 *
 * \warning: This metric has only been evaluated on CT, where image intensity is
 * monotonically related to tissue density, and therefore mass. Performance in other
 * modalities such as MRI has not been explored.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "SumSquaredTissueVolumeDifference")</tt>
 * \parameter AirValue: Intensity value of air. \n
 *    example: <tt>(AirValue -1000.0)</tt> \n
 *    Default is -1000.0.
 * \parameter TissueValue: Intensity value of tissue. \n
 *    example: <tt>(TissueValue 55.0)</tt> \n
 *    Default is 55.0.
 *
 * \sa SumSquaredTissueVolumeDifferenceImageToImageMetric
 * \ingroup Metrics
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT SumSquaredTissueVolumeDifferenceMetric
  : public itk::SumSquaredTissueVolumeDifferenceImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                                   typename MetricBase<TElastix>::MovingImageType>
  , public MetricBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SumSquaredTissueVolumeDifferenceMetric);

  /** Standard ITK-stuff. */
  using Self = SumSquaredTissueVolumeDifferenceMetric;
  using Superclass1 =
    itk::SumSquaredTissueVolumeDifferenceImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                            typename MetricBase<TElastix>::MovingImageType>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SumSquaredTissueVolumeDifferenceMetric, itk::SumSquaredTissueVolumeDifferenceImageToImageMetric);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "SumSquaredTissueVolumeDifference")</tt>\n
   */
  elxClassNameMacro("SumSquaredTissueVolumeDifference");

  /** Typedefs from the superclass. */
  using typename Superclass1::CoordinateRepresentationType;
  using typename Superclass1::MovingImageType;
  using typename Superclass1::MovingImagePixelType;
  using typename Superclass1::MovingImageConstPointer;
  using typename Superclass1::FixedImageType;
  using typename Superclass1::FixedImageConstPointer;
  using typename Superclass1::FixedImageRegionType;
  using typename Superclass1::TransformType;
  using typename Superclass1::TransformPointer;
  using typename Superclass1::InputPointType;
  using typename Superclass1::OutputPointType;
  using typename Superclass1::TransformParametersType;
  using typename Superclass1::TransformJacobianType;
  using typename Superclass1::InterpolatorType;
  using typename Superclass1::InterpolatorPointer;
  using typename Superclass1::RealType;
  using typename Superclass1::GradientPixelType;
  using typename Superclass1::GradientImageType;
  using typename Superclass1::GradientImagePointer;
  using typename Superclass1::GradientImageFilterType;
  using typename Superclass1::GradientImageFilterPointer;
  using typename Superclass1::FixedImageMaskType;
  using typename Superclass1::FixedImageMaskPointer;
  using typename Superclass1::MovingImageMaskType;
  using typename Superclass1::MovingImageMaskPointer;
  using typename Superclass1::MeasureType;
  using typename Superclass1::DerivativeType;
  using typename Superclass1::ParametersType;
  using typename Superclass1::FixedImagePixelType;
  using typename Superclass1::MovingImageRegionType;
  using typename Superclass1::ImageSamplerType;
  using typename Superclass1::ImageSamplerPointer;
  using typename Superclass1::ImageSampleContainerType;
  using typename Superclass1::ImageSampleContainerPointer;
  using typename Superclass1::FixedImageLimiterType;
  using typename Superclass1::MovingImageLimiterType;
  using typename Superclass1::FixedImageLimiterOutputType;
  using typename Superclass1::MovingImageLimiterOutputType;
  using typename Superclass1::MovingImageDerivativeScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedef's inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;


  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize() override;

  /**
   * Do some things before each resolution:
   * \li Set AirValue setting
   * \li Set TissueValue setting
   */
  void
  BeforeEachResolution() override;

protected:
  /** The constructor. */
  SumSquaredTissueVolumeDifferenceMetric() = default;
  /** The destructor. */
  ~SumSquaredTissueVolumeDifferenceMetric() override = default;

private:
  elxOverrideGetSelfMacro;

}; // end class SumSquaredTissueVolumeDifferenceMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxSumSquaredTissueVolumeDifferenceMetric.hxx"
#endif

#endif // end #ifndef elxSumSquaredTissueVolumeDifferenceMetric_h
