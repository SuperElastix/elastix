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
#ifndef elxCorrespondingPointsEuclideanDistanceMetric_h
#define elxCorrespondingPointsEuclideanDistanceMetric_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkCorrespondingPointsEuclideanDistancePointMetric.h"

namespace elastix
{

/**
 * \class CorrespondingPointsEuclideanDistanceMetric
 * \brief An metric based on the itk::CorrespondingPointsEuclideanDistancePointMetric.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "CorrespondingPointsEuclideanDistanceMetric")</tt>
 *
 * \ingroup Metrics
 *
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT CorrespondingPointsEuclideanDistanceMetric
  : public itk::CorrespondingPointsEuclideanDistancePointMetric<typename MetricBase<TElastix>::FixedPointSetType,
                                                                typename MetricBase<TElastix>::MovingPointSetType>
  , public MetricBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef CorrespondingPointsEuclideanDistanceMetric Self;
  typedef itk::CorrespondingPointsEuclideanDistancePointMetric<typename MetricBase<TElastix>::FixedPointSetType,
                                                               typename MetricBase<TElastix>::MovingPointSetType>
                                        Superclass1;
  typedef MetricBase<TElastix>          Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CorrespondingPointsEuclideanDistanceMetric, itk::CorrespondingPointsEuclideanDistancePointMetric);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "CorrespondingPointsEuclideanDistanceMetric")</tt>\n
   */
  elxClassNameMacro("CorrespondingPointsEuclideanDistanceMetric");

  /** Typedefs from the superclass. */
  typedef typename Superclass1::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass1::FixedPointSetType            FixedPointSetType;
  typedef typename Superclass1::FixedPointSetConstPointer    FixedPointSetConstPointer;
  typedef typename Superclass1::MovingPointSetType           MovingPointSetType;
  typedef typename Superclass1::MovingPointSetConstPointer   MovingPointSetConstPointer;

  //  typedef typename Superclass1::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass1::TransformType           TransformType;
  typedef typename Superclass1::TransformPointer        TransformPointer;
  typedef typename Superclass1::InputPointType          InputPointType;
  typedef typename Superclass1::OutputPointType         OutputPointType;
  typedef typename Superclass1::TransformParametersType TransformParametersType;
  typedef typename Superclass1::TransformJacobianType   TransformJacobianType;
  //  typedef typename Superclass1::RealType                   RealType;
  typedef typename Superclass1::FixedImageMaskType     FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer  FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType    MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer MovingImageMaskPointer;
  typedef typename Superclass1::MeasureType            MeasureType;
  typedef typename Superclass1::DerivativeType         DerivativeType;
  typedef typename Superclass1::ParametersType         ParametersType;

  /** Typedefs inherited from elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;
  typedef typename Superclass2::FixedImageType       FixedImageType;
  typedef typename Superclass2::MovingImageType      MovingImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Assuming fixed and moving pointsets are of equal type, which implicitly
   * assumes that the fixed and moving image are of the same type.
   */
  typedef FixedPointSetType PointSetType;
  typedef FixedImageType    ImageType;

  /** Sets up a timer to measure the initialization time and calls the
   * Superclass' implementation.
   */
  void
  Initialize(void) override;

  /**
   * Do some things before all:
   * \li Check and print the command line arguments fp and mp.
   *   This should be done in BeforeAllBase and not BeforeAll.
   */
  int
  BeforeAllBase(void) override;

  /**
   * Do some things before registration:
   * \li Load and set the pointsets.
   */
  void
  BeforeRegistration(void) override;

  /** Function to read the corresponding points. */
  unsigned int
  ReadLandmarks(const std::string &                    landmarkFileName,
                typename PointSetType::Pointer &       pointSet,
                const typename ImageType::ConstPointer image);

  /** Overwrite to silence warning. */
  void
  SelectNewSamples(void) override
  {}

protected:
  /** The constructor. */
  CorrespondingPointsEuclideanDistanceMetric() = default;
  /** The destructor. */
  ~CorrespondingPointsEuclideanDistanceMetric() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  CorrespondingPointsEuclideanDistanceMetric(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxCorrespondingPointsEuclideanDistanceMetric.hxx"
#endif

#endif // end #ifndef elxCorrespondingPointsEuclideanDistanceMetric_h
