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

template <typename TElastix>
class ITK_TEMPLATE_EXPORT CorrespondingPointsEuclideanDistanceMetric
  : public itk::CorrespondingPointsEuclideanDistancePointMetric<typename MetricBase<TElastix>::FixedPointSetType,
                                                                typename MetricBase<TElastix>::MovingPointSetType>
  , public MetricBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CorrespondingPointsEuclideanDistanceMetric);

  /** Standard ITK-stuff. */
  using Self = CorrespondingPointsEuclideanDistanceMetric;
  using Superclass1 =
    itk::CorrespondingPointsEuclideanDistancePointMetric<typename MetricBase<TElastix>::FixedPointSetType,
                                                         typename MetricBase<TElastix>::MovingPointSetType>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CorrespondingPointsEuclideanDistanceMetric);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "CorrespondingPointsEuclideanDistanceMetric")</tt>\n
   */
  elxClassNameMacro("CorrespondingPointsEuclideanDistanceMetric");

  /** Typedefs from the superclass. */
  using typename Superclass1::FixedPointSetType;
  using typename Superclass1::MovingPointSetType;

  /** Typedefs inherited from elastix. */
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Assuming fixed and moving pointsets are of equal type, which implicitly
   * assumes that the fixed and moving image are of the same type.
   */
  using PointSetType = FixedPointSetType;
  using ImageType = FixedImageType;

  /** Sets up a timer to measure the initialization time and calls the
   * Superclass' implementation.
   */
  void
  Initialize() override;

  /**
   * Do some things before all:
   * \li Check and print the command line arguments fp and mp.
   *   This should be done in BeforeAllBase and not BeforeAll.
   */
  int
  BeforeAllBase() override;

  /**
   * Do some things before registration:
   * \li Load and set the pointsets.
   */
  void
  BeforeRegistration() override;

  /** Overwrite to silence warning. */
  void
  SelectNewSamples() override
  {}

protected:
  /** The constructor. */
  CorrespondingPointsEuclideanDistanceMetric() = default;
  /** The destructor. */
  ~CorrespondingPointsEuclideanDistanceMetric() override = default;

private:
  elxOverrideGetSelfMacro;

  /** Function to read the corresponding points. */
  itk::SmartPointer<PointSetType>
  ReadLandmarks(const std::string & landmarkFileName, const typename ImageType::ConstPointer image);
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxCorrespondingPointsEuclideanDistanceMetric.hxx"
#endif

#endif // end #ifndef elxCorrespondingPointsEuclideanDistanceMetric_h
