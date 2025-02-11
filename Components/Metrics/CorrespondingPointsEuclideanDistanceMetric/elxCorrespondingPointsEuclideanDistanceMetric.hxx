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
#ifndef elxCorrespondingPointsEuclideanDistanceMetric_hxx
#define elxCorrespondingPointsEuclideanDistanceMetric_hxx

#include "elxCorrespondingPointsEuclideanDistanceMetric.h"
#include "itkTransformixInputPointFileReader.h"
#include "itkTimeProbe.h"
#include <itkDeref.h>
#include <cstdint> // For int64_t.

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <typename TElastix>
void
CorrespondingPointsEuclideanDistanceMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of CorrespondingPointsEuclideanDistance metric took: "
                                 << static_cast<long>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeAllBase ***********************
 */

template <typename TElastix>
int
CorrespondingPointsEuclideanDistanceMetric<TElastix>::BeforeAllBase()
{
  this->Superclass2::BeforeAllBase();

  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Check if the current configuration uses this metric. */
  unsigned int count = 0;
  for (unsigned int i = 0; i < configuration.CountNumberOfParameterEntries("Metric"); ++i)
  {
    std::string metricName = "";
    configuration.ReadParameter(metricName, "Metric", i);
    if (metricName == "CorrespondingPointsEuclideanDistanceMetric")
    {
      ++count;
    }
  }
  if (count == 0)
  {
    return 0;
  }

  /** Check Command line options and print them to the log file. */
  log::info("Command line options from CorrespondingPointsEuclideanDistanceMetric:");

  /** Check for appearance of "-fp". */
  if (const std::string commandLineArgument = configuration.GetCommandLineArgument("-fp"); commandLineArgument.empty())
  {
    log::info("-fp       unspecified");
  }
  else
  {
    log::info("-fp       " + commandLineArgument);
  }

  /** Check for appearance of "-mp". */
  if (const std::string commandLineArgument = configuration.GetCommandLineArgument("-mp"); commandLineArgument.empty())
  {
    log::info("-mp       unspecified");
  }
  else
  {
    log::info("-mp       " + commandLineArgument);
  }

  /** Return a value. */
  return 0;

} // end BeforeAllBase()


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
CorrespondingPointsEuclideanDistanceMetric<TElastix>::BeforeRegistration()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());
  const TElastix &      elastixObject = itk::Deref(Superclass2::GetElastix());

  const auto makeConstPointSet = [](const itk::Object * object) -> itk::SmartPointer<const PointSetType> {
    using ContainerType = typename PointSetType::PointsContainer;

    if (const auto * const points = dynamic_cast<const ContainerType *>(object))
    {
      const auto pointSet = PointSetType::New();
      // Note: This const_cast should be safe, because the returned point set is "const".
      pointSet->SetPoints(const_cast<ContainerType *>(points));
      return pointSet;
    }
    return nullptr;
  };

  itk::SmartPointer<const PointSetType> fixedPointSet = makeConstPointSet(elastixObject.GetFixedPoints());
  itk::SmartPointer<const PointSetType> movingPointSet = makeConstPointSet(elastixObject.GetMovingPoints());

  if (const std::string commandLineArgument = configuration.GetCommandLineArgument("-fp"); !commandLineArgument.empty())
  {
    /** Read and set the fixed pointset. */
    fixedPointSet = ReadLandmarks(commandLineArgument, elastixObject.GetFixedImage());
  }

  if (const std::string commandLineArgument = configuration.GetCommandLineArgument("-mp"); !commandLineArgument.empty())
  {
    /** Read and set the moving pointset. */
    movingPointSet = ReadLandmarks(commandLineArgument, elastixObject.GetMovingImage());
  }

  if (fixedPointSet)
  {
    this->SetFixedPointSet(fixedPointSet);
  }
  else
  {
    itkExceptionMacro("ERROR: the fixed pointset is not specified!");
  }

  if (movingPointSet)
  {
    this->SetMovingPointSet(movingPointSet);
  }
  else
  {
    itkExceptionMacro("ERROR: the moving pointset is not specified!");
  }

  const itk::SizeValueType nrOfFixedPoints = fixedPointSet->GetNumberOfPoints();
  const itk::SizeValueType nrOfMovingPoints = movingPointSet->GetNumberOfPoints();

  /** Check. */
  if (nrOfFixedPoints != nrOfMovingPoints)
  {
    itkExceptionMacro("ERROR: the number of points in the fixed pointset ("
                      << nrOfFixedPoints << ") does not match that of the moving pointset (" << nrOfMovingPoints
                      << "). The points do not correspond. ");
  }

} // end BeforeRegistration()


/**
 * ***************** ReadLandmarks ***********************
 */

template <typename TElastix>
auto
CorrespondingPointsEuclideanDistanceMetric<TElastix>::ReadLandmarks(const std::string & landmarkFileName,
                                                                    const typename ImageType::ConstPointer image)
  -> itk::SmartPointer<PointSetType>
{
  /** Typedefs. */
  using IndexType = typename ImageType::IndexType;
  using IndexValueType = typename ImageType::IndexValueType;
  using PointType = typename ImageType::PointType;

  log::info(std::ostringstream{} << "Loading landmarks for " << this->GetComponentLabel() << ":"
                                 << this->elxGetClassName() << ".");

  /** Read the landmarks. */
  auto reader = itk::TransformixInputPointFileReader<PointSetType>::New();
  reader->SetFileName(landmarkFileName);
  log::info(std::ostringstream{} << "  Reading landmark file: " << landmarkFileName);
  try
  {
    reader->Update();
  }
  catch (const itk::ExceptionObject & err)
  {
    log::error(std::ostringstream{} << "  Error while opening " << landmarkFileName << '\n' << err);
    itkExceptionMacro("ERROR: unable to configure " << this->GetComponentLabel());
  }

  /** Some user-feedback. */
  const unsigned int nrofpoints = reader->GetNumberOfPoints();
  if (reader->GetPointsAreIndices())
  {
    log::info("  Landmarks are specified as image indices.");
  }
  else
  {
    log::info("  Landmarks are specified in world coordinates.");
  }
  log::info(std::ostringstream{} << "  Number of specified points: " << nrofpoints);

  /** Get the pointset. */
  const itk::SmartPointer<PointSetType> pointSet = reader->GetOutput();

  /** Convert from index to point if necessary */
  pointSet->DisconnectPipeline();
  if (reader->GetPointsAreIndices())
  {
    /** Convert to world coordinates */
    for (unsigned int j = 0; j < nrofpoints; ++j)
    {
      /** The landmarks from the pointSet are indices. We first cast to the
       * proper type, and then convert it to world coordinates.
       */
      PointType point;
      IndexType index;
      pointSet->GetPoint(j, &point);
      for (unsigned int d = 0; d < FixedImageDimension; ++d)
      {
        index[d] = static_cast<IndexValueType>(itk::Math::Round<std::int64_t>(point[d]));
      }

      /** Compute the input point in physical coordinates. */
      image->TransformIndexToPhysicalPoint(index, point);
      pointSet->SetPoint(j, point);

    } // end for all points
  } // end for points are indices

  return pointSet;

} // end ReadLandmarks()


} // end namespace elastix

#endif // end #ifndef elxCorrespondingPointsEuclideanDistanceMetric_hxx
