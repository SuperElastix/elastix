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
#ifndef elxGenericPyramidHelper_h
#define elxGenericPyramidHelper_h

#include <string>      // For is_same.
#include <type_traits> // For is_same.

#include "elxConfiguration.h"
#include "elxDeref.h"

namespace elastix
{
template <class TElastix>
class FixedGenericPyramid;

template <class TElastix>
class MovingGenericPyramid;

/** Internal helper class for FixedGenericPyramid and MovingGenericPyramid. */
class GenericPyramidHelper
{
public:
  /** Sets the schedule of a fixed or moving generic pyramid. */
  template <typename TPyramid>
  static void
  SetSchedule(TPyramid & pyramid)
  {
    using ElastixType = typename TPyramid::ElastixType;

    constexpr bool isFixed = std::is_same<TPyramid, FixedGenericPyramid<ElastixType>>::value;
    constexpr bool isMoving = std::is_same<TPyramid, MovingGenericPyramid<ElastixType>>::value;

    static_assert(isMoving != isFixed, "TPyramid must be either FixedGenericPyramid or MovingGenericPyramid!");

    const std::string      parameterPrefix = isFixed ? "Fixed" : "Moving";
    constexpr const char * pyramidAdjective = isFixed ? "fixed" : "moving";

    // It is assumed for more than ten years already that the configuration is not null, looking at
    // "src/Components/FixedImagePyramids/FixedGenericPyramid/elxFixedGenericPyramid.hxx" revision
    // f84ac0d1094ebdb13e456b1b8cf1f6f9bfcd0a38 "ENH: First checkin of a generic pyramid...", Marius Staring, 2012-02-02

    const Configuration & configuration = Deref(pyramid.GetConfiguration());

    /** Get the ImageDimension. */
    const unsigned int ImageDimension = TPyramid::ImageDimension;

    /** Read numberOfResolutions. */
    unsigned int numberOfResolutions = 3;
    configuration.ReadParameter(numberOfResolutions, "NumberOfResolutions", 0, true);
    if (numberOfResolutions == 0)
    {
      numberOfResolutions = 1;
    }

    /** Create a default schedule. Set the numberOfLevels first. */
    pyramid.GetAsITKBaseType()->SetNumberOfLevels(numberOfResolutions);
    typename TPyramid::RescaleScheduleType   rescaleSchedule = pyramid.GetRescaleSchedule();
    typename TPyramid::SmoothingScheduleType smoothingSchedule = pyramid.GetSmoothingSchedule();

    /** Set the schedule for rescaling.
     * The following parameter file fields can be used:
     * - ImagePyramidRescaleSchedule
     * - ImagePyramidSchedule
     * - Fixed/MovingImagePyramidRescaleSchedule
     * - Fixed/MovingImagePyramidSchedule
     */
    bool foundRescale = true;
    for (unsigned int i = 0; i < numberOfResolutions; ++i)
    {
      for (unsigned int j = 0; j < ImageDimension; ++j)
      {
        bool               ijfound = false;
        const unsigned int entrynr = i * ImageDimension + j;
        ijfound |= configuration.ReadParameter(rescaleSchedule[i][j], "ImagePyramidRescaleSchedule", entrynr, false);
        ijfound |= configuration.ReadParameter(rescaleSchedule[i][j], "ImagePyramidSchedule", entrynr, false);
        ijfound |= configuration.ReadParameter(
          rescaleSchedule[i][j], parameterPrefix + "ImagePyramidRescaleSchedule", entrynr, false);
        ijfound |=
          configuration.ReadParameter(rescaleSchedule[i][j], parameterPrefix + "ImagePyramidSchedule", entrynr, false);

        /** Remember if for at least one schedule element no value could be found. */
        foundRescale &= ijfound;

      } // end for ImageDimension
    }   // end for numberOfResolutions

    if (!foundRescale && pyramid.GetConfiguration()->GetPrintErrorMessages())
    {
      log::warn(std::ostringstream{} << "WARNING: the " << pyramidAdjective
                                     << " pyramid rescale schedule is not fully specified!\n"
                                     << "  A default pyramid rescale schedule is used.");
    }
    else
    {
      /** Set the rescale schedule into this class. */
      pyramid.SetRescaleSchedule(rescaleSchedule);

      const auto newSchedule = pyramid.GetRescaleSchedule();

      if (newSchedule != rescaleSchedule)
      {
        log::warn(
          std::ostringstream{} << "WARNING: the " << pyramidAdjective
                               << " pyramid rescale schedule is adjusted!\n  Input schedule from configuration:\n"
                               << rescaleSchedule << "\n  Adjusted schedule:\n"
                               << newSchedule);
      }
    }

    /** Set the schedule for smoothing (sigmas).
     * The following parameter file fields can be used:
     * - ImagePyramidSmoothingSchedule
     * - Fixed/MovingImagePyramidSmoothingSchedule
     */
    bool foundSmoothing = true;
    for (unsigned int i = 0; i < numberOfResolutions; ++i)
    {
      for (unsigned int j = 0; j < ImageDimension; ++j)
      {
        bool               ijfound = false;
        const unsigned int entrynr = i * ImageDimension + j;
        ijfound |=
          configuration.ReadParameter(smoothingSchedule[i][j], "ImagePyramidSmoothingSchedule", entrynr, false);
        ijfound |= configuration.ReadParameter(
          smoothingSchedule[i][j], parameterPrefix + "ImagePyramidSmoothingSchedule", entrynr, false);

        /** Remember if for at least one schedule element no value could be found. */
        foundSmoothing &= ijfound;

      } // end for ImageDimension
    }   // end for numberOfResolutions

    if (!foundSmoothing && pyramid.GetConfiguration()->GetPrintErrorMessages())
    {
      log::warn(std::ostringstream{} << "WARNING: the " << pyramidAdjective
                                     << " pyramid smoothing schedule is not fully specified!\n"
                                     << "  A default pyramid smoothing schedule is used.");
    }
    else
    {
      /** Set the rescale schedule into this class. */
      pyramid.SetSmoothingSchedule(smoothingSchedule);

      const auto newSchedule = pyramid.GetSmoothingSchedule();

      if (newSchedule != smoothingSchedule)
      {
        log::warn(
          std::ostringstream{} << "WARNING: the " << pyramidAdjective
                               << " pyramid smoothing schedule is adjusted!\n  Input schedule from configuration:\n"
                               << smoothingSchedule << "\n  Adjusted schedule:\n"
                               << newSchedule);
      }
    }

    // configuration.CountNumberOfParameterEntries( "ImagePyramidRescaleSchedule" );

    /** Use or skip rescaling within the pyramid. */
    /** Currently not used since the same can be obtained by just specifying
     * all ones in the rescale schedule. */
    /*bool useRescaleSchedule = true;
    configuration.ReadParameter( useRescaleSchedule,
          "UseImagePyramidRescaleSchedule", 0, false );
    pyramid.SetUseMultiResolutionRescaleSchedule( useRescaleSchedule );*/

    /** Use or skip smoothing within the pyramid. */
    /** Currently not used since the same can be obtained by just specifying
     * all zeros in the smoothing schedule. */
    /*bool useSmoothingSchedule = true;
    configuration.ReadParameter( useSmoothingSchedule,
          "UseImagePyramidSmoothingSchedule", 0, false );
    pyramid.SetUseMultiResolutionSmoothingSchedule( useSmoothingSchedule );*/

    /** Use or the resampler or the shrinker for rescaling within the pyramid. */
    bool useShrinkImageFilter = false;
    configuration.ReadParameter(useShrinkImageFilter, "ImagePyramidUseShrinkImageFilter", 0, false);
    pyramid.SetUseShrinkImageFilter(useShrinkImageFilter);

    /** Decide whether or not to compute the pyramid images only for the current
     * resolution. Setting the option to true saves memory, since only one level
     * of the pyramid gets allocated per resolution.
     */
    bool computeThisResolution = false;
    configuration.ReadParameter(computeThisResolution, "ComputePyramidImagesPerResolution", 0, false);
    pyramid.SetComputeOnlyForCurrentLevel(computeThisResolution);
  }
};

} // namespace elastix

#endif
