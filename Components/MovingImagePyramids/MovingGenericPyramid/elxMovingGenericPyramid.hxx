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
#ifndef elxMovingGenericPyramid_hxx
#define elxMovingGenericPyramid_hxx

#include "elxMovingGenericPyramid.h"

namespace elastix
{

/**
 * ******************* SetMovingSchedule ***********************
 */

template <class TElastix>
void
MovingGenericPyramid<TElastix>::SetMovingSchedule()
{
  /** Get the ImageDimension. */
  const unsigned int MovingImageDimension = InputImageType::ImageDimension;

  /** Read numberOfResolutions. */
  unsigned int numberOfResolutions = 3;
  this->m_Configuration->ReadParameter(numberOfResolutions, "NumberOfResolutions", 0, true);
  if (numberOfResolutions == 0)
  {
    numberOfResolutions = 1;
  }

  /** Create a default movingSchedule. Set the numberOfLevels first. */
  this->GetAsITKBaseType()->SetNumberOfLevels(numberOfResolutions);
  RescaleScheduleType   rescaleSchedule = this->GetRescaleSchedule();
  SmoothingScheduleType smoothingSchedule = this->GetSmoothingSchedule();

  /** Set the schedule for rescaling.
   * The following parameter file fields can be used:
   * - ImagePyramidRescaleSchedule
   * - ImagePyramidSchedule
   * - MovingImagePyramidRescaleSchedule
   * - MovingImagePyramidSchedule
   */
  bool foundRescale = true;
  for (unsigned int i = 0; i < numberOfResolutions; ++i)
  {
    for (unsigned int j = 0; j < MovingImageDimension; ++j)
    {
      bool               ijfound = false;
      const unsigned int entrynr = i * MovingImageDimension + j;
      ijfound |=
        this->m_Configuration->ReadParameter(rescaleSchedule[i][j], "ImagePyramidRescaleSchedule", entrynr, false);
      ijfound |= this->m_Configuration->ReadParameter(rescaleSchedule[i][j], "ImagePyramidSchedule", entrynr, false);
      ijfound |= this->m_Configuration->ReadParameter(
        rescaleSchedule[i][j], "MovingImagePyramidRescaleSchedule", entrynr, false);
      ijfound |=
        this->m_Configuration->ReadParameter(rescaleSchedule[i][j], "MovingImagePyramidSchedule", entrynr, false);

      /** Remember if for at least one schedule element no value could be found. */
      foundRescale &= ijfound;

    } // end for MovingImageDimension
  }   // end for numberOfResolutions

  if (!foundRescale && this->GetConfiguration()->GetPrintErrorMessages())
  {
    xl::xout["warning"] << "WARNING: the moving pyramid rescale schedule is not fully specified!\n";
    xl::xout["warning"] << "  A default pyramid rescale schedule is used." << std::endl;
  }
  else
  {
    /** Set the rescale schedule into this class. */
    this->SetRescaleSchedule(rescaleSchedule);
  }

  /** Set the schedule for smoothing (sigmas).
   * The following parameter file fields can be used:
   * - ImagePyramidSmoothingSchedule
   * - MovingImagePyramidSmoothingSchedule
   */
  bool foundSmoothing = true;
  for (unsigned int i = 0; i < numberOfResolutions; ++i)
  {
    for (unsigned int j = 0; j < MovingImageDimension; ++j)
    {
      bool               ijfound = false;
      const unsigned int entrynr = i * MovingImageDimension + j;
      ijfound |=
        this->m_Configuration->ReadParameter(smoothingSchedule[i][j], "ImagePyramidSmoothingSchedule", entrynr, false);
      ijfound |= this->m_Configuration->ReadParameter(
        smoothingSchedule[i][j], "MovingImagePyramidSmoothingSchedule", entrynr, false);

      /** Remember if for at least one schedule element no value could be found. */
      foundSmoothing &= ijfound;

    } // end for MovingImageDimension
  }   // end for numberOfResolutions

  if (!foundSmoothing && this->GetConfiguration()->GetPrintErrorMessages())
  {
    xl::xout["warning"] << "WARNING: the moving pyramid smoothing schedule is not fully specified!\n";
    xl::xout["warning"] << "  A default pyramid smoothing schedule is used." << std::endl;
  }
  else
  {
    /** Set the rescale schedule into this class. */
    this->SetSmoothingSchedule(smoothingSchedule);
  }

  // this->m_Configuration->CountNumberOfParameterEntries( "ImagePyramidRescaleSchedule" );

  /** Use or skip rescaling within the pyramid. */
  /** Currently not used since the same can be obtained by just specifying
   * all ones in the rescale schedule. */
  /*bool useRescaleSchedule = true;
  this->m_Configuration->ReadParameter( useRescaleSchedule,
        "UseImagePyramidRescaleSchedule", 0, false );
  this->SetUseMultiResolutionRescaleSchedule( useRescaleSchedule );*/

  /** Use or skip smoothing within the pyramid. */
  /** Currently not used since the same can be obtained by just specifying
   * all zeros in the smoothing schedule. */
  /*bool useSmoothingSchedule = true;
  this->m_Configuration->ReadParameter( useSmoothingSchedule,
        "UseImagePyramidSmoothingSchedule", 0, false );
  this->SetUseMultiResolutionSmoothingSchedule( useSmoothingSchedule );*/

  /** Use or the resampler or the shrinker for rescaling within the pyramid. */
  bool useShrinkImageFilter = false;
  this->m_Configuration->ReadParameter(useShrinkImageFilter, "ImagePyramidUseShrinkImageFilter", 0, false);
  this->SetUseShrinkImageFilter(useShrinkImageFilter);

  /** Decide whether or not to compute the pyramid images only for the current
   * resolution. Setting the option to true saves memory, since only one level
   * of the pyramid gets allocated per resolution.
   */
  bool computeThisResolution = false;
  this->m_Configuration->ReadParameter(computeThisResolution, "ComputePyramidImagesPerResolution", 0, false);
  this->SetComputeOnlyForCurrentLevel(computeThisResolution);

} // end SetMovingSchedule()


/**
 * ******************* BeforeEachResolution ***********************
 */

template <class TElastix>
void
MovingGenericPyramid<TElastix>::BeforeEachResolution()
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** We let the pyramid filter know that we are in a next level.
   * Depending on a flag only at this point the output of the current level is computed,
   * or it was computed for all levels at once at initialization.
   */
  this->SetCurrentLevel(level);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxMovingGenericPyramid_hxx
