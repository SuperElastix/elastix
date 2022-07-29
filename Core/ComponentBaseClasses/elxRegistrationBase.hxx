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
#ifndef elxRegistrationBase_hxx
#define elxRegistrationBase_hxx

#include "elxRegistrationBase.h"

namespace elastix
{

/**
 * ********************* ReadMaskParameters ************************
 */

template <class TElastix>
bool
RegistrationBase<TElastix>::ReadMaskParameters(UseMaskErosionArrayType & useMaskErosionArray,
                                               const unsigned int        nrOfMasks,
                                               const std::string &       whichMask,
                                               const unsigned int        level) const
{
  /** Read whether mask erosion is wanted, if any masks were supplied. */

  /** Bool that remembers if mask erosion is wanted in any of the masks
   * remains false when no masks are used. This bool will be output.
   */
  bool useMaskErosion = false;

  /** Array of bools, that remembers for each mask if erosion is wanted. */
  useMaskErosionArray.resize(nrOfMasks, false);

  /** "ErodeFixedMask" or "ErodeMovingMask". */
  std::string whichErodeMaskOption("Erode");
  whichErodeMaskOption += whichMask;
  whichErodeMaskOption += "Mask";

  /** Read the parameters. */
  if (nrOfMasks > 0)
  {
    /** Default values for all masks. Look for ErodeMask, or Erode<Fixed,Moving>Mask. */
    bool erosionOrNot = true;
    this->GetConfiguration()->ReadParameter(erosionOrNot, "ErodeMask", "", level, 0, false);
    this->GetConfiguration()->ReadParameter(erosionOrNot, whichErodeMaskOption, "", level, 0);
    if (erosionOrNot)
    {
      /** fill with 'true's. */
      useMaskErosionArray.clear();
      useMaskErosionArray.resize(nrOfMasks, true);
    }

    /** Try to read an erode mask parameter given for a specified mask only:
     * (ErodeFixedMask0 "true" "false" ) for example.
     */
    for (unsigned int i = 0; i < nrOfMasks; ++i)
    {
      std::ostringstream makestring;
      makestring << whichErodeMaskOption << i; // key for parameter file
      bool erosionOrNot_i = erosionOrNot;      // default value
      this->GetConfiguration()->ReadParameter(erosionOrNot_i, makestring.str(), "", level, 0, false);
      if (erosionOrNot_i)
      {
        useMaskErosionArray[i] = true;
      }
      else
      {
        useMaskErosionArray[i] = false;
      }
      /** Check if mask erosion is wanted in any of the masks. */
      useMaskErosion |= useMaskErosionArray[i];
    }
  } // end if nrOfMasks > 0

  return useMaskErosion;

} // end ReadMaskParameters()


/**
 * ******************* GenerateFixedMaskSpatialObject **********************
 */

template <class TElastix>
auto
RegistrationBase<TElastix>::GenerateFixedMaskSpatialObject(const FixedMaskImageType *    maskImage,
                                                           bool                          useMaskErosion,
                                                           const FixedImagePyramidType * pyramid,
                                                           unsigned int level) const -> FixedMaskSpatialObjectPointer
{
  FixedMaskSpatialObjectPointer fixedMaskSpatialObject; // default-constructed (null)
  if (!maskImage)
  {
    return fixedMaskSpatialObject;
  }
  fixedMaskSpatialObject = FixedMaskSpatialObjectType::New();

  /** Just convert to spatial object if no erosion is needed. */
  if (!useMaskErosion || !pyramid)
  {
    fixedMaskSpatialObject->SetImage(maskImage);
    fixedMaskSpatialObject->Update();
    return fixedMaskSpatialObject;
  }

  /** Erode, and convert to spatial object. */
  FixedMaskErodeFilterPointer erosion = FixedMaskErodeFilterType::New();
  erosion->SetInput(maskImage);
  erosion->SetSchedule(pyramid->GetSchedule());
  erosion->SetIsMovingMask(false);
  erosion->SetResolutionLevel(level);

  /** Set output of the erosion to fixedImageMaskAsImage. */
  FixedMaskImagePointer erodedFixedMaskAsImage = erosion->GetOutput();

  /** Do the erosion. */
  try
  {
    erodedFixedMaskAsImage->Update();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("RegistrationBase - UpdateMasks()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError while eroding the fixed mask.\n";
    excp.SetDescription(err_str);
    /** Pass the exception to an higher level. */
    throw;
  }

  /** Release some memory. */
  erodedFixedMaskAsImage->DisconnectPipeline();

  fixedMaskSpatialObject->SetImage(erodedFixedMaskAsImage);
  fixedMaskSpatialObject->Update();
  return fixedMaskSpatialObject;

} // end GenerateFixedMaskSpatialObject()


/**
 * ******************* GenerateMovingMaskSpatialObject **********************
 */

template <class TElastix>
auto
RegistrationBase<TElastix>::GenerateMovingMaskSpatialObject(const MovingMaskImageType *    maskImage,
                                                            bool                           useMaskErosion,
                                                            const MovingImagePyramidType * pyramid,
                                                            unsigned int level) const -> MovingMaskSpatialObjectPointer
{
  MovingMaskSpatialObjectPointer movingMaskSpatialObject; // default-constructed (null)
  if (!maskImage)
  {
    return movingMaskSpatialObject;
  }
  movingMaskSpatialObject = MovingMaskSpatialObjectType::New();

  /** Just convert to spatial object if no erosion is needed. */
  if (!useMaskErosion || !pyramid)
  {
    movingMaskSpatialObject->SetImage(maskImage);
    movingMaskSpatialObject->Update();
    return movingMaskSpatialObject;
  }

  /** Erode, and convert to spatial object. */
  MovingMaskErodeFilterPointer erosion = MovingMaskErodeFilterType::New();
  erosion->SetInput(maskImage);
  erosion->SetSchedule(pyramid->GetSchedule());
  erosion->SetIsMovingMask(true);
  erosion->SetResolutionLevel(level);

  /** Set output of the erosion to movingImageMaskAsImage. */
  MovingMaskImagePointer erodedMovingMaskAsImage = erosion->GetOutput();

  /** Do the erosion. */
  try
  {
    erodedMovingMaskAsImage->Update();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("RegistrationBase - UpdateMasks()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError while eroding the moving mask.\n";
    excp.SetDescription(err_str);
    /** Pass the exception to an higher level. */
    throw;
  }

  /** Release some memory */
  erodedMovingMaskAsImage->DisconnectPipeline();

  movingMaskSpatialObject->SetImage(erodedMovingMaskAsImage);
  movingMaskSpatialObject->Update();
  return movingMaskSpatialObject;

} // end GenerateMovingMaskSpatialObject()


} // end namespace elastix

#endif // end #ifndef elxRegistrationBase_hxx
