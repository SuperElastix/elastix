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

#ifndef elxMovingImagePyramidBase_h
#define elxMovingImagePyramidBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkObject.h"

#include "itkMultiResolutionPyramidImageFilter.h"

namespace elastix
{

/**
 * \class MovingImagePyramidBase
 * \brief This class is the elastix base class for all MovingImagePyramids.
 *
 * This class contains all the common functionality for MovingImagePyramids.
 *
 * \parameter MovingImagePyramidSchedule: downsampling factors for the image pyramid.\n
 *    For each dimension, for each resolution level, the downsampling factor of the
 *    moving image can be specified.\n
 *    Syntax for 2D images:\n
 *    <tt>(MovingImagePyramidSchedule <reslevel0,dim0> <reslevel0,dim1> <reslevel1,dim0> <reslevel1,dim1> ...)</tt>
 *    example: <tt>(MovingImagePyramidSchedule  4 4 2 2 1 1)</tt>\n
 *    Default: isotropic, halved in each resolution, so, like in the example. If
 *    ImagePyramidSchedule is specified, that schedule is used for both fixed and moving image pyramid.
 * \parameter ImagePyramidSchedule: downsampling factors for fixed and moving image pyramids.\n
 *    example: <tt>(ImagePyramidSchedule  4 4 2 2 1 1)</tt> \n
 *    Used as a default when MovingImagePyramidSchedule is not specified. If both are omitted,
 *    a default schedule is assumed: isotropic, halved in each resolution, so, like in the example.
 * \parameter WritePyramidImagesAfterEachResolution: ...\n
 *    example: <tt>(WritePyramidImagesAfterEachResolution "true")</tt>\n
 *    default "false".
 *
 * \ingroup ImagePyramids
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MovingImagePyramidBase : public BaseComponentSE<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MovingImagePyramidBase);

  /** Standard ITK-stuff. */
  using Self = MovingImagePyramidBase;
  using Superclass = BaseComponentSE<TElastix>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MovingImagePyramidBase, BaseComponentSE);

  /** Typedefs inherited from the superclass. */
  using typename Superclass::ElastixType;
  using typename Superclass::RegistrationType;

  /** Typedefs inherited from Elastix. */
  using InputImageType = typename ElastixType::MovingImageType;
  using OutputImageType = typename ElastixType::MovingImageType;

  /** Other typedef's. */
  using ITKBaseType = itk::MultiResolutionPyramidImageFilter<InputImageType, OutputImageType>;

  /** Typedef's from ITKBaseType. */
  using ScheduleType = typename ITKBaseType::ScheduleType;

  /** Retrieves this object as ITKBaseType. */
  ITKBaseType *
  GetAsITKBaseType()
  {
    return &(this->GetSelf());
  }


  /** Retrieves this object as ITKBaseType, to use in const functions. */
  const ITKBaseType *
  GetAsITKBaseType() const
  {
    return &(this->GetSelf());
  }


  /** Execute stuff before the actual registration:
   * \li Set the schedule of the moving image pyramid.
   */
  void
  BeforeRegistrationBase() override;

  /** Execute stuff before each resolution:
   * \li Write the pyramid image to file.
   */
  void
  BeforeEachResolutionBase() override;

  /** Method for setting the schedule. */
  virtual void
  SetMovingSchedule();

  /** Method to write the pyramid image. */
  void
  WritePyramidImage(const std::string & filename,
                    const unsigned int  level); // const;

protected:
  /** The constructor. */
  MovingImagePyramidBase() = default;
  /** The destructor. */
  ~MovingImagePyramidBase() override = default;

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMovingImagePyramidBase.hxx"
#endif

#endif // end #ifndef elxMovingImagePyramidBase_h
