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
#ifndef elxMovingGenericPyramid_h
#define elxMovingGenericPyramid_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkGenericMultiResolutionPyramidImageFilter.h"

namespace elastix
{

/**
 * \class MovingGenericPyramid
 * \brief A pyramid based on the itk::GenericMultiResolutionPyramidImageFilter.
 *
 * It is generic since it has all functionality that the MovingRecursivePyramid,
 * MovingShrinkingPyramid and MovingSmoothingPyramid have together.
 * This pyramid has two separate schedules: one for size rescaling and one for
 * the Gaussian smoothing factor sigma. In addition, it has an option to compute
 * pyramid output per resolution, and not all at once, to reduce memory consumption.
 *
 * The parameters used in this class are:
 * \parameter MovingImagePyramid: Select this pyramid as follows:\n
 *    <tt>(MovingImagePyramid "MovingGenericImagePyramid")</tt>
 * \parameter MovingImagePyramidRescaleSchedule: downsampling factors for the moving image pyramid.\n
 *    For each dimension, for each resolution level, the downsampling factor of the
 *    moving image can be specified.\n
 *    Syntax for 2D images:\n
 *    <tt>(MovingImagePyramidRescaleSchedule <reslevel0,dim0> <reslevel0,dim1> <reslevel1,dim0> <reslevel1,dim1>
 * ...)</tt>\n example: <tt>(MovingImagePyramidRescaleSchedule 4 4 2 2 1 1)</tt>\n Default: isotropic, halved in each
 * resolution, so, like in the example. If ImagePyramidRescaleSchedule is specified, that schedule is used for both
 * moving and moving image pyramid. \parameter ImagePyramidRescaleSchedule: rescale schedule for both pyramids
 * \parameter ImagePyramidSchedule: same as ImagePyramidRescaleSchedule
 * \parameter MovingImagePyramidSchedule: same as MovingImagePyramidRescaleSchedule
 * \parameter MovingImagePyramidSmoothingSchedule: sigma's for smoothing the moving image pyramid.\n
 *    For each dimension, for each resolution level, the sigma of the
 *    moving image can be specified.\n
 *    Syntax for 2D images:\n
 *    <tt>(MovingImagePyramidSmoothingSchedule <reslevel0,dim0> <reslevel0,dim1> <reslevel1,dim0> <reslevel1,dim1>
 * ...)</tt>\n example: <tt>(MovingImagePyramidSmoothingSchedule 4 4 2 2 1 1)</tt>\n Default: 0.5 x rescale_factor x
 * moving_image_spacing.\n If ImagePyramidSmoothingSchedule is specified, that schedule is used for both moving and
 * moving image pyramid. \parameter ImagePyramidSmoothingSchedule: smoothing schedule for both pyramids \parameter
 * ComputePyramidImagesPerResolution: Flag to specify if all resolution levels are computed at once, or per resolution.
 * Latter saves memory.\n example: <tt>(ComputePyramidImagesPerResolution "true")</tt>\n Default false. \parameter
 * ImagePyramidUseShrinkImageFilter: Flag to specify if the ShrinkingImageFilter is used for rescaling the image, or the
 * ResampleImageFilter. Shrinker is faster.\n example: <tt>(ImagePyramidUseShrinkImageFilter "true")</tt>\n Default
 * false, so by default the resampler is used.
 *
 * \ingroup ImagePyramids
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MovingGenericPyramid
  : public itk::GenericMultiResolutionPyramidImageFilter<typename MovingImagePyramidBase<TElastix>::InputImageType,
                                                         typename MovingImagePyramidBase<TElastix>::OutputImageType>
  , public MovingImagePyramidBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MovingGenericPyramid);

  /** Standard ITK-stuff. */
  using Self = MovingGenericPyramid;
  using Superclass1 =
    itk::GenericMultiResolutionPyramidImageFilter<typename MovingImagePyramidBase<TElastix>::InputImageType,
                                                  typename MovingImagePyramidBase<TElastix>::OutputImageType>;
  using Superclass2 = MovingImagePyramidBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MovingGenericPyramid, GenericMultiResolutionPyramidImageFilter);

  /** Name of this class.
   * Use this name in the parameter file to select this specific pyramid. \n
   * example: <tt>(MovingImagePyramid "MovingGenericImagePyramid")</tt>\n
   */
  elxClassNameMacro("MovingGenericImagePyramid");

  /** Get the ImageDimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::InputImageType;
  using typename Superclass1::OutputImageType;
  using typename Superclass1::InputImagePointer;
  using typename Superclass1::OutputImagePointer;
  using typename Superclass1::InputImageConstPointer;
  using typename Superclass1::ScheduleType;
  using typename Superclass1::RescaleScheduleType;
  using typename Superclass1::SmoothingScheduleType;

  /** Typedefs inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Method for setting the schedule. Override from MovingImagePyramidBase,
   * since we now have two schedules, rescaling and smoothing.
   */
  void
  SetMovingSchedule() override;

  /** Update the current resolution level. */
  void
  BeforeEachResolution() override;

protected:
  /** The constructor. */
  MovingGenericPyramid() = default;
  /** The destructor. */
  ~MovingGenericPyramid() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMovingGenericPyramid.hxx"
#endif

#endif // end #ifndef elxMovingGenericPyramid_h
