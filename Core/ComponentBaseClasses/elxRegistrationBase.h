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
#ifndef elxRegistrationBase_h
#define elxRegistrationBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkMultiResolutionImageRegistrationMethod2.h"

/** Mask support. */
#include "itkImageMaskSpatialObject.h"
#include "itkErodeMaskImageFilter.h"

namespace elastix
{

/**
 * \class RegistrationBase
 * \brief This class is the elastix base class for all Registration schemes.
 *
 * This class contains all the common functionality for Registrations.
 *
 * \parameter ErodeMask: a flag to determine if the masks should be eroded
 *    from one resolution level to another. Choose from {"true", "false"} \n
 *    example: <tt>(ErodeMask "false")</tt> \n
 *    The default is "true". The parameter may be specified for each
 *    resolution differently, but that's not obliged. The actual amount of
 *    erosion depends on the image pyramid. \n
 *    Erosion of the mask prevents the border / edge of the mask taken into account.
 *    This can be useful for example for ultrasound images,
 *    where you don't want to take into account values outside
 *    the US-beam, but where you also don't want to match the
 *    edge / border of this beam.
 *    For example for MRI's of the head, the borders of the head
 *    may be wanted to match, and there erosion should be avoided.
 * \parameter ErodeFixedMask: a flag to determine if the fixed mask(s) should be eroded
 *    from one resolution level to another. Choose from {"true", "false"} \n
 *    example: <tt>(ErodeFixedMask "true" "false")</tt>
 *    This setting overrules ErodeMask.\n
 * \parameter ErodeMovingMask: a flag to determine if the moving mask(s) should be eroded
 *    from one resolution level to another. Choose from {"true", "false"} \n
 *    example: <tt>(ErodeMovingMask "true" "false")</tt>
 *    This setting overrules ErodeMask.\n
 * \parameter ErodeFixedMask\<i\>: a flag to determine if the i-th fixed mask should be eroded
 *    from one resolution level to another. Choose from {"true", "false"} \n
 *    example: <tt>(ErodeFixedMask2 "true" "false")</tt>
 *    This setting overrules ErodeMask and ErodeFixedMask.\n
 * \parameter ErodeMovingMask\<i\>: a flag to determine if the i-th moving mask should be eroded
 *    from one resolution level to another. Choose from {"true", "false"} \n
 *    example: <tt>(ErodeMovingMask2 "true" "false")</tt>
 *    This setting overrules ErodeMask and ErodeMovingMask.\n
 *
 * \ingroup Registrations
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT RegistrationBase : public BaseComponentSE<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(RegistrationBase);

  /** Standard ITK stuff. */
  using Self = RegistrationBase;
  using Superclass = BaseComponentSE<TElastix>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(RegistrationBase, BaseComponentSE);

  /** Typedef's from Elastix. */
  using typename Superclass::ElastixType;
  using typename Superclass::RegistrationType;

  /** Other typedef's. */
  using FixedImageType = typename ElastixType::FixedImageType;
  using MovingImageType = typename ElastixType::MovingImageType;

  /** Get the dimension of the fixed image. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  /** Get the dimension of the moving image. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedef for ITKBaseType. */
  using ITKBaseType = itk::MultiResolutionImageRegistrationMethod2<FixedImageType, MovingImageType>;

  /** Typedef for mask erosion options */
  using UseMaskErosionArrayType = std::vector<bool>;

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


  /** Function to read the mask parameters from the configuration object.
   * \todo: move to RegistrationBase
   * Input:
   * \li an array that will contain a bool for each mask, saying if it needs erosion or not
   * \li the number of masks
   * \li whichMask: "Fixed" or "Moving"
   * \li the current resolution level
   * Output:
   * \li The function returns a bool that says if any mask needs erosion. If the number of masks
   * is zero, this bool will be false.
   * \li The useMaskErosionArray, which indicates for each mask whether it should be eroded.
   * If the number of masks is zero, this array will be empty.
   *
   * The function first checks Erode<Fixed,Moving>Mask\<i\>, with i the mask number,
   * then Erode<Fixed,Moving>Mask, and finally ErodeMask. So, if you do not specify
   * Erode<Fixed,Moving>Mask\<i\>, Erode<Fixed,Moving>Mask is tried, and then ErodeMask.
   * If you specify ErodeMask, that option will be used for all masks, fixed and moving!
   * All options can be specified for each resolution specifically, or at once for all
   * resolutions.
   */
  virtual bool
  ReadMaskParameters(UseMaskErosionArrayType & useMaskErosionArray,
                     const unsigned int        nrOfMasks,
                     const std::string &       whichMask,
                     const unsigned int        level) const;

protected:
  /** The constructor. */
  RegistrationBase() = default;
  /** The destructor. */
  ~RegistrationBase() override = default;

  /** Typedef's for mask support. */
  using MaskPixelType = typename ElastixType::MaskPixelType;
  using FixedMaskImageType = typename ElastixType::FixedMaskType;
  using MovingMaskImageType = typename ElastixType::MovingMaskType;
  using FixedMaskImagePointer = typename FixedMaskImageType::Pointer;
  using MovingMaskImagePointer = typename MovingMaskImageType::Pointer;
  using FixedMaskSpatialObjectType = itk::ImageMaskSpatialObject<Self::FixedImageDimension>;
  using MovingMaskSpatialObjectType = itk::ImageMaskSpatialObject<Self::MovingImageDimension>;
  using FixedMaskSpatialObjectPointer = typename FixedMaskSpatialObjectType::Pointer;
  using MovingMaskSpatialObjectPointer = typename MovingMaskSpatialObjectType::Pointer;

  using FixedImagePyramidType = typename ITKBaseType::FixedImagePyramidType;
  using MovingImagePyramidType = typename ITKBaseType::MovingImagePyramidType;

  /** Some typedef's used for eroding the masks */
  using FixedMaskErodeFilterType = itk::ErodeMaskImageFilter<FixedMaskImageType>;
  using FixedMaskErodeFilterPointer = typename FixedMaskErodeFilterType::Pointer;
  using MovingMaskErodeFilterType = itk::ErodeMaskImageFilter<MovingMaskImageType>;
  using MovingMaskErodeFilterPointer = typename MovingMaskErodeFilterType::Pointer;

  /** Generate a spatial object from a mask image, possibly after eroding the image
   * Input:
   * \li the mask as an image, consisting of 1's and 0's;
   * \li a boolean that determines whether mask erosion is needed
   * \li the image pyramid, which is needed to determines the amount of erosion
   *  (can be set to 0 if useMaskErosion == false
   * \li the resolution level
   * Output:
   * \li the mask as a spatial object, which can be set in a metric for example
   *
   * This function is used by the registration components
   */
  FixedMaskSpatialObjectPointer
  GenerateFixedMaskSpatialObject(const FixedMaskImageType *    maskImage,
                                 bool                          useMaskErosion,
                                 const FixedImagePyramidType * pyramid,
                                 unsigned int                  level) const;

  /** Generate a spatial object from a mask image, possibly after eroding the image
   * Input:
   * \li the mask as an image, consisting of 1's and 0's;
   * \li a boolean that determines whether mask erosion is needed
   * \li the image pyramid, which is needed to determines the amount of erosion
   *  (can be set to 0 if useMaskErosion == false
   * \li the resolution level
   * Output:
   * \li the mask as a spatial object, which can be set in a metric for example
   *
   * This function is used by the registration components
   */
  MovingMaskSpatialObjectPointer
  GenerateMovingMaskSpatialObject(const MovingMaskImageType *    maskImage,
                                  bool                           useMaskErosion,
                                  const MovingImagePyramidType * pyramid,
                                  unsigned int                   level) const;

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRegistrationBase.hxx"
#endif

#endif // end #ifndef elxRegistrationBase_h
