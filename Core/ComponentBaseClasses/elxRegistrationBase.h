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
#ifndef __elxRegistrationBase_h
#define __elxRegistrationBase_h

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

template< class TElastix >
class RegistrationBase : public BaseComponentSE< TElastix >
{
public:

  /** Standard ITK stuff. */
  typedef RegistrationBase            Self;
  typedef BaseComponentSE< TElastix > Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro( RegistrationBase, BaseComponentSE );

  /** Typedef's from Elastix. */
  typedef typename Superclass::ElastixType          ElastixType;
  typedef typename Superclass::ElastixPointer       ElastixPointer;
  typedef typename Superclass::ConfigurationType    ConfigurationType;
  typedef typename Superclass::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass::RegistrationType     RegistrationType;
  typedef typename Superclass::RegistrationPointer  RegistrationPointer;

  /** Other typedef's. */
  typedef typename ElastixType::FixedImageType  FixedImageType;
  typedef typename ElastixType::MovingImageType MovingImageType;

  /** Get the dimension of the fixed image. */
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
  /** Get the dimension of the moving image. */
  itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

  /** Typedef for ITKBaseType. */
  typedef itk::MultiResolutionImageRegistrationMethod2<
    FixedImageType, MovingImageType >       ITKBaseType;

  /** Typedef for mask erosion options */
  typedef std::vector< bool > UseMaskErosionArrayType;

  /** Cast to ITKBaseType. */
  virtual ITKBaseType * GetAsITKBaseType( void )
  {
    return dynamic_cast< ITKBaseType * >( this );
  }


  /** Cast to ITKBaseType, to use in const functions. */
  virtual const ITKBaseType * GetAsITKBaseType( void ) const
  {
    return dynamic_cast< const ITKBaseType * >( this );
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
  virtual bool ReadMaskParameters(
    UseMaskErosionArrayType & useMaskErosionArray,
    const unsigned int nrOfMasks,
    const std::string & whichMask,
    const unsigned int level ) const;

protected:

  /** The constructor. */
  RegistrationBase() {}
  /** The destructor. */
  ~RegistrationBase() override {}

  /** Typedef's for mask support. */
  typedef typename ElastixType::MaskPixelType   MaskPixelType;
  typedef typename ElastixType::FixedMaskType   FixedMaskImageType;
  typedef typename ElastixType::MovingMaskType  MovingMaskImageType;
  typedef typename FixedMaskImageType::Pointer  FixedMaskImagePointer;
  typedef typename MovingMaskImageType::Pointer MovingMaskImagePointer;
  typedef itk::ImageMaskSpatialObject<
    itkGetStaticConstMacro( FixedImageDimension ) >          FixedMaskSpatialObjectType;
  typedef itk::ImageMaskSpatialObject<
    itkGetStaticConstMacro( MovingImageDimension ) >         MovingMaskSpatialObjectType;
  typedef typename
    FixedMaskSpatialObjectType::Pointer FixedMaskSpatialObjectPointer;
  typedef typename
    MovingMaskSpatialObjectType::Pointer MovingMaskSpatialObjectPointer;

  typedef typename ITKBaseType::FixedImagePyramidType  FixedImagePyramidType;
  typedef typename ITKBaseType::MovingImagePyramidType MovingImagePyramidType;

  /** Some typedef's used for eroding the masks */
  typedef itk::ErodeMaskImageFilter< FixedMaskImageType >  FixedMaskErodeFilterType;
  typedef typename FixedMaskErodeFilterType::Pointer       FixedMaskErodeFilterPointer;
  typedef itk::ErodeMaskImageFilter< MovingMaskImageType > MovingMaskErodeFilterType;
  typedef typename MovingMaskErodeFilterType::Pointer      MovingMaskErodeFilterPointer;

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
  FixedMaskSpatialObjectPointer GenerateFixedMaskSpatialObject(
    const FixedMaskImageType * maskImage, bool useMaskErosion,
    const FixedImagePyramidType * pyramid, unsigned int level ) const;

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
  MovingMaskSpatialObjectPointer GenerateMovingMaskSpatialObject(
    const MovingMaskImageType * maskImage, bool useMaskErosion,
    const MovingImagePyramidType * pyramid, unsigned int level ) const;

private:

  /** The private constructor. */
  RegistrationBase( const Self & );   // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );     // purposely not implemented

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxRegistrationBase.hxx"
#endif

#endif // end #ifndef __elxRegistrationBase_h
