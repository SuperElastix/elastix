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
#ifndef elxFixedImagePyramidBase_h
#define elxFixedImagePyramidBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkObject.h"
#include "itkMultiResolutionPyramidImageFilter.h"

namespace elastix
{

/**
 * \class FixedImagePyramidBase
 * \brief This class is the elastix base class for all FixedImagePyramids.
 *
 * This class contains all the common functionality for FixedImagePyramids.
 *
 * \parameter FixedImagePyramidSchedule: downsampling factors for the fixed image pyramid.\n
 *    For each dimension, for each resolution level, the downsampling factor of the
 *    fixed image can be specified.\n
 *    Syntax for 2D images:\n
 *    <tt>(FixedImagePyramidSchedule <reslevel0,dim0> <reslevel0,dim1> <reslevel1,dim0> <reslevel1,dim1> ...)</tt>\n
 *    example: <tt>(FixedImagePyramidSchedule 4 4 2 2 1 1)</tt>\n
 *    Default: isotropic, halved in each resolution, so, like in the example. If
 *    ImagePyramidSchedule is specified, that schedule is used for both fixed and moving image pyramid.
 * \parameter ImagePyramidSchedule: downsampling factors for fixed and moving image pyramids.\n
 *    example: <tt>(ImagePyramidSchedule 4 4 2 2 1 1)</tt> \n
 *    Used as a default when FixedImagePyramidSchedule is not specified. If both are omitted,
 *    a default schedule is assumed: isotropic, halved in each resolution, so, like in the example.
 * \parameter WritePyramidImagesAfterEachResolution: ...\n
 *    example: <tt>(WritePyramidImagesAfterEachResolution "true")</tt>\n
 *    default "false".
 *
 * \ingroup ImagePyramids
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT FixedImagePyramidBase : public BaseComponentSE<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef FixedImagePyramidBase     Self;
  typedef BaseComponentSE<TElastix> Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro(FixedImagePyramidBase, BaseComponentSE);

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::ElastixType          ElastixType;
  typedef typename Superclass::ElastixPointer       ElastixPointer;
  typedef typename Superclass::ConfigurationType    ConfigurationType;
  typedef typename Superclass::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass::RegistrationType     RegistrationType;
  typedef typename Superclass::RegistrationPointer  RegistrationPointer;

  /** Typedefs inherited from Elastix. */
  typedef typename ElastixType::FixedImageType InputImageType;
  typedef typename ElastixType::FixedImageType OutputImageType;

  /** Other typedef's. */
  typedef itk::MultiResolutionPyramidImageFilter<InputImageType, OutputImageType> ITKBaseType;

  /** Typedef's from ITKBaseType. */
  typedef typename ITKBaseType::ScheduleType ScheduleType;

  /** Retrieves this object as ITKBaseType. */
  ITKBaseType *
  GetAsITKBaseType(void)
  {
    return &(this->GetSelf());
  }


  /** Retrieves this object as ITKBaseType, to use in const functions. */
  const ITKBaseType *
  GetAsITKBaseType(void) const
  {
    return &(this->GetSelf());
  }


  /** Execute stuff before the actual registration:
   * \li Set the schedule of the fixed image pyramid.
   */
  void
  BeforeRegistrationBase(void) override;

  /** Execute stuff before each resolution:
   * \li Write the pyramid image to file.
   */
  void
  BeforeEachResolutionBase(void) override;

  /** Method for setting the schedule. */
  virtual void
  SetFixedSchedule(void);

  /** Method to write the pyramid image. */
  virtual void
  WritePyramidImage(const std::string &  filename,
                    const unsigned int & level); // const;

protected:
  /** The constructor. */
  FixedImagePyramidBase() = default;
  /** The destructor. */
  ~FixedImagePyramidBase() override = default;

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);

  /** The deleted copy constructor. */
  FixedImagePyramidBase(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxFixedImagePyramidBase.hxx"
#endif

#endif // end #ifndef elxFixedImagePyramidBase_h
