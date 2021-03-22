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
#ifndef elxMovingRecursivePyramid_h
#define elxMovingRecursivePyramid_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkRecursiveMultiResolutionPyramidImageFilter.h"

namespace elastix
{

/**
 * \class MovingRecursivePyramid
 * \brief A pyramid based on the itkRecursiveMultiResolutionPyramidImageFilter.
 *
 * The parameters used in this class are:
 * \parameter MovingImagePyramid: Select this pyramid as follows:\n
 *    <tt>(MovingImagePyramid "MovingRecursiveImagePyramid")</tt>
 *
 * \ingroup ImagePyramids
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MovingRecursivePyramid
  : public itk::RecursiveMultiResolutionPyramidImageFilter<typename MovingImagePyramidBase<TElastix>::InputImageType,
                                                           typename MovingImagePyramidBase<TElastix>::OutputImageType>
  , public MovingImagePyramidBase<TElastix>
{
public:
  /** Standard ITK. */
  typedef MovingRecursivePyramid Self;
  typedef itk::RecursiveMultiResolutionPyramidImageFilter<typename MovingImagePyramidBase<TElastix>::InputImageType,
                                                          typename MovingImagePyramidBase<TElastix>::OutputImageType>
                                           Superclass1;
  typedef MovingImagePyramidBase<TElastix> Superclass2;
  typedef itk::SmartPointer<Self>          Pointer;
  typedef itk::SmartPointer<const Self>    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MovingRecursivePyramid, RecursiveMultiResolutionPyramidImageFilter);

  /** Name of this class.
   * Use this name in the parameter file to select this specific pyramid. \n
   * example: <tt>(MovingImagePyramid "MovingRecursiveImagePyramid")</tt>\n
   */
  elxClassNameMacro("MovingRecursiveImagePyramid");

  /** Get the ImageDimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedefs inherited from Superclass1. */
  typedef typename Superclass1::InputImageType         InputImageType;
  typedef typename Superclass1::OutputImageType        OutputImageType;
  typedef typename Superclass1::InputImagePointer      InputImagePointer;
  typedef typename Superclass1::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass1::InputImageConstPointer InputImageConstPointer;

  /** Typedefs inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

protected:
  /** The constructor. */
  MovingRecursivePyramid() = default;
  /** The destructor. */
  ~MovingRecursivePyramid() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  MovingRecursivePyramid(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMovingRecursivePyramid.hxx"
#endif

#endif // end #ifndef elxMovingRecursivePyramid_h
