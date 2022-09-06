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
#ifndef elxFixedSmoothingPyramid_h
#define elxFixedSmoothingPyramid_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkMultiResolutionGaussianSmoothingPyramidImageFilter.h"

namespace elastix
{

/**
 * \class FixedSmoothingPyramid
 * \brief A pyramid based on the itk::MultiResolutionGaussianSmoothingPyramidImageFilter.
 *
 * The parameters used in this class are:
 * \parameter FixedImagePyramid: Select this pyramid as follows:\n
 *    <tt>(FixedImagePyramid "FixedSmoothingImagePyramid")</tt>
 *
 * \ingroup ImagePyramids
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT FixedSmoothingPyramid
  : public itk::MultiResolutionGaussianSmoothingPyramidImageFilter<
      typename FixedImagePyramidBase<TElastix>::InputImageType,
      typename FixedImagePyramidBase<TElastix>::OutputImageType>
  , public FixedImagePyramidBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FixedSmoothingPyramid);

  /** Standard ITK-stuff. */
  using Self = FixedSmoothingPyramid;
  using Superclass1 =
    itk::MultiResolutionGaussianSmoothingPyramidImageFilter<typename FixedImagePyramidBase<TElastix>::InputImageType,
                                                            typename FixedImagePyramidBase<TElastix>::OutputImageType>;
  using Superclass2 = FixedImagePyramidBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FixedSmoothingPyramid, itk::MultiResolutionGaussianSmoothingPyramidImageFilter);

  /** Name of this class.
   * Use this name in the parameter file to select this specific pyramid. \n
   * example: <tt>(FixedImagePyramid "FixedSmoothingImagePyramid")</tt>\n
   */
  elxClassNameMacro("FixedSmoothingImagePyramid");

  /** Get the ImageDimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::InputImageType;
  using typename Superclass1::OutputImageType;
  using typename Superclass1::InputImagePointer;
  using typename Superclass1::OutputImagePointer;
  using typename Superclass1::InputImageConstPointer;
  using typename Superclass1::ScheduleType;

  /** Typedefs inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

protected:
  /** The constructor. */
  FixedSmoothingPyramid() = default;
  /** The destructor. */
  ~FixedSmoothingPyramid() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxFixedSmoothingPyramid.hxx"
#endif

#endif // end #ifndef elxFixedSmoothingPyramid_h
