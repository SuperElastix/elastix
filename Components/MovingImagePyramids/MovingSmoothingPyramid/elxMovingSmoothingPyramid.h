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
#ifndef elxMovingSmoothingPyramid_h
#define elxMovingSmoothingPyramid_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkMultiResolutionGaussianSmoothingPyramidImageFilter.h"

namespace elastix
{

/**
 * \class MovingSmoothingPyramid
 * \brief A pyramid based on the itkMultiResolutionGaussianSmoothingPyramidImageFilter.
 *
 * The parameters used in this class are:
 * \parameter MovingImagePyramid: Select this pyramid as follows:\n
 *    <tt>(MovingImagePyramid "MovingSmoothingImagePyramid")</tt>
 *
 * \ingroup ImagePyramids
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MovingSmoothingPyramid
  : public itk::MultiResolutionGaussianSmoothingPyramidImageFilter<
      typename MovingImagePyramidBase<TElastix>::InputImageType,
      typename MovingImagePyramidBase<TElastix>::OutputImageType>
  , public MovingImagePyramidBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MovingSmoothingPyramid);

  /** Standard ITK. */
  using Self = MovingSmoothingPyramid;
  using Superclass1 =
    itk::MultiResolutionGaussianSmoothingPyramidImageFilter<typename MovingImagePyramidBase<TElastix>::InputImageType,
                                                            typename MovingImagePyramidBase<TElastix>::OutputImageType>;
  using Superclass2 = MovingImagePyramidBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MovingSmoothingPyramid, MultiResolutionGaussianSmoothingPyramidImageFilter);

  /** Name of this class.
   * Use this name in the parameter file to select this specific pyramid. \n
   * example: <tt>(MovingImagePyramid "MovingSmoothingImagePyramid")</tt>\n
   */
  elxClassNameMacro("MovingSmoothingImagePyramid");

  /** Get the ImageDimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedefs inherited from Superclass1. */
  using typename Superclass1::InputImageType;
  using typename Superclass1::OutputImageType;
  using typename Superclass1::InputImagePointer;
  using typename Superclass1::OutputImagePointer;
  using typename Superclass1::InputImageConstPointer;

  /** Typedefs inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

protected:
  /** The constructor. */
  MovingSmoothingPyramid() = default;
  /** The destructor. */
  ~MovingSmoothingPyramid() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMovingSmoothingPyramid.hxx"
#endif

#endif // end #ifndef elxMovingSmoothingPyramid_h
