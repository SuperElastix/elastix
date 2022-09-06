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
  ITK_DISALLOW_COPY_AND_MOVE(MovingRecursivePyramid);

  /** Standard ITK. */
  using Self = MovingRecursivePyramid;
  using Superclass1 =
    itk::RecursiveMultiResolutionPyramidImageFilter<typename MovingImagePyramidBase<TElastix>::InputImageType,
                                                    typename MovingImagePyramidBase<TElastix>::OutputImageType>;
  using Superclass2 = MovingImagePyramidBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

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
  MovingRecursivePyramid() = default;
  /** The destructor. */
  ~MovingRecursivePyramid() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMovingRecursivePyramid.hxx"
#endif

#endif // end #ifndef elxMovingRecursivePyramid_h
