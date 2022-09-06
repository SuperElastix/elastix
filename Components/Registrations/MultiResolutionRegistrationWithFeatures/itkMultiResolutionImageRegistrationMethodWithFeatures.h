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
#ifndef itkMultiResolutionImageRegistrationMethodWithFeatures_h
#define itkMultiResolutionImageRegistrationMethodWithFeatures_h

#include "itkMultiInputMultiResolutionImageRegistrationMethodBase.h"

namespace itk
{

/** \class MultiResolutionImageRegistrationMethodWithFeatures
 * \brief Class for multi-resolution image registration methods
 *
 * This class is an extension of the itk class
 * MultiResolutionImageRegistrationMethod. It allows the use
 * of multiple metrics, which are summed, multiple images,
 * multiple interpolators, and/or multiple image pyramids.
 *
 * Make sure the following is true:\n
 *   nrofmetrics >= nrofinterpolators >= nrofmovingpyramids >= nrofmovingimages\n
 *   nrofmetrics >= nroffixedpyramids >= nroffixedimages\n
 *   nroffixedregions == nroffixedimages\n
 *
 *   nrofinterpolators == nrofmetrics OR nrofinterpolators == 1\n
 *   nroffixedimages == nrofmetrics OR nroffixedimages == 1\n
 *   etc...
 *
 * You may also set an interpolator/fixedimage/etc to NULL, if you
 * happen to know that the corresponding metric is not an
 * ImageToImageMetric, but a regularizer for example (which does
 * not need an image.
 *
 *
 * \sa ImageRegistrationMethod
 * \sa MultiResolutionImageRegistrationMethod
 * \ingroup RegistrationFilters
 */

template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT MultiResolutionImageRegistrationMethodWithFeatures
  : public MultiInputMultiResolutionImageRegistrationMethodBase<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiResolutionImageRegistrationMethodWithFeatures);

  /** Standard class typedefs. */
  using Self = MultiResolutionImageRegistrationMethodWithFeatures;
  using Superclass = MultiInputMultiResolutionImageRegistrationMethodBase<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiResolutionImageRegistrationMethodWithFeatures,
               MultiInputMultiResolutionImageRegistrationMethodBase);

  /**  Superclass types */
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::FixedImageRegionPyramidType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImageConstPointer;

  using typename Superclass::MetricType;
  using typename Superclass::MetricPointer;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::OptimizerType;
  using OptimizerPointer = typename OptimizerType::Pointer;
  using typename Superclass::FixedImagePyramidType;
  using typename Superclass::FixedImagePyramidPointer;
  using typename Superclass::MovingImagePyramidType;
  using typename Superclass::MovingImagePyramidPointer;

  using typename Superclass::TransformOutputType;
  using typename Superclass::TransformOutputPointer;
  using typename Superclass::TransformOutputConstPointer;

  using typename Superclass::ParametersType;
  using typename Superclass::DataObjectPointer;

protected:
  /** Constructor. */
  MultiResolutionImageRegistrationMethodWithFeatures() = default;

  /** Destructor. */
  ~MultiResolutionImageRegistrationMethodWithFeatures() override = default;

  /** Function called by PreparePyramids, which checks if the user input
   * regarding the image pyramids is ok.
   */
  void
  CheckPyramids() override;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiResolutionImageRegistrationMethodWithFeatures.hxx"
#endif

#endif // end #ifndef itkMultiResolutionImageRegistrationMethodWithFeatures_h
