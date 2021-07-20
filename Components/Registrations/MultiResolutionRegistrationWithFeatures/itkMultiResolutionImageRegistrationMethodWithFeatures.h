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
  /** Standard class typedefs. */
  typedef MultiResolutionImageRegistrationMethodWithFeatures                              Self;
  typedef MultiInputMultiResolutionImageRegistrationMethodBase<TFixedImage, TMovingImage> Superclass;
  typedef SmartPointer<Self>                                                              Pointer;
  typedef SmartPointer<const Self>                                                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiResolutionImageRegistrationMethodWithFeatures,
               MultiInputMultiResolutionImageRegistrationMethodBase);

  /**  Superclass types */
  typedef typename Superclass::FixedImageType              FixedImageType;
  typedef typename Superclass::FixedImageConstPointer      FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType        FixedImageRegionType;
  typedef typename Superclass::FixedImageRegionPyramidType FixedImageRegionPyramidType;
  typedef typename Superclass::MovingImageType             MovingImageType;
  typedef typename Superclass::MovingImageConstPointer     MovingImageConstPointer;

  typedef typename Superclass::MetricType                MetricType;
  typedef typename Superclass::MetricPointer             MetricPointer;
  typedef typename Superclass::TransformType             TransformType;
  typedef typename Superclass::TransformPointer          TransformPointer;
  typedef typename Superclass::InterpolatorType          InterpolatorType;
  typedef typename Superclass::InterpolatorPointer       InterpolatorPointer;
  typedef typename Superclass::OptimizerType             OptimizerType;
  typedef typename OptimizerType::Pointer                OptimizerPointer;
  typedef typename Superclass::FixedImagePyramidType     FixedImagePyramidType;
  typedef typename Superclass::FixedImagePyramidPointer  FixedImagePyramidPointer;
  typedef typename Superclass::MovingImagePyramidType    MovingImagePyramidType;
  typedef typename Superclass::MovingImagePyramidPointer MovingImagePyramidPointer;

  typedef typename Superclass::TransformOutputType         TransformOutputType;
  typedef typename Superclass::TransformOutputPointer      TransformOutputPointer;
  typedef typename Superclass::TransformOutputConstPointer TransformOutputConstPointer;

  typedef typename Superclass::ParametersType    ParametersType;
  typedef typename Superclass::DataObjectPointer DataObjectPointer;

protected:
  /** Constructor. */
  MultiResolutionImageRegistrationMethodWithFeatures() = default;

  /** Destructor. */
  ~MultiResolutionImageRegistrationMethodWithFeatures() override = default;

  /** Function called by PreparePyramids, which checks if the user input
   * regarding the image pyramids is ok.
   */
  void
  CheckPyramids(void) override;

private:
  MultiResolutionImageRegistrationMethodWithFeatures(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiResolutionImageRegistrationMethodWithFeatures.hxx"
#endif

#endif // end #ifndef itkMultiResolutionImageRegistrationMethodWithFeatures_h
