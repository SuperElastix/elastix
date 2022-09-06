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
#ifndef elxBSplineResampleInterpolatorFloat_h
#define elxBSplineResampleInterpolatorFloat_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkBSplineInterpolateImageFunction.h"

namespace elastix
{

/**
 * \class BSplineResampleInterpolatorFloat
 * \brief A resample-interpolator based on B-splines.
 *
 * Compared to the BSplineResampleInterpolator this class uses
 * a float CoefficientType, instead of double. You can select
 * this resample interpolator if memory burden is an issue.
 *
 * The parameters used in this class are:
 * \parameter ResampleInterpolator: Select this resample interpolator as follows:\n
 *   <tt>(ResampleInterpolator "FinalBSplineInterpolatorFloat")</tt>
 * \parameter FinalBSplineInterpolationOrder: the order of the B-spline used to resample
 *    the deformed moving image; possible values: (0-5) \n
 *    example: <tt>(FinalBSplineInterpolationOrder 3 ) </tt> \n
 *    Default: 3.
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter FinalBSplineInterpolationOrder: the order of the B-spline used to resample
 *    the deformed moving image; possible values: (0-5) \n
 *    example: <tt>(FinalBSplineInterpolationOrder 3) </tt> \n
 *    Default: 3.
 *
 * \ingroup ResampleInterpolators
 * \sa BSplineResampleInterpolator
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT BSplineResampleInterpolatorFloat
  : public itk::BSplineInterpolateImageFunction<typename ResampleInterpolatorBase<TElastix>::InputImageType,
                                                typename ResampleInterpolatorBase<TElastix>::CoordRepType,
                                                float>
  , // CoefficientType
    public ResampleInterpolatorBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineResampleInterpolatorFloat);

  /** Standard ITK-stuff. */
  using Self = BSplineResampleInterpolatorFloat;
  using Superclass1 = itk::BSplineInterpolateImageFunction<typename ResampleInterpolatorBase<TElastix>::InputImageType,
                                                           typename ResampleInterpolatorBase<TElastix>::CoordRepType,
                                                           float>;
  using Superclass2 = ResampleInterpolatorBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineResampleInterpolatorFloat, BSplineInterpolateImageFunction);

  /** Name of this class.
   * Use this name in the parameter file to select this specific resample interpolator. \n
   * example: <tt>(ResampleInterpolator "FinalBSplineInterpolatorFloat")</tt>\n
   */
  elxClassNameMacro("FinalBSplineInterpolatorFloat");

  /** Dimension of the image. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedef's inherited from the superclass. */
  using typename Superclass1::OutputType;
  using typename Superclass1::InputImageType;
  using typename Superclass1::IndexType;
  using typename Superclass1::ContinuousIndexType;
  using typename Superclass1::PointType;
  using typename Superclass1::Iterator;
  using typename Superclass1::CoefficientDataType;
  using typename Superclass1::CoefficientImageType;
  using typename Superclass1::CoefficientFilter;
  using typename Superclass1::CoefficientFilterPointer;
  using typename Superclass1::CovariantVectorType;

  /** Typedef's from ResampleInterpolatorBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using typename Superclass2::ParameterMapType;

  /** Execute stuff before the actual registration:
   * \li Set the spline order.
   */
  void
  BeforeRegistration() override;

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile() override;

protected:
  /** The constructor. */
  BSplineResampleInterpolatorFloat() = default;
  /** The destructor. */
  ~BSplineResampleInterpolatorFloat() override = default;

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) interpolator type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxBSplineResampleInterpolatorFloat.hxx"
#endif

#endif // end elxBSplineResampleInterpolatorFloat_h
