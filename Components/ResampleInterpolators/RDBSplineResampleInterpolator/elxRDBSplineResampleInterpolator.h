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
#ifndef __elxReducedDimensionBSplineResampleInterpolator_h
#define __elxReducedDimensionBSplineResampleInterpolator_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkReducedDimensionBSplineInterpolateImageFunction.h"

namespace elastix
{

/**
 * \class ReducedDimensionBSplineResampleInterpolator
 * \brief A resample-interpolator based on B-splines which ignores the last dimension.
 *
 * The parameters used in this class are:
 * \parameter ResampleInterpolator: Select this resample interpolator as follows:\n
 *   <tt>(ResampleInterpolator "FinalReducedDimensionBSplineInterpolator")</tt>
 * \parameter FinalReducedDimensionBSplineInterpolationOrder: the order of the B-spline used to resample
 *    the deformed moving image; possible values: (0-5) \n
 *    example: <tt>(FinalReducedDimensionBSplineInterpolationOrder 3) </tt> \n
 *    Default: 3.
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter FinalReducedDimensionBSplineInterpolationOrder: the order of the B-spline used to resample
 *    the deformed moving image; possible values: (0-5) \n
 *    example: <tt>(FinalReducedDimensionBSplineInterpolationOrder 3) </tt> \n
 *    Default: 3.
 *
 * If you are in memory problems, you may use the LinearResampleInterpolator,
 * or the NearestNeighborResampleInterpolator. Note that the former will also
 * interpolate in the last dimension.
 *
 * \ingroup ResampleInterpolators
 * \sa ReducedDimensionBSplineResampleInterpolatorFloat
 */

template <class TElastix>
class ReducedDimensionBSplineResampleInterpolator
  : public itk::ReducedDimensionBSplineInterpolateImageFunction<
      typename ResampleInterpolatorBase<TElastix>::InputImageType,
      typename ResampleInterpolatorBase<TElastix>::CoordRepType,
      double>
  , // CoefficientType
    public ResampleInterpolatorBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef ReducedDimensionBSplineResampleInterpolator Self;
  typedef itk::BSplineInterpolateImageFunction<typename ResampleInterpolatorBase<TElastix>::InputImageType,
                                               typename ResampleInterpolatorBase<TElastix>::CoordRepType,
                                               double>
                                             Superclass1;
  typedef ResampleInterpolatorBase<TElastix> Superclass2;
  typedef itk::SmartPointer<Self>            Pointer;
  typedef itk::SmartPointer<const Self>      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ReducedDimensionBSplineResampleInterpolator, itk::ReducedDimensionBSplineInterpolateImageFunction);

  /** Name of this class.
   * Use this name in the parameter file to select this specific resample interpolator. \n
   * example: <tt>(ResampleInterpolator "FinalBSplineInterpolator")</tt>\n
   */
  elxClassNameMacro("FinalReducedDimensionBSplineInterpolator");

  /** Dimension of the image. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedef's inherited from the superclass. */
  typedef typename Superclass1::OutputType               OutputType;
  typedef typename Superclass1::InputImageType           InputImageType;
  typedef typename Superclass1::IndexType                IndexType;
  typedef typename Superclass1::ContinuousIndexType      ContinuousIndexType;
  typedef typename Superclass1::PointType                PointType;
  typedef typename Superclass1::Iterator                 Iterator;
  typedef typename Superclass1::CoefficientDataType      CoefficientDataType;
  typedef typename Superclass1::CoefficientImageType     CoefficientImageType;
  typedef typename Superclass1::CoefficientFilter        CoefficientFilter;
  typedef typename Superclass1::CoefficientFilterPointer CoefficientFilterPointer;
  typedef typename Superclass1::CovariantVectorType      CovariantVectorType;

  /** Typedef's from ResampleInterpolatorBase. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Execute stuff before the actual registration:
   * \li Set the spline order.
   */
  void
  BeforeRegistration(void) override;

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile(void) override;

  /** Function to write transform-parameters to a file. */
  void
  WriteToFile(void) const override;

protected:
  /** The constructor. */
  ReducedDimensionBSplineResampleInterpolator() {}
  /** The destructor. */
  ~ReducedDimensionBSplineResampleInterpolator() override {}

private:
  /** The private constructor. */
  ReducedDimensionBSplineResampleInterpolator(const Self &); // purposely not implemented
  /** The private copy constructor. */
  void
  operator=(const Self &); // purposely not implemented
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRDBSplineResampleInterpolator.hxx"
#endif

#endif // end __elxReducedDimensionBSplineResampleInterpolator_h
