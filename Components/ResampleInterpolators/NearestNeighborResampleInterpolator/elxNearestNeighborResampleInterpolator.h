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
#ifndef elxNearestNeighborResampleInterpolator_h
#define elxNearestNeighborResampleInterpolator_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkNearestNeighborInterpolateImageFunction.h"

namespace elastix
{

/**
 * \class NearestNeighborResampleInterpolator
 * \brief A nearest neighbor resample-interpolator.
 *
 * Compared to the BSplineResampleInterpolator and BSplineResampleInterpolatorFloat
 * with SplineOrder 0 this class uses less (in fact, no) memory. You can select
 * this resample interpolator if memory burden is an issue and nearest neighbor interpolation
 * is sufficient.
 *
 * The parameters used in this class are:
 * \parameter ResampleInterpolator: Select this resample interpolator as follows:\n
 *   <tt>(ResampleInterpolator "FinalNearestNeighborInterpolator")</tt>
 *
 * \ingroup ResampleInterpolators
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT NearestNeighborResampleInterpolator
  : public itk::NearestNeighborInterpolateImageFunction<typename ResampleInterpolatorBase<TElastix>::InputImageType,
                                                        typename ResampleInterpolatorBase<TElastix>::CoordRepType>
  , public ResampleInterpolatorBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef NearestNeighborResampleInterpolator Self;
  typedef itk::NearestNeighborInterpolateImageFunction<typename ResampleInterpolatorBase<TElastix>::InputImageType,
                                                       typename ResampleInterpolatorBase<TElastix>::CoordRepType>
                                             Superclass1;
  typedef ResampleInterpolatorBase<TElastix> Superclass2;
  typedef itk::SmartPointer<Self>            Pointer;
  typedef itk::SmartPointer<const Self>      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NearestNeighborResampleInterpolator, NearestNeighborInterpolateImageFunction);

  /** Name of this class.
   * Use this name in the parameter file to select this specific resample interpolator. \n
   * example: <tt>(ResampleInterpolator "FinalNearestNeighborInterpolator")</tt>\n
   */
  elxClassNameMacro("FinalNearestNeighborInterpolator");

  /** Dimension of the image. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedef's inherited from the superclass. */
  typedef typename Superclass1::OutputType          OutputType;
  typedef typename Superclass1::InputImageType      InputImageType;
  typedef typename Superclass1::IndexType           IndexType;
  typedef typename Superclass1::ContinuousIndexType ContinuousIndexType;

  /** Typedef's from ResampleInterpolatorBase. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

protected:
  /** The constructor. */
  NearestNeighborResampleInterpolator() = default;
  /** The destructor. */
  ~NearestNeighborResampleInterpolator() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  NearestNeighborResampleInterpolator(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxNearestNeighborResampleInterpolator.hxx"
#endif

#endif // end elxNearestNeighborResampleInterpolator_h
