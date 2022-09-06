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
  ITK_DISALLOW_COPY_AND_MOVE(NearestNeighborResampleInterpolator);

  /** Standard ITK-stuff. */
  using Self = NearestNeighborResampleInterpolator;
  using Superclass1 =
    itk::NearestNeighborInterpolateImageFunction<typename ResampleInterpolatorBase<TElastix>::InputImageType,
                                                 typename ResampleInterpolatorBase<TElastix>::CoordRepType>;
  using Superclass2 = ResampleInterpolatorBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

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
  using typename Superclass1::OutputType;
  using typename Superclass1::InputImageType;
  using typename Superclass1::IndexType;
  using typename Superclass1::ContinuousIndexType;

  /** Typedef's from ResampleInterpolatorBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

protected:
  /** The constructor. */
  NearestNeighborResampleInterpolator() = default;
  /** The destructor. */
  ~NearestNeighborResampleInterpolator() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxNearestNeighborResampleInterpolator.hxx"
#endif

#endif // end elxNearestNeighborResampleInterpolator_h
