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
#ifndef elxRandomSamplerSparseMask_h
#define elxRandomSamplerSparseMask_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkImageRandomSamplerSparseMask.h"

namespace elastix
{

/**
 * \class RandomSamplerSparseMask
 * \brief An interpolator based on the itk::ImageRandomSamplerSparseMask.
 *
 * This image sampler randomly samples 'NumberOfSamples' voxels in
 * the InputImageRegion. Voxels may be selected multiple times.
 * If a mask is given, the sampler tries to find samples within the
 * mask. If the mask is very sparse, this image sampler is a better
 * choice than the random sampler.
 *
 * \todo Write something similar for the RandomCoordinate sampler.
 *
 * This sampler is suitable to used in combination with the
 * NewSamplesEveryIteration parameter (defined in the elx::OptimizerBase).
 *
 * The parameters used in this class are:
 * \parameter ImageSampler: Select this image sampler as follows:\n
 *    <tt>(ImageSampler "RandomSparseMask")</tt>
 * \parameter NumberOfSpatialSamples: The number of image voxels used for computing the
 *    metric value and its derivative in each iteration. Must be given for each resolution.\n
 *    example: <tt>(NumberOfSpatialSamples 2048 2048 4000)</tt> \n
 *    The default is 5000.
 *
 * \ingroup ImageSamplers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT RandomSamplerSparseMask
  : public itk::ImageRandomSamplerSparseMask<typename elx::ImageSamplerBase<TElastix>::InputImageType>
  , public elx::ImageSamplerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(RandomSamplerSparseMask);

  /** Standard ITK-stuff. */
  using Self = RandomSamplerSparseMask;
  using Superclass1 = itk::ImageRandomSamplerSparseMask<typename elx::ImageSamplerBase<TElastix>::InputImageType>;
  using Superclass2 = elx::ImageSamplerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RandomSamplerSparseMask, itk::ImageRandomSamplerSparseMask);

  /** Name of this class.
   * Use this name in the parameter file to select this specific interpolator. \n
   * example: <tt>(ImageSampler "RandomSparseMask")</tt>\n
   */
  elxClassNameMacro("RandomSparseMask");

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::DataObjectPointer;
  using typename Superclass1::OutputVectorContainerType;
  using typename Superclass1::OutputVectorContainerPointer;
  using typename Superclass1::InputImageType;
  using typename Superclass1::InputImagePointer;
  using typename Superclass1::InputImageConstPointer;
  using typename Superclass1::InputImageRegionType;
  using typename Superclass1::InputImagePixelType;
  using typename Superclass1::ImageSampleType;
  using typename Superclass1::ImageSampleContainerType;
  using typename Superclass1::MaskType;
  using typename Superclass1::InputImageIndexType;
  using typename Superclass1::InputImagePointType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass1::InputImageDimension);

  /** Typedefs inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Execute stuff before each resolution:
   * \li Set the number of samples.
   */
  void
  BeforeEachResolution() override;

protected:
  /** The constructor. */
  RandomSamplerSparseMask() = default;
  /** The destructor. */
  ~RandomSamplerSparseMask() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRandomSamplerSparseMask.hxx"
#endif

#endif // end #ifndef elxRandomSamplerSparseMask_h
