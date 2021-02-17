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
  /** Standard ITK-stuff. */
  typedef RandomSamplerSparseMask                                                                     Self;
  typedef itk::ImageRandomSamplerSparseMask<typename elx::ImageSamplerBase<TElastix>::InputImageType> Superclass1;
  typedef elx::ImageSamplerBase<TElastix>                                                             Superclass2;
  typedef itk::SmartPointer<Self>                                                                     Pointer;
  typedef itk::SmartPointer<const Self>                                                               ConstPointer;

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
  typedef typename Superclass1::DataObjectPointer            DataObjectPointer;
  typedef typename Superclass1::OutputVectorContainerType    OutputVectorContainerType;
  typedef typename Superclass1::OutputVectorContainerPointer OutputVectorContainerPointer;
  typedef typename Superclass1::InputImageType               InputImageType;
  typedef typename Superclass1::InputImagePointer            InputImagePointer;
  typedef typename Superclass1::InputImageConstPointer       InputImageConstPointer;
  typedef typename Superclass1::InputImageRegionType         InputImageRegionType;
  typedef typename Superclass1::InputImagePixelType          InputImagePixelType;
  typedef typename Superclass1::ImageSampleType              ImageSampleType;
  typedef typename Superclass1::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass1::MaskType                     MaskType;
  typedef typename Superclass1::InputImageIndexType          InputImageIndexType;
  typedef typename Superclass1::InputImagePointType          InputImagePointType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass1::InputImageDimension);

  /** Typedefs inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Execute stuff before each resolution:
   * \li Set the number of samples.
   */
  void
  BeforeEachResolution(void) override;

protected:
  /** The constructor. */
  RandomSamplerSparseMask() = default;
  /** The destructor. */
  ~RandomSamplerSparseMask() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  RandomSamplerSparseMask(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRandomSamplerSparseMask.hxx"
#endif

#endif // end #ifndef elxRandomSamplerSparseMask_h
