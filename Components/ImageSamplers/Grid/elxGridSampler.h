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
#ifndef elxGridSampler_h
#define elxGridSampler_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkImageGridSampler.h"

namespace elastix
{

/**
 * \class GridSampler
 * \brief An interpolator based on the itk::ImageGridSampler.
 *
 * This image sampler samples voxels on a uniform grid.
 * This sampler is in most cases not recommended.
 *
 * This sampler does not react on the
 * NewSamplesEveryIteration parameter.
 *
 * The parameters used in this class are:
 * \parameter ImageSampler: Select this image sampler as follows:\n
 *    <tt>(ImageSampler "Grid")</tt>
 * \parameter SampleGridSpacing: Defines the sampling grid in case of a Grid ImageSampler.\n
 *    An integer downsampling factor must be specified for each dimension, for each resolution.\n
 *    example: <tt>(SampleGridSpacing 4 4 2 2)</tt>\n
 *    Default is 2 for each dimension for each resolution.
 *
 * \ingroup ImageSamplers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT GridSampler
  : public itk::ImageGridSampler<typename elx::ImageSamplerBase<TElastix>::InputImageType>
  , public elx::ImageSamplerBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef GridSampler                                                                     Self;
  typedef itk::ImageGridSampler<typename elx::ImageSamplerBase<TElastix>::InputImageType> Superclass1;
  typedef elx::ImageSamplerBase<TElastix>                                                 Superclass2;
  typedef itk::SmartPointer<Self>                                                         Pointer;
  typedef itk::SmartPointer<const Self>                                                   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GridSampler, itk::ImageGridSampler);

  /** Name of this class.
   * Use this name in the parameter file to select this specific interpolator. \n
   * example: <tt>(ImageSampler "Grid")</tt>\n
   */
  elxClassNameMacro("Grid");

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
  typedef typename Superclass1::SampleGridSpacingType        GridSpacingType;
  typedef typename Superclass1::SampleGridSpacingValueType   SampleGridSpacingValueType;

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
   * \li Set the sampling grid size.
   */
  void
  BeforeEachResolution(void) override;

protected:
  /** The constructor. */
  GridSampler() = default;
  /** The destructor. */
  ~GridSampler() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  GridSampler(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxGridSampler.hxx"
#endif

#endif // end #ifndef elxGridSampler_h
