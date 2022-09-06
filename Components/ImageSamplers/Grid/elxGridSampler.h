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
  ITK_DISALLOW_COPY_AND_MOVE(GridSampler);

  /** Standard ITK-stuff. */
  using Self = GridSampler;
  using Superclass1 = itk::ImageGridSampler<typename elx::ImageSamplerBase<TElastix>::InputImageType>;
  using Superclass2 = elx::ImageSamplerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

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
  using GridSpacingType = typename Superclass1::SampleGridSpacingType;
  using typename Superclass1::SampleGridSpacingValueType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass1::InputImageDimension);

  /** Typedefs inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Execute stuff before each resolution:
   * \li Set the sampling grid size.
   */
  void
  BeforeEachResolution() override;

protected:
  /** The constructor. */
  GridSampler() = default;
  /** The destructor. */
  ~GridSampler() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxGridSampler.hxx"
#endif

#endif // end #ifndef elxGridSampler_h
