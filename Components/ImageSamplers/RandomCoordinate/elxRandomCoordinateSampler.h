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
#ifndef elxRandomCoordinateSampler_h
#define elxRandomCoordinateSampler_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkImageRandomCoordinateSampler.h"

namespace elastix
{

/**
 * \class RandomCoordinateSampler
 * \brief An interpolator based on the itk::ImageRandomCoordinateSampler.
 *
 * This image sampler randomly samples 'NumberOfSamples' coordinates in
 * the InputImageRegion. If a mask is given, the sampler tries to find
 * samples within the mask. If the mask is very sparse, this may take some time.
 * The RandomCoordinate sampler samples not only positions that correspond
 * to voxels, but also positions between voxels. An interpolator for the fixed image is thus
 * required. A B-spline interpolator is used, the order of which can be specified
 * by the user. Typically, the RandomCoordinate gives a smoother cost function,
 * because the so-called 'grid-effect' is avoided.
 *
 * This sampler is suitable to used in combination with the
 * NewSamplesEveryIteration parameter (defined in the elx::OptimizerBase).
 *
 * The parameters used in this class are:
 * \parameter ImageSampler: Select this image sampler as follows:\n
 *    <tt>(ImageSampler "RandomCoordinate")</tt>
 * \parameter NumberOfSpatialSamples: The number of image voxels used for computing the
 *    metric value and its derivative in each iteration. Must be given for each resolution.\n
 *    example: <tt>(NumberOfSpatialSamples 2048 2048 4000)</tt> \n
 *    The default is 5000.
 * \parameter UseRandomSampleRegion: Defines whether to randomly select a subregion of the image
 *    in each iteration. When set to "true", also specify the SampleRegionSize.
 *    By setting this option to "true", in combination with the NewSamplesEveryIteration parameter,
 *    a "localised" similarity measure is obtained. This can give better performance in case
 *    of the presence of large inhomogeneities in the image, for example.\n
 *    example: <tt>(UseRandomSampleRegion "true")</tt>\n
 *    Default: false.
 * \parameter SampleRegionSize: the size of the subregions that are selected when using
 *    the UseRandomSampleRegion option. The size should be specified in mm, for each dimension.
 *    As a rule of thumb, you may try a value ~1/3 of the image size.\n
 *    example: <tt>(SampleRegionSize 50.0 50.0 50.0)</tt>\n
 *    You can also specify one number, which will be used for all dimensions. Also, you
 *    can specify different values for each resolution:\n
 *    example: <tt>(SampleRegionSize 50.0 50.0 50.0 30.0 30.0 30.0)</tt>\n
 *    In this example, in the first resolution 50mm is used for each of the 3 dimensions,
 *    and in the second resolution 30mm.\n
 *    Default: sampleRegionSize[i] = min ( fixedImageSize[i], max_i ( fixedImageSize[i]/3 ) ),
 *    with fixedImageSize in mm. So, approximately 1/3 of the fixed image size.
 * \parameter FixedImageBSplineInterpolationOrder: When using a RandomCoordinate sampler,
 *    the fixed image needs to be interpolated. This is done using a B-spline interpolator.
 *    With this option you can specify the order of interpolation.\n
 *    example: <tt>(FixedImageBSplineInterpolationOrder 0 0 1)</tt>\n
 *    Default value: 1. The parameter can be specified for each resolution.
 *
 * \ingroup ImageSamplers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT RandomCoordinateSampler
  : public itk::ImageRandomCoordinateSampler<typename elx::ImageSamplerBase<TElastix>::InputImageType>
  , public elx::ImageSamplerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(RandomCoordinateSampler);

  /** Standard ITK-stuff. */
  using Self = RandomCoordinateSampler;
  using Superclass1 = itk::ImageRandomCoordinateSampler<typename elx::ImageSamplerBase<TElastix>::InputImageType>;
  using Superclass2 = elx::ImageSamplerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RandomCoordinateSampler, ImageRandomCoordinateSampler);

  /** Name of this class.
   * Use this name in the parameter file to select this specific interpolator. \n
   * example: <tt>(ImageSampler "RandomCoordinate")</tt>\n
   */
  elxClassNameMacro("RandomCoordinate");

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
  using typename Superclass1::InputImageSizeType;
  using typename Superclass1::InputImageSpacingType;
  using typename Superclass1::InputImagePointValueType;
  using typename Superclass1::ImageSampleValueType;

  /** This image sampler samples the image on physical coordinates and thus
   * needs an interpolator. */
  using typename Superclass1::CoordRepType;
  using typename Superclass1::InterpolatorType;
  using typename Superclass1::DefaultInterpolatorType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass1::InputImageDimension);

  /** Typedefs inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Execute stuff before each resolution:
   * \li Set the number of samples.
   * \li Set the fixed image interpolation order
   * \li Set the UseRandomSampleRegion flag and the SampleRegionSize
   */
  void
  BeforeEachResolution() override;

protected:
  /** The constructor. */
  RandomCoordinateSampler() = default;
  /** The destructor. */
  ~RandomCoordinateSampler() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRandomCoordinateSampler.hxx"
#endif

#endif // end #ifndef elxRandomCoordinateSampler_h
