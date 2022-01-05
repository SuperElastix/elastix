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
#ifndef itkMultiInputImageRandomCoordinateSampler_h
#define itkMultiInputImageRandomCoordinateSampler_h

#include "itkImageRandomSamplerBase.h"
#include "itkInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

namespace itk
{

/** \class MultiInputImageRandomCoordinateSampler
 *
 * \brief Samples an image by randomly composing a set of physical coordinates
 *
 * This image sampler generates not only samples that correspond with
 * pixel locations, but selects points in physical space.
 *
 * \ingroup ImageSamplers
 */

template <class TInputImage>
class ITK_TEMPLATE_EXPORT MultiInputImageRandomCoordinateSampler : public ImageRandomSamplerBase<TInputImage>
{
public:
  /** Standard ITK-stuff. */
  using Self = MultiInputImageRandomCoordinateSampler;
  using Superclass = ImageRandomSamplerBase<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiInputImageRandomCoordinateSampler, ImageRandomSamplerBase);

  /** Typedefs inherited from the superclass. */
  using typename Superclass::DataObjectPointer;
  using typename Superclass::OutputVectorContainerType;
  using typename Superclass::OutputVectorContainerPointer;
  using typename Superclass::InputImageType;
  using typename Superclass::InputImagePointer;
  using typename Superclass::InputImageConstPointer;
  using typename Superclass::InputImageRegionType;
  using typename Superclass::InputImagePixelType;
  using typename Superclass::ImageSampleType;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::MaskType;
  using typename Superclass::InputImageSizeType;
  using InputImageSpacingType = typename InputImageType::SpacingType;
  using typename Superclass::InputImageIndexType;
  using typename Superclass::InputImagePointType;
  using typename Superclass::InputImagePointValueType;
  using typename Superclass::ImageSampleValueType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass::InputImageDimension);

  /** This image sampler samples the image on physical coordinates and thus
   * needs an interpolator.
   */
  using CoordRepType = double;
  using InterpolatorType = InterpolateImageFunction<InputImageType, CoordRepType>;
  using InterpolatorPointer = typename InterpolatorType::Pointer;
  using DefaultInterpolatorType = BSplineInterpolateImageFunction<InputImageType, CoordRepType, double>;

  /** The random number generator used to generate random coordinates. */
  using RandomGeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
  using RandomGeneratorPointer = typename RandomGeneratorType::Pointer;

  /** Set/Get the interpolator. A 3rd order B-spline interpolator is used by default. */
  itkSetObjectMacro(Interpolator, InterpolatorType);
  itkGetModifiableObjectMacro(Interpolator, InterpolatorType);

  /** Set/Get the sample region size (in mm). Only needed when UseRandomSampleRegion==true;
   * default: filled with ones.
   */
  itkSetMacro(SampleRegionSize, InputImageSpacingType);
  itkGetConstReferenceMacro(SampleRegionSize, InputImageSpacingType);

  /** Set/Get whether to use randomly selected sample regions, or just the whole image
   * Default: false. */
  itkGetConstMacro(UseRandomSampleRegion, bool);
  itkSetMacro(UseRandomSampleRegion, bool);

protected:
  using InputImageContinuousIndexType = typename InterpolatorType::ContinuousIndexType;

  /** The constructor. */
  MultiInputImageRandomCoordinateSampler();

  /** The destructor. */
  ~MultiInputImageRandomCoordinateSampler() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Function that does the work. */
  void
  GenerateData() override;

  /** Generate a point randomly in a bounding box.
   * This method can be overwritten in subclasses if a different distribution is desired. */
  virtual void
  GenerateRandomCoordinate(const InputImageContinuousIndexType & smallestContIndex,
                           const InputImageContinuousIndexType & largestContIndex,
                           InputImageContinuousIndexType &       randomContIndex);

  InterpolatorPointer    m_Interpolator;
  RandomGeneratorPointer m_RandomGenerator;
  InputImageSpacingType  m_SampleRegionSize;

  /** Generate the two corners of a sampling region. */
  virtual void
  GenerateSampleRegion(InputImageContinuousIndexType & smallestContIndex,
                       InputImageContinuousIndexType & largestContIndex);

private:
  /** The deleted copy constructor. */
  MultiInputImageRandomCoordinateSampler(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

  bool m_UseRandomSampleRegion;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiInputImageRandomCoordinateSampler.hxx"
#endif

#endif // end #ifndef itkMultiInputImageRandomCoordinateSampler_h
