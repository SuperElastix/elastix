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
#ifndef itkImageRandomCoordinateSampler_h
#define itkImageRandomCoordinateSampler_h

#include "itkImageRandomSamplerBase.h"
#include "itkInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

namespace itk
{

/** \class ImageRandomCoordinateSampler
 *
 * \brief Samples an image by randomly composing a set of physical coordinates
 *
 * This image sampler generates not only samples that correspond with
 * pixel locations, but selects points in physical space.
 *
 * \ingroup ImageSamplers
 */

template <class TInputImage>
class ITK_TEMPLATE_EXPORT ImageRandomCoordinateSampler : public ImageRandomSamplerBase<TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageRandomCoordinateSampler);

  /** Standard ITK-stuff. */
  using Self = ImageRandomCoordinateSampler;
  using Superclass = ImageRandomSamplerBase<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageRandomCoordinateSampler, ImageRandomSamplerBase);

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
  using typename Superclass::ImageSampleContainerPointer;
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
   * needs an interpolator. */
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
   * default: filled with ones.  */
  itkSetMacro(SampleRegionSize, InputImageSpacingType);
  itkGetConstReferenceMacro(SampleRegionSize, InputImageSpacingType);

  /** Set/Get whether to use randomly selected sample regions, or just the whole image
   * Default: false. */
  itkGetConstMacro(UseRandomSampleRegion, bool);
  itkSetMacro(UseRandomSampleRegion, bool);

protected:
  using InputImageContinuousIndexType = typename InterpolatorType::ContinuousIndexType;

  /** The constructor. */
  ImageRandomCoordinateSampler() = default;

  /** The destructor. */
  ~ImageRandomCoordinateSampler() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Function that does the work. */
  void
  GenerateData() override;

  /** Multi-threaded functionality that does the work. */
  void
  BeforeThreadedGenerateData() override;

  void
  ThreadedGenerateData(const InputImageRegionType & inputRegionForThread, ThreadIdType threadId) override;

  /** Generate a point randomly in a bounding box. */
  virtual void
  GenerateRandomCoordinate(const InputImageContinuousIndexType & smallestContIndex,
                           const InputImageContinuousIndexType & largestContIndex,
                           InputImageContinuousIndexType &       randomContIndex);

  InterpolatorPointer m_Interpolator = [] {
    const auto interpolator = DefaultInterpolatorType::New();
    interpolator->SetSplineOrder(3);
    return interpolator;
  }();

  RandomGeneratorPointer m_RandomGenerator{ RandomGeneratorType::GetInstance() };
  InputImageSpacingType  m_SampleRegionSize{ itk::MakeFilled<InputImageSpacingType>(1.0) };

  /** Generate the two corners of a sampling region, given the two corners
   * of an image. If UseRandomSampleRegion=false, the smallesPoint and largestPoint
   * are just copies of the smallestImagePoint and largestImagePoint
   * Otherwise, the midpoint of the sample region is randomly selected and
   * the two corners are computed using the SampleRegionSize */
  virtual void
  GenerateSampleRegion(const InputImageContinuousIndexType & smallestImageContIndex,
                       const InputImageContinuousIndexType & largestImageContIndex,
                       InputImageContinuousIndexType &       smallestContIndex,
                       InputImageContinuousIndexType &       largestContIndex);

private:
  bool m_UseRandomSampleRegion{ false };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageRandomCoordinateSampler.hxx"
#endif

#endif // end #ifndef itkImageRandomCoordinateSampler_h
