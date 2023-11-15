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
#ifndef itkImageRandomSampler_h
#define itkImageRandomSampler_h

#include "itkImageRandomSamplerBase.h"
#include <optional>

namespace itk
{
/** \class ImageRandomSampler
 *
 * \brief Samples randomly some voxels of an image.
 *
 * This image sampler randomly samples 'NumberOfSamples' voxels in
 * the InputImageRegion. Voxels may be selected multiple times.
 * If a mask is given, the sampler tries to find samples within the
 * mask. If the mask is very sparse, this may take some time. In this case,
 * consider using the ImageRandomSamplerSparseMask.
 *
 * \ingroup ImageSamplers
 */

template <class TInputImage>
class ITK_TEMPLATE_EXPORT ImageRandomSampler : public ImageRandomSamplerBase<TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageRandomSampler);

  /** Standard ITK-stuff. */
  using Self = ImageRandomSampler;
  using Superclass = ImageRandomSamplerBase<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageRandomSampler, ImageRandomSamplerBase);

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
  using typename Superclass::ImageSampleValueType;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::MaskType;
  using typename Superclass::InputImageSizeType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass::InputImageDimension);

  /** Other typedefs. */
  using InputImageIndexType = typename InputImageType::IndexType;
  using InputImagePointType = typename InputImageType::PointType;

  using SeedIntegerType = Statistics::MersenneTwisterRandomVariateGenerator::IntegerType;

  /** Allows setting the randomization seed. Optional: when the argument does not have a seed value, an arbitrary seed
   * may be used for randomization (which is the default behavior). */
  void
  SetOptionalSeed(const std::optional<SeedIntegerType> arg)
  {
    if (arg != m_OptionalSeed)
    {
      m_OptionalSeed = arg;
      this->Modified();
    }
  }

  /** Sets the randomization seed as specified. */
  void
  SetSeed(const SeedIntegerType arg)
  {
    SetOptionalSeed(std::make_optional(arg));
  }

  /** Retrieves the optional user specified randomization seed. */
  [[nodiscard]] std::optional<SeedIntegerType>
  GetOptionalSeed() const
  {
    return m_OptionalSeed;
  }

protected:
  /** The constructor. */
  ImageRandomSampler() = default;
  /** The destructor. */
  ~ImageRandomSampler() override = default;

  /** Functions that do the work. */
  void
  GenerateData() override;

private:
  /** Generates the list of random numbers. */
  void
  GenerateRandomNumberList();

  struct UserData
  {
    ITK_DISALLOW_COPY_AND_MOVE(UserData);

    UserData(const std::vector<double> &    randomNumberList,
             const InputImageType &         inputImage,
             const InputImageRegionType &   region,
             std::vector<ImageSampleType> & samples)
      : m_RandomNumberList(randomNumberList)
      , m_InputImage(inputImage)
      , m_RegionIndex(region.GetIndex())
      , m_RegionSize(region.GetSize())
      , m_Samples(samples)
    {}

    const std::vector<double> &    m_RandomNumberList;
    const InputImageType &         m_InputImage;
    InputImageIndexType            m_RegionIndex{};
    InputImageSizeType             m_RegionSize{};
    std::vector<ImageSampleType> & m_Samples;
  };

  std::optional<UserData> m_OptionalUserData{};

  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ThreaderCallback(void * arg);

  /** Member variable used when threading. */
  std::vector<double> m_RandomNumberList{};

  std::optional<SeedIntegerType> m_OptionalSeed{};
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageRandomSampler.hxx"
#endif

#endif // end #ifndef itkImageRandomSampler_h
