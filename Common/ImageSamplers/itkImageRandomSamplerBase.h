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
#ifndef itkImageRandomSamplerBase_h
#define itkImageRandomSamplerBase_h

#include "itkImageSamplerBase.h"
#include <itkMersenneTwisterRandomVariateGenerator.h>
#include "elxDefaultConstruct.h"
#include <optional>

namespace itk
{

/** \class ImageRandomSamplerBase
 *
 * \brief This class is a base class for any image sampler that randomly picks samples.
 *
 * It adds the Set/GetNumberOfSamples function.
 *
 * \ingroup ImageSamplers
 */

template <typename TInputImage>
class ITK_TEMPLATE_EXPORT ImageRandomSamplerBase : public ImageSamplerBase<TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageRandomSamplerBase);

  /** Standard ITK-stuff. */
  using Self = ImageRandomSamplerBase;
  using Superclass = ImageSamplerBase<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ImageRandomSamplerBase);

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

  void
  SetRandomVariateGenerator(Statistics::MersenneTwisterRandomVariateGenerator & randomVariateGenerator)
  {
    m_RandomVariateGenerator = &randomVariateGenerator;
  }


  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass::InputImageDimension);

protected:
  /** The constructor. */
  ImageRandomSamplerBase();

  /** The destructor. */
  ~ImageRandomSamplerBase() override = default;

  /** Generates the list of random numbers. */
  void
  GenerateRandomNumberList();

  Statistics::MersenneTwisterRandomVariateGenerator &
  GetRandomVariateGenerator()
  {
    return *m_RandomVariateGenerator;
  }

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Member variable used when threading. */
  std::vector<double> m_RandomNumberList{};

private:
  std::optional<SeedIntegerType> m_OptionalSeed{};
  SeedIntegerType                m_Seed{ 121212 + 1 };

  elx::DefaultConstruct<Statistics::MersenneTwisterRandomVariateGenerator> m_DefaultRandomVariateGenerator{};
  Statistics::MersenneTwisterRandomVariateGenerator * m_RandomVariateGenerator{ &m_DefaultRandomVariateGenerator };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageRandomSamplerBase.hxx"
#endif

#endif // end #ifndef itkImageRandomSamplerBase_h
