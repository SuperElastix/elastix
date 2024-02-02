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
#ifndef itkImageFullSampler_h
#define itkImageFullSampler_h

#include "itkImageSamplerBase.h"
#include "elxMaskHasSameImageDomain.h"

namespace itk
{
/** \class ImageFullSampler
 *
 * \brief Samples all voxels in the InputImageRegion.
 *
 * This ImageSampler samples all voxels in the InputImageRegion.
 * If a mask is given: only those voxels within the mask AND the
 * InputImageRegion.
 *
 * \ingroup ImageSamplers
 */

template <class TInputImage>
class ITK_TEMPLATE_EXPORT ImageFullSampler : public ImageSamplerBase<TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageFullSampler);

  /** Standard ITK-stuff. */
  using Self = ImageFullSampler;
  using Superclass = ImageSamplerBase<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageFullSampler, ImageSamplerBase);

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

  // Clang/macos-12/Xcode_14.2 does not like `using typename Superclass::MaskType`, saying "error: 'MaskType' is not a
  // class, namespace, or enumeration"
  using MaskType = typename Superclass::MaskType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass::InputImageDimension);

  /** Other typdefs. */
  using typename Superclass::InputImageIndexType;
  // using typename Superclass::InputImageSpacingType;
  using typename Superclass::InputImagePointType;

  /** Typedefs for support of user defined grid spacing for the spatial samples. */
  using InputImageSizeType = typename InputImageType::SizeType;

  /** Selecting new samples makes no sense if nothing changed. The same samples would be selected anyway. */
  bool
  SelectNewSamplesOnUpdate() override
  {
    return false;
  }


  /** Returns whether the sampler supports SelectNewSamplesOnUpdate(). */
  bool
  SelectingNewSamplesOnUpdateSupported() const override
  {
    return false;
  }


protected:
  /** The constructor. */
  ImageFullSampler() = default;

  /** The destructor. */
  ~ImageFullSampler() override = default;

  using Superclass::PrintSelf;

  /** Function that does the work. */
  void
  GenerateData() override;

private:
  using WorldToObjectTransformType = AffineTransform<double, InputImageDimension>;

  struct WorkUnit
  {
    const InputImageRegionType imageRegion{};

    // Should point to the first sample for this specific work unit.
    ImageSampleType * const Samples{};

    // The number of samples retrieved by this work unit. Only used when a mask is specified.
    size_t NumberOfSamples{};
  };

  struct UserData
  {
    ITK_DISALLOW_COPY_AND_MOVE(UserData);

    const InputImageType &                   InputImage;
    const MaskType * const                   Mask{};
    const WorldToObjectTransformType * const WorldToObjectTransform{};
    std::vector<WorkUnit>                    WorkUnits{};
  };

  template <elastix::MaskCondition VMaskCondition>
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ThreaderCallback(void * arg);

  /** Generates the work units, to be processed when doing multi-threading. */
  static std::vector<WorkUnit>
  GenerateWorkUnits(const ThreadIdType             numberOfWorkUnits,
                    const InputImageRegionType &   croppedInputImageRegion,
                    std::vector<ImageSampleType> & samples);

  static void
  SingleThreadedGenerateData(const TInputImage &            inputImage,
                             const MaskType * const         mask,
                             const InputImageRegionType &   croppedInputImageRegion,
                             std::vector<ImageSampleType> & samples);
  static void
  MultiThreadedGenerateData(MultiThreaderBase &            multiThreader,
                            const ThreadIdType             numberOfWorkUnits,
                            const TInputImage &            inputImage,
                            const MaskType * const         mask,
                            const InputImageRegionType &   croppedInputImageRegion,
                            std::vector<ImageSampleType> & samples);

  /** Generates the data for one specific work unit. */
  template <elastix::MaskCondition VMaskCondition>
  static void
  GenerateDataForWorkUnit(WorkUnit &, const InputImageType &, const MaskType *, const WorldToObjectTransformType *);
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageFullSampler.hxx"
#endif

#endif // end #ifndef itkImageFullSampler_h
