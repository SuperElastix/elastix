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
#ifndef itkImageGridSampler_h
#define itkImageGridSampler_h

#include "itkImageSamplerBase.h"
#include "elxMaskHasSameImageDomain.h"

namespace itk
{

/** \class ImageGridSampler
 *
 * \brief Samples image voxels on a regular grid.
 *
 * This ImageSampler samples voxels that lie on a regular grid.
 * The grid can be specified by an integer downsampling factor for
 * each dimension.
 *
 * \parameter SampleGridSpacing: This parameter controls the spacing
 *    of the uniform grid in all dimensions. This should be given in
 *    index coordinates. \n
 *    example: <tt>(SampleGridSpacing 4 4 4)</tt> \n
 *    Default is 2 in each dimension.
 *
 * \ingroup ImageSamplers
 */

template <typename TInputImage>
class ITK_TEMPLATE_EXPORT ImageGridSampler : public ImageSamplerBase<TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageGridSampler);

  /** Standard ITK-stuff. */
  using Self = ImageGridSampler;
  using Superclass = ImageSamplerBase<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageGridSampler, ImageSamplerBase);

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
  using SampleGridSpacingType = typename InputImageType::OffsetType;
  using SampleGridSpacingValueType = typename SampleGridSpacingType::OffsetValueType;
  using SampleGridSizeType = typename InputImageType::SizeType;
  using SampleGridIndexType = InputImageIndexType;
  using InputImageSizeType = typename InputImageType::SizeType;

  /** Set/Get the sample grid spacing for each dimension (only integer factors)
   * This function overrules previous calls to SetNumberOfSamples.
   * Moreover, it calls SetNumberOfSamples(0) (see below), to make sure
   * that the user-set sample grid spacing is never overruled.
   */
  void
  SetSampleGridSpacing(const SampleGridSpacingType & arg);

  itkGetConstReferenceMacro(SampleGridSpacing, SampleGridSpacingType);

  /** Define an isotropic SampleGridSpacing such that the desired number
   * of samples is approximately realized. The following formula is used:
   *
   * spacing = max[ 1, round( (availablevoxels / nrofsamples)^(1/dimension) ) ],
   * with
   * availablevoxels = nr of voxels in bounding box of the mask.
   *
   * The InputImageRegion needs to be specified beforehand.
   * However, the sample grid spacing is recomputed in the update phase, when the
   * bounding box of the mask is known. Supplying nrofsamples=0 turns off the
   * (re)computation of the SampleGridSpacing. Once nrofsamples=0 has been given,
   * the last computed SampleGridSpacing is simply considered as a user parameter,
   * which is not modified automatically anymore.
   *
   * This function overrules any previous calls to SetSampleGridSpacing.
   */
  void
  SetNumberOfSamples(unsigned long nrofsamples) override;

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
  ImageGridSampler() = default;

  /** The destructor. */
  ~ImageGridSampler() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Function that does the work. */
  void
  GenerateData() override;

private:
  struct WorkUnit
  {
    const SampleGridIndexType GridIndex{};
    const SampleGridSizeType  GridSize{};

    // Should point to the first sample for this specific work unit.
    ImageSampleType * const Samples{};

    // The number of samples retrieved by this work unit. Only used when a mask is specified.
    size_t NumberOfSamples{};
  };

  struct UserData
  {
    const InputImageType &      InputImage;
    const MaskType * const      Mask{};
    const SampleGridSpacingType GridSpacing{};
    std::vector<WorkUnit>       WorkUnits{};
  };

  template <elastix::MaskCondition VMaskCondition>
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ThreaderCallback(void * arg);

  /** Retrieves the sample grid size along the axis, specified by VIndex */
  template <unsigned int VIndex>
  static unsigned int
  GetGridSizeValue(const SampleGridSizeType & gridSize)
  {
    if constexpr (VIndex < InputImageDimension)
    {
      return gridSize[VIndex];
    }
    else
    {
      return 1;
    }
  }

  /** Jumps to the next grid position along the axis, specified by VIndex */
  template <unsigned int VIndex>
  static void
  JumpToNextGridPosition(SampleGridIndexType &         index,
                         const SampleGridIndexType &   gridIndex,
                         const SampleGridSpacingType & gridSpacing)
  {
    static_assert(VIndex > 0);

    if constexpr (VIndex < InputImageDimension)
    {
      index[VIndex - 1] = gridIndex[VIndex - 1];
      index[VIndex] += gridSpacing[VIndex];
    }
  }


  /** Determine the grid. */
  static std::pair<SampleGridIndexType, SampleGridSizeType>
  DetermineGridIndexAndSize(const InputImageRegionType &  croppedInputImageRegion,
                            const SampleGridSpacingType & gridSpacing);

  /** Generates the work units, to be processed when doing multi-threading. */
  static std::vector<WorkUnit>
  GenerateWorkUnits(const ThreadIdType             numberOfWorkUnits,
                    const InputImageRegionType &   croppedInputImageRegion,
                    const SampleGridIndexType      gridIndex,
                    const SampleGridSpacingType    gridSpacing,
                    std::vector<ImageSampleType> & samples);

  static void
  SingleThreadedGenerateData(const TInputImage &            inputImage,
                             const MaskType * const         mask,
                             const InputImageRegionType &   croppedInputImageRegion,
                             const SampleGridSpacingType &  gridSpacing,
                             std::vector<ImageSampleType> & samples);
  static void
  MultiThreadedGenerateData(MultiThreaderBase &            multiThreader,
                            const ThreadIdType             numberOfWorkUnits,
                            const TInputImage &            inputImage,
                            const MaskType * const         mask,
                            const InputImageRegionType &   croppedInputImageRegion,
                            const SampleGridSpacingType &  gridSpacing,
                            std::vector<ImageSampleType> & samples);

  /** Generates the data for one specific work unit. */
  template <elastix::MaskCondition VMaskCondition>
  static void
  GenerateDataForWorkUnit(WorkUnit &, const InputImageType &, const MaskType *, const SampleGridSpacingType &);

  /** An array of integer spacing factors */
  SampleGridSpacingType m_SampleGridSpacing{ itk::MakeFilled<SampleGridSpacingType>(1) };

  /** The number of samples entered in the SetNumberOfSamples method */
  unsigned long m_RequestedNumberOfSamples{ 0 };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageGridSampler.hxx"
#endif

#endif // end #ifndef itkImageGridSampler_h
