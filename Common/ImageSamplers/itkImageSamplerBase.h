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
#ifndef itkImageSamplerBase_h
#define itkImageSamplerBase_h

#include "itkVectorContainerSource.h"
#include "itkImageSample.h"
#include "itkVectorDataContainer.h"
#include "itkSpatialObject.h"

namespace itk
{
/** \class ImageSamplerBase
 *
 * \brief This class is a base class for any image sampler.
 *
 * \parameter ImageSampler: The way samples are taken from the fixed image in
 *    order to compute the metric value and its derivative in each iteration.
 *    Can be given for each resolution. Select one of {Random, Full, Grid, RandomCoordinate}.\n
 *    example: <tt>(ImageSampler "Random")</tt> \n
 *    The default is Random.
 *
 * \ingroup ImageSamplers
 */

template <class TInputImage>
class ITK_TEMPLATE_EXPORT ImageSamplerBase
  : public VectorContainerSource<VectorDataContainer<std::size_t, ImageSample<TInputImage>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageSamplerBase);

  /** Standard ITK-stuff. */
  using Self = ImageSamplerBase;
  using Superclass = VectorContainerSource<VectorDataContainer<std::size_t, ImageSample<TInputImage>>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageSamplerBase, Superclass);

  /** Typedefs inherited from the superclass. */
  using typename Superclass::DataObjectPointer;
  using typename Superclass::OutputVectorContainerType;
  using typename Superclass::OutputVectorContainerPointer;

  /** Some Image related typedefs. */
  using InputImageType = TInputImage;
  using InputImagePointer = typename InputImageType::Pointer;
  using InputImageConstPointer = typename InputImageType::ConstPointer;
  using InputImageRegionType = typename InputImageType::RegionType;
  using InputImagePixelType = typename InputImageType::PixelType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, InputImageType::ImageDimension);

  /** Other typdefs. */
  using ImageSampleType = ImageSample<InputImageType>;
  using ImageSampleContainerType = VectorDataContainer<std::size_t, ImageSampleType>;
  using ImageSampleContainerPointer = typename ImageSampleContainerType::Pointer;
  using InputImageSizeType = typename InputImageType::SizeType;
  using InputImageIndexType = typename InputImageType::IndexType;
  using InputImagePointType = typename InputImageType::PointType;
  using InputImagePointValueType = typename InputImagePointType::ValueType;
  using ImageSampleValueType = typename ImageSampleType::RealType;
  using MaskType = SpatialObject<Self::InputImageDimension>;
  using MaskPointer = typename MaskType::Pointer;
  using MaskConstPointer = typename MaskType::ConstPointer;
  using MaskVectorType = std::vector<MaskConstPointer>;
  using InputImageRegionVectorType = std::vector<InputImageRegionType>;

  /** Create a valid output. */
  DataObject::Pointer
  MakeOutput(unsigned int idx) override;

  /** Set the input image of this process object.  */
  void
  SetInput(unsigned int idx, const InputImageType * input);

  /** Set the input image of this process object.  */
  void
  SetInput(const InputImageType * input);

  /** Get the input image of this process object.  */
  const InputImageType *
  GetInput();

  /** Get the input image of this process object.  */
  const InputImageType *
  GetInput(unsigned int idx);

  /** Get the output Mesh of this process object.  */
  OutputVectorContainerType *
  GetOutput();

  /** Prepare the output. */
  // virtual void GenerateOutputInformation();

  /** A version of GenerateData() specific for image processing
   * filters.  This implementation will split the processing across
   * multiple threads. The buffer is allocated by this method. Then
   * the BeforeThreadedGenerateData() method is called (if
   * provided). Then, a series of threads are spawned each calling
   * ThreadedGenerateData(). After all the threads have completed
   * processing, the AfterThreadedGenerateData() method is called (if
   * provided). If an image processing filter cannot be threaded, the
   * filter should provide an implementation of GenerateData(). That
   * implementation is responsible for allocating the output buffer.
   * If a filter an be threaded, it should NOT provide a
   * GenerateData() method but should provide a ThreadedGenerateData()
   * instead.
   *
   * \sa ThreadedGenerateData() */
  void
  GenerateData() override;

  /** If an imaging filter can be implemented as a multithreaded
   * algorithm, the filter will provide an implementation of
   * ThreadedGenerateData().  This superclass will automatically split
   * the output image into a number of pieces, spawn multiple threads,
   * and call ThreadedGenerateData() in each thread. Prior to spawning
   * threads, the BeforeThreadedGenerateData() method is called. After
   * all the threads have completed, the AfterThreadedGenerateData()
   * method is called. If an image processing filter cannot support
   * threading, that filter should provide an implementation of the
   * GenerateData() method instead of providing an implementation of
   * ThreadedGenerateData().  If a filter provides a GenerateData()
   * method as its implementation, then the filter is responsible for
   * allocating the output data.  If a filter provides a
   * ThreadedGenerateData() method as its implementation, then the
   * output memory will allocated automatically by this superclass.
   * The ThreadedGenerateData() method should only produce the output
   * specified by "outputThreadRegion"
   * parameter. ThreadedGenerateData() cannot write to any other
   * portion of the output image (as this is responsibility of a
   * different thread).
   *
   * \sa GenerateData(), SplitRequestedRegion() */
  virtual void
  ThreadedGenerateData(const InputImageRegionType & inputRegionForThread, ThreadIdType threadId);

  /** ******************** Masks ******************** */

  /** Set the masks. */
  virtual void
  SetMask(const MaskType * _arg, unsigned int pos);

  /** Set the first mask. NB: the first mask is used to
   * compute a bounding box in which samples are considered.
   */
  virtual void
  SetMask(const MaskType * _arg)
  {
    this->SetMask(_arg, 0);
  }


  /** Get the masks. */
  virtual const MaskType *
  GetMask(unsigned int pos) const;

  /** Get the first mask. */
  virtual const MaskType *
  GetMask() const
  {
    return this->GetMask(0);
  }


  /** Set the number of masks. */
  virtual void
  SetNumberOfMasks(const unsigned int _arg);

  /** Get the number of masks. */
  itkGetConstMacro(NumberOfMasks, unsigned int);

  /** ******************** Regions ******************** */

  /** Set the region over which the samples will be taken. */
  virtual void
  SetInputImageRegion(const InputImageRegionType _arg, unsigned int pos);

  /** Set the region over which the samples will be taken. */
  virtual void
  SetInputImageRegion(const InputImageRegionType _arg)
  {
    this->SetInputImageRegion(_arg, 0);
  }


  /** Get the input image regions. */
  virtual const InputImageRegionType &
  GetInputImageRegion(unsigned int pos) const;

  /** Get the first input image region. */
  virtual const InputImageRegionType &
  GetInputImageRegion() const
  {
    return this->GetInputImageRegion(0);
  }


  /** Set the number of input image regions. */
  virtual void
  SetNumberOfInputImageRegions(const unsigned int _arg);

  /** Get the number of input image regions. */
  itkGetConstMacro(NumberOfInputImageRegions, unsigned int);

  /** ******************** Other ******************** */

  /** SelectNewSamplesOnUpdate. When this function is called, the sampler
   * will generate a new sample set after calling Update(). The return bool
   * is false when this feature is not supported by the sampler.
   */
  virtual bool
  SelectNewSamplesOnUpdate();

  /** Returns whether the sampler supports SelectNewSamplesOnUpdate() */
  virtual bool
  SelectingNewSamplesOnUpdateSupported() const
  {
    return true;
  }


  /** Get a handle to the cropped InputImageregion. */
  itkGetConstReferenceMacro(CroppedInputImageRegion, InputImageRegionType);

  /** Set/Get the number of samples. */
  itkSetClampMacro(NumberOfSamples, unsigned long, 1, NumericTraits<unsigned long>::max());
  itkGetConstMacro(NumberOfSamples, unsigned long);

  /** Allows disabling the use of multi-threading, by `SetUseMultiThread(false)`. */
  itkSetMacro(UseMultiThread, bool);

protected:
  /** The constructor. */
  ImageSamplerBase();

  /** The destructor. */
  ~ImageSamplerBase() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** GenerateInputRequestedRegion. */
  void
  GenerateInputRequestedRegion() override;

  /** IsInsideAllMasks. */
  virtual bool
  IsInsideAllMasks(const InputImagePointType & point) const;

  /** UpdateAllMasks. */
  virtual void
  UpdateAllMasks();

  /** Checks if the InputImageRegions are a subregion of the
   * LargestPossibleRegions.
   */
  virtual bool
  CheckInputImageRegions();

  /** Compute the intersection of the InputImageRegion and the bounding box of the mask. */
  void
  CropInputImageRegion();

  /** Multi-threaded function that does the work. */
  virtual void
  BeforeThreadedGenerateData();

  virtual void
  AfterThreadedGenerateData();

  /***/
  unsigned long                             m_NumberOfSamples{ 0 };
  std::vector<std::vector<ImageSampleType>> m_ThreaderSampleVectors{};

  /** `UseMultiThread == true` indicated that the sampler _may_ use multi-threading (if implemented), while
   * `UseMultiThread == false` indicates that the sampler should _not_ use multi-threading */
  bool m_UseMultiThread{ true };

private:
  /** Static function used as a "callback" by the PlatformMultiThreader.  The threading
   * library will call this routine for each thread, which will delegate the
   * control to ThreadedGenerateData(). */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ThreaderCallback(void * arg);

  /** Internal structure used for passing image data into the threading library */
  struct ThreadStruct //?
  {
    Pointer Sampler;
  };

  /** Split the input region into "numberOfSplits" pieces, returning
   * region "i" as "splitRegion". This method is called "numberOfSplits" times. The
   * regions must not overlap. The method returns the number of pieces that
   * the routine is capable of splitting the input region,
   * i.e. return value is less than or equal to "numberOfSplits". */
  static unsigned int
  SplitRegion(const InputImageRegionType & inputRegion,
              const ThreadIdType           threadId,
              const ThreadIdType           numberOfSplits,
              InputImageRegionType &       splitRegion);

  /** Member variables. */
  MaskConstPointer           m_Mask{ nullptr };
  MaskVectorType             m_MaskVector{};
  unsigned int               m_NumberOfMasks{ 0 };
  InputImageRegionType       m_InputImageRegion{};
  InputImageRegionVectorType m_InputImageRegionVector{};
  unsigned int               m_NumberOfInputImageRegions{ 0 };

  InputImageRegionType m_CroppedInputImageRegion{};
  InputImageRegionType m_DummyInputImageRegion{};
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageSamplerBase.hxx"
#endif

#endif // end #ifndef itkImageSamplerBase_h
