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
#ifndef itkImageToVectorContainerFilter_h
#define itkImageToVectorContainerFilter_h

#include "itkVectorContainerSource.h"
#include "itkPlatformMultiThreader.h"

namespace itk
{

/** \class ImageToVectorContainerFilter
 *
 * \brief Base class that takes in an image and pops out
 * a vector container.
 */

template <class TInputImage, class TOutputVectorContainer>
class ITK_TEMPLATE_EXPORT ImageToVectorContainerFilter : public VectorContainerSource<TOutputVectorContainer>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageToVectorContainerFilter);

  /** Standard ITK-stuff. */
  using Self = ImageToVectorContainerFilter;
  using Superclass = VectorContainerSource<TOutputVectorContainer>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageToVectorContainerFilter, VectorContainerSource);

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

  /** If an imaging filter needs to perform processing after the buffer
   * has been allocated but before threads are spawned, the filter can
   * can provide an implementation for BeforeThreadedGenerateData(). The
   * execution flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void
  BeforeThreadedGenerateData()
  {}

  /** If an imaging filter needs to perform processing after all
   * processing threads have completed, the filter can can provide an
   * implementation for AfterThreadedGenerateData(). The execution
   * flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void
  AfterThreadedGenerateData()
  {}

  /** Split the output's RequestedRegion into "numberOfSplits" pieces, returning
   * region "i" as "splitRegion". This method is called "numberOfSplits" times. The
   * regions must not overlap. The method returns the number of pieces that
   * the routine is capable of splitting the output RequestedRegion,
   * i.e. return value is less than or equal to "numberOfSplits". */
  virtual unsigned int
  SplitRequestedRegion(const ThreadIdType &   threadId,
                       const ThreadIdType &   numberOfSplits,
                       InputImageRegionType & splitRegion);

  /** Static function used as a "callback" by the PlatformMultiThreader.  The threading
   * library will call this routine for each thread, which will delegate the
   * control to ThreadedGenerateData(). */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ThreaderCallback(void * arg);

  /** Internal structure used for passing image data into the threading library */
  struct ThreadStruct //?
  {
    Pointer Filter;
  };

protected:
  /** The constructor. */
  ImageToVectorContainerFilter();
  /** The destructor. */
  ~ImageToVectorContainerFilter() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageToVectorContainerFilter.hxx"
#endif

#endif // end #ifndef itkImageToVectorContainerFilter_h
