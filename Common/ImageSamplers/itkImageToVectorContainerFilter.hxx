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
#ifndef itkImageToVectorContainerFilter_hxx
#define itkImageToVectorContainerFilter_hxx

#include "itkImageToVectorContainerFilter.h"

#include "itkMath.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TInputImage, class TOutputVectorContainer>
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::ImageToVectorContainerFilter()
{
  this->ProcessObject::SetNumberOfRequiredInputs(1);

  OutputVectorContainerPointer output = dynamic_cast<OutputVectorContainerType *>(this->MakeOutput(0).GetPointer());

  this->ProcessObject::SetNumberOfRequiredOutputs(1);
  this->ProcessObject::SetNthOutput(0, output.GetPointer());

} // end Constructor


/**
 * ******************* MakeOutput *******************
 */

template <class TInputImage, class TOutputVectorContainer>
DataObject::Pointer
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::MakeOutput(unsigned int itkNotUsed(idx))
{
  OutputVectorContainerPointer outputVectorContainer = OutputVectorContainerType::New();
  return dynamic_cast<DataObject *>(outputVectorContainer.GetPointer());
} // end MakeOutput()


/**
 * ******************* SetInput *******************
 */

template <class TInputImage, class TOutputVectorContainer>
void
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::SetInput(unsigned int           idx,
                                                                            const InputImageType * input)
{
  // process object is not const-correct, the const_cast
  // is required here.
  this->ProcessObject::SetNthInput(idx, const_cast<InputImageType *>(input));
} // end SetInput()


/**
 * ******************* SetInput *******************
 */

template <class TInputImage, class TOutputVectorContainer>
void
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::SetInput(const InputImageType * input)
{
  this->ProcessObject::SetNthInput(0, const_cast<InputImageType *>(input));
} // end SetInput()


/**
 * ******************* GetInput *******************
 */

template <class TInputImage, class TOutputVectorContainer>
auto
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::GetInput() -> const InputImageType *
{
  return dynamic_cast<const InputImageType *>(this->ProcessObject::GetInput(0));
} // end GetInput()

/**
 * ******************* GetInput *******************
 */

template <class TInputImage, class TOutputVectorContainer>
auto
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::GetInput(unsigned int idx) -> const InputImageType *
{
  return dynamic_cast<const InputImageType *>(this->ProcessObject::GetInput(idx));
} // end GetInput()

/**
 * ******************* GetOutput *******************
 */

template <class TInputImage, class TOutputVectorContainer>
auto
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::GetOutput() -> OutputVectorContainerType *
{
  return dynamic_cast<OutputVectorContainerType *>(this->ProcessObject::GetOutput(0));
} // end GetOutput()

/**
 * ******************* PrintSelf *******************
 */

template <class TInputImage, class TOutputVectorContainer>
void
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
} // end PrintSelf()


/**
 * ******************* SplitRequestedRegion *******************
 */

template <class TInputImage, class TOutputVectorContainer>
unsigned int
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::SplitRequestedRegion(
  const ThreadIdType &   threadId,
  const ThreadIdType &   numberOfSplits,
  InputImageRegionType & splitRegion)
{
  // Get the input pointer
  const InputImageType *                 inputPtr = this->GetInput();
  const typename TInputImage::SizeType & requestedRegionSize = inputPtr->GetRequestedRegion().GetSize();
  // \todo: requested region -> this->GetCroppedInputImageRegion()

  int                             splitAxis;
  typename TInputImage::IndexType splitIndex;
  typename TInputImage::SizeType  splitSize;

  // Initialize the splitRegion to the output requested region
  splitRegion = inputPtr->GetRequestedRegion();
  splitIndex = splitRegion.GetIndex();
  splitSize = splitRegion.GetSize();

  // split on the outermost dimension available
  splitAxis = inputPtr->GetImageDimension() - 1;
  while (requestedRegionSize[splitAxis] == 1)
  {
    --splitAxis;
    if (splitAxis < 0)
    { // cannot split
      itkDebugMacro("  Cannot Split");
      return 1;
    }
  }

  // determine the actual number of pieces that will be generated
  typename TInputImage::SizeType::SizeValueType range = requestedRegionSize[splitAxis];
  unsigned int valuesPerThread = Math::Ceil<unsigned int>(range / (double)numberOfSplits);
  unsigned int maxThreadIdUsed = Math::Ceil<unsigned int>(range / (double)valuesPerThread) - 1;

  // Split the region
  if (threadId < maxThreadIdUsed)
  {
    splitIndex[splitAxis] += threadId * valuesPerThread;
    splitSize[splitAxis] = valuesPerThread;
  }
  if (threadId == maxThreadIdUsed)
  {
    splitIndex[splitAxis] += threadId * valuesPerThread;
    // last thread needs to process the "rest" dimension being split
    splitSize[splitAxis] = splitSize[splitAxis] - threadId * valuesPerThread;
  }

  // set the split region ivars
  splitRegion.SetIndex(splitIndex);
  splitRegion.SetSize(splitSize);

  itkDebugMacro(<< "  Split Piece: " << splitRegion);

  return maxThreadIdUsed + 1;

} // end SplitRequestedRegion()


/**
 * ******************* GenerateData *******************
 */

template <class TInputImage, class TOutputVectorContainer>
void
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::GenerateData()
{
  // Call a method that can be overriden by a subclass to allocate
  // memory for the filter's outputs
  // this->AllocateOutputs();

  // Call a method that can be overridden by a subclass to perform
  // some calculations prior to splitting the main computations into
  // separate threads
  this->BeforeThreadedGenerateData();

  // Set up the multithreaded processing
  ThreadStruct str;
  str.Filter = this;

  this->GetMultiThreader()->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
  this->GetMultiThreader()->SetSingleMethod(this->ThreaderCallback, &str);

  // multithread the execution
  this->GetMultiThreader()->SingleMethodExecute();

  // Call a method that can be overridden by a subclass to perform
  // some calculations after all the threads have completed
  this->AfterThreadedGenerateData();

} // end GenerateData()


/**
 * ******************* ThreadedGenerateData *******************
 */

template <class TInputImage, class TOutputVectorContainer>
void
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::ThreadedGenerateData(const InputImageRegionType &,
                                                                                        ThreadIdType)
{
  // The following code is equivalent to:
  // itkExceptionMacro("subclass should override this method!!!");
  // The ExceptionMacro is not used because gcc warns that a
  // 'noreturn' function does return
  std::ostringstream message;
  message << "itk::ERROR: " << this->GetNameOfClass() << "(" << this << "): Subclass should override this method!!!";
  ExceptionObject e_(__FILE__, __LINE__, message.str().c_str(), ITK_LOCATION);
  throw e_;

} // end ThreadedGenerateData()


/**
 * ******************* ThreaderCallback *******************
 */

// Callback routine used by the threading library. This routine just calls
// the ThreadedGenerateData method after setting the correct region for this
// thread.
template <class TInputImage, class TOutputVectorContainer>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
ImageToVectorContainerFilter<TInputImage, TOutputVectorContainer>::ThreaderCallback(void * arg)
{
  ThreadStruct * str;
  ThreadIdType   threadId = ((PlatformMultiThreader::WorkUnitInfo *)(arg))->WorkUnitID;
  ThreadIdType   threadCount = ((PlatformMultiThreader::WorkUnitInfo *)(arg))->NumberOfWorkUnits;

  str = (ThreadStruct *)(((PlatformMultiThreader::WorkUnitInfo *)(arg))->UserData);

  // execute the actual method with appropriate output region
  // first find out how many pieces extent can be split into.
  typename TInputImage::RegionType splitRegion;
  unsigned int                     total = str->Filter->SplitRequestedRegion(threadId, threadCount, splitRegion);

  if (threadId < total)
  {
    str->Filter->ThreadedGenerateData(splitRegion, threadId);
  }
  // else
  //   {
  //   otherwise don't use this thread. Sometimes the threads dont
  //   break up very well and it is just as efficient to leave a
  //   few threads idle.
  //   }

  return itk::ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end ThreaderCallback()


} // end namespace itk

#endif // end #ifndef itkImageToVectorContainerFilter_hxx
