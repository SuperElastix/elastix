/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageToVectorContainerFilter_txx
#define __ImageToVectorContainerFilter_txx

#include "itkImageToVectorContainerFilter.h"

#include "elxTimer.h" // debugging
#include "itkMath.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TInputImage, class TOutputVectorContainer >
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::ImageToVectorContainerFilter()
{
  this->ProcessObject::SetNumberOfRequiredInputs( 1 );

  OutputVectorContainerPointer output
    = dynamic_cast<OutputVectorContainerType*>( this->MakeOutput(0).GetPointer() );

  this->ProcessObject::SetNumberOfRequiredOutputs( 1 );
  this->ProcessObject::SetNthOutput( 0, output.GetPointer() );

} // end Constructor


/**
 * ******************* MakeOutput *******************
 */

template< class TInputImage, class TOutputVectorContainer >
DataObject::Pointer
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::MakeOutput( unsigned int itkNotUsed( idx ) )
{
  OutputVectorContainerPointer outputVectorContainer = OutputVectorContainerType::New();
  return dynamic_cast< DataObject * >( outputVectorContainer.GetPointer() );
} // end MakeOutput()


/**
 * ******************* SetInput *******************
 */

template< class TInputImage, class TOutputVectorContainer >
void
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::SetInput( unsigned int idx, const InputImageType *input )
{
  // process object is not const-correct, the const_cast
  // is required here.
  this->ProcessObject::SetNthInput( idx,
    const_cast< InputImageType * >( input ) );
} // end SetInput()


/**
 * ******************* SetInput *******************
 */

template< class TInputImage, class TOutputVectorContainer >
void
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::SetInput( const InputImageType *input )
{
  this->ProcessObject::SetNthInput( 0, const_cast< InputImageType * >( input ) );
} // end SetInput()


/**
 * ******************* GetInput *******************
 */

template< class TInputImage, class TOutputVectorContainer >
const typename ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >::InputImageType *
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::GetInput( void )
{
  return dynamic_cast< const InputImageType * >(
    this->ProcessObject::GetInput( 0 ) );
} // end GetInput()


/**
 * ******************* GetInput *******************
 */

template< class TInputImage, class TOutputVectorContainer >
const typename ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >::InputImageType *
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::GetInput( unsigned int idx )
{
  return dynamic_cast< const InputImageType * >(
    this->ProcessObject::GetInput( idx ) );
} // end GetInput()


/**
 * ******************* GetOutput *******************
 */

template< class TInputImage, class TOutputVectorContainer >
typename ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >::OutputVectorContainerType *
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::GetOutput( void )
{
  return dynamic_cast< OutputVectorContainerType * >(
    this->ProcessObject::GetOutput( 0 ) );
} // end GetOutput()


/**
 * ******************* PrintSelf *******************
 */

template< class TInputImage, class TOutputVectorContainer >
void
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
} // end PrintSelf()


/**
 * ******************* SplitRequestedRegion *******************
 */

template< class TInputImage, class TOutputVectorContainer >
unsigned int
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::SplitRequestedRegion( const ThreadIdType & threadId,
  const ThreadIdType & numberOfSplits, InputImageRegionType & splitRegion )
{
  // Get the input pointer
  const InputImageType * inputPtr = this->GetInput();
  const typename TInputImage::SizeType & requestedRegionSize 
    = inputPtr->GetRequestedRegion().GetSize();
  // \todo: requested region -> this->GetCroppedInputImageRegion()

  int splitAxis;
  typename TInputImage::IndexType splitIndex;
  typename TInputImage::SizeType splitSize;

  // Initialize the splitRegion to the output requested region
  splitRegion = inputPtr->GetRequestedRegion();
  splitIndex = splitRegion.GetIndex();
  splitSize = splitRegion.GetSize();

  // split on the outermost dimension available
  splitAxis = inputPtr->GetImageDimension() - 1;
  while ( requestedRegionSize[splitAxis] == 1 )
  {
    --splitAxis;
    if ( splitAxis < 0 )
    { // cannot split
      itkDebugMacro("  Cannot Split");
      return 1;
    }
  }

  // determine the actual number of pieces that will be generated
  typename TInputImage::SizeType::SizeValueType range = requestedRegionSize[ splitAxis ];
  unsigned int valuesPerThread = Math::Ceil<unsigned int>( range / (double) numberOfSplits );
  unsigned int maxThreadIdUsed = Math::Ceil<unsigned int>( range / (double) valuesPerThread ) - 1;

  // Split the region
  if ( threadId < maxThreadIdUsed )
  {
    splitIndex[ splitAxis ] += threadId * valuesPerThread;
    splitSize[ splitAxis ] = valuesPerThread;
  }
  if ( threadId == maxThreadIdUsed )
  {
    splitIndex[ splitAxis ] += threadId * valuesPerThread;
    // last thread needs to process the "rest" dimension being split
    splitSize[ splitAxis ] = splitSize[ splitAxis ] - threadId * valuesPerThread;
  }

  // set the split region ivars
  splitRegion.SetIndex( splitIndex );
  splitRegion.SetSize( splitSize );

  itkDebugMacro( << "  Split Piece: " << splitRegion );

  return maxThreadIdUsed + 1;

} // end SplitRequestedRegion()


/**
 * ******************* GenerateData *******************
 */

template< class TInputImage, class TOutputVectorContainer >
void
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::GenerateData( void )
{
  // Call a method that can be overriden by a subclass to allocate
  // memory for the filter's outputs
  //this->AllocateOutputs();

  // Call a method that can be overridden by a subclass to perform
  // some calculations prior to splitting the main computations into
  // separate threads
  this->BeforeThreadedGenerateData();

  // Set up the multithreaded processing
  ThreadStruct str;
  str.Filter = this;

  this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
  this->GetMultiThreader()->SetSingleMethod( this->ThreaderCallback, &str );

  // multithread the execution
  this->GetMultiThreader()->SingleMethodExecute();

  // Call a method that can be overridden by a subclass to perform
  // some calculations after all the threads have completed
  this->AfterThreadedGenerateData();

} // end GenerateData()


/**
 * ******************* ThreadedGenerateData *******************
 */

template< class TInputImage, class TOutputVectorContainer >
void
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::ThreadedGenerateData( const InputImageRegionType &, ThreadIdType )
{
  // The following code is equivalent to:
  // itkExceptionMacro("subclass should override this method!!!");
  // The ExceptionMacro is not used because gcc warns that a 
  // 'noreturn' function does return
  std::ostringstream message;
  message << "itk::ERROR: " << this->GetNameOfClass()
    << "(" << this << "): " << "Subclass should override this method!!!";
  ExceptionObject e_(__FILE__, __LINE__, message.str().c_str(),ITK_LOCATION);
  throw e_;

} // end ThreadedGenerateData()


/**
 * ******************* ThreaderCallback *******************
 */

// Callback routine used by the threading library. This routine just calls
// the ThreadedGenerateData method after setting the correct region for this
// thread.
template< class TInputImage, class TOutputVectorContainer >
ITK_THREAD_RETURN_TYPE
ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::ThreaderCallback( void *arg )
{
  ThreadStruct *str;
  ThreadIdType threadId = ((MultiThreader::ThreadInfoStruct *)(arg))->ThreadID;
  ThreadIdType threadCount = ((MultiThreader::ThreadInfoStruct *)(arg))->NumberOfThreads;

  str = (ThreadStruct *)(((MultiThreader::ThreadInfoStruct *)(arg))->UserData);

  // execute the actual method with appropriate output region
  // first find out how many pieces extent can be split into.
  typename TInputImage::RegionType splitRegion;
  unsigned int total = str->Filter->SplitRequestedRegion( threadId, threadCount, splitRegion );

  if ( threadId < total )
  {
    str->Filter->ThreadedGenerateData( splitRegion, threadId );
  }
  // else
  //   {
  //   otherwise don't use this thread. Sometimes the threads dont
  //   break up very well and it is just as efficient to leave a 
  //   few threads idle.
  //   }

  return ITK_THREAD_RETURN_VALUE;

} // end ThreaderCallback()


} // end namespace itk

#endif // end #ifndef __ImageToVectorContainerFilter_txx
