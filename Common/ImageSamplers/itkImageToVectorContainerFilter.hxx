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
#ifndef __ImageToVectorContainerFilter_hxx
#define __ImageToVectorContainerFilter_hxx

#include "itkImageToVectorContainerFilter.h"

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
    = dynamic_cast< OutputVectorContainerType * >( this->MakeOutput( 0 ).GetPointer() );

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
::SetInput( unsigned int idx, const InputImageType * input )
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
::SetInput( const InputImageType * input )
{
  this->ProcessObject::SetNthInput( 0, const_cast< InputImageType * >( input ) );
} // end SetInput()


/**
 * ******************* GetInput *******************
 */

template< class TInputImage, class TOutputVectorContainer >
const typename ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >::InputImageType
* ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::GetInput( void )
{
  return dynamic_cast< const InputImageType * >(
    this->ProcessObject::GetInput( 0 ) );
} // end GetInput()

/**
 * ******************* GetInput *******************
 */

template< class TInputImage, class TOutputVectorContainer >
const typename ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >::InputImageType
* ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
::GetInput( unsigned int idx )
{
  return dynamic_cast< const InputImageType * >(
    this->ProcessObject::GetInput( idx ) );
} // end GetInput()

/**
 * ******************* GetOutput *******************
 */

template< class TInputImage, class TOutputVectorContainer >
typename ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >::OutputVectorContainerType
* ImageToVectorContainerFilter< TInputImage, TOutputVectorContainer >
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
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
} // end PrintSelf()


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
  this->GetMultiThreader()->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
  this->GetMultiThreader()->template ParallelizeImageRegion<InputImageType::ImageDimension>(
    this->GetInput()->GetRequestedRegion(),
    [this](const InputImageRegionType & regionForThread) {
      this->DynamicThreadedGenerateData(regionForThread);
    },
    this);

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
::DynamicThreadedGenerateData( const InputImageRegionType & )
{
  // The following code is equivalent to:
  // itkExceptionMacro("subclass should override this method!!!");
  // The ExceptionMacro is not used because gcc warns that a
  // 'noreturn' function does return
  std::ostringstream message;
  message << "itk::ERROR: " << this->GetNameOfClass()
          << "(" << this << "): " << "Subclass should override this method!!!";
  ExceptionObject e_( __FILE__, __LINE__, message.str().c_str(), ITK_LOCATION );
  throw e_;

} // end ThreadedGenerateData()


} // end namespace itk

#endif // end #ifndef __ImageToVectorContainerFilter_hxx
