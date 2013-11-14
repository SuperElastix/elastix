/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageRandomSamplerSparseMask_txx
#define __ImageRandomSamplerSparseMask_txx

#include "itkImageRandomSamplerSparseMask.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TInputImage >
ImageRandomSamplerSparseMask< TInputImage >
::ImageRandomSamplerSparseMask()
{
  /** Setup random generator. */
  this->m_RandomGenerator = RandomGeneratorType::GetInstance();

  this->m_InternalFullSampler = InternalFullSamplerType::New();

} // end Constructor


/**
 * ******************* GenerateData *******************
 */

template< class TInputImage >
void
ImageRandomSamplerSparseMask< TInputImage >
::GenerateData( void )
{
  /** Get a handle to the mask. */
  typename MaskType::ConstPointer mask = this->GetMask();

  /** Sanity check. */
  if ( mask.IsNull() )
  {
    itkExceptionMacro( << "ERROR: do not call this function when no mask is supplied." );
  }

  /** Get handles to the input image and output sample container. */
  InputImageConstPointer inputImage = this->GetInput();
  ImageSampleContainerPointer sampleContainer = this->GetOutput();

  /** Clear the container. */
  sampleContainer->Initialize();

  /** Make sure the internal full sampler is up-to-date. */
  this->m_InternalFullSampler->SetInput( inputImage );
  this->m_InternalFullSampler->SetMask( mask );
  this->m_InternalFullSampler->SetInputImageRegion( this->GetCroppedInputImageRegion() );

  /** Use try/catch, since the full sampler may crash, due to insufficient memory. */
  try
  {
    this->m_InternalFullSampler->Update();
  }
  catch ( ExceptionObject & err )
  {
    std::string message = "ERROR: This ImageSampler internally uses the "
      "ImageFullSampler. Updating of this internal sampler raised the "
      "exception:\n";
    message += err.GetDescription();

    std::string fullSamplerMessage = err.GetDescription();
    std::string::size_type loc = fullSamplerMessage.find(
      "ERROR: failed to allocate memory for the sample container", 0 );
    if ( loc != std::string::npos && this->GetMask() == 0 )
    {
      message += "\nYou are using the ImageRandomSamplerSparseMask sampler, "
        "but you did not set a mask. The internal ImageFullSampler therefore "
        "requires a lot of memory. Consider using the ImageRandomSampler "
        "instead.";
    }
    const char * message2 = message.c_str();
    itkExceptionMacro( << message2 );
  }

  /** If desired we exercise a multi-threaded version. */
  if ( this->m_UseMultiThread )
  {
    /** Calls ThreadedGenerateData(). */
    return Superclass::GenerateData();
  }

  /** Get a handle to the full sampler output. */
  typename ImageSampleContainerType::Pointer allValidSamples
    = this->m_InternalFullSampler->GetOutput();
  unsigned long numberOfValidSamples = allValidSamples->Size();

  /** Take random samples from the allValidSamples-container. */
  for ( unsigned int i = 0; i < this->GetNumberOfSamples(); ++i )
  {
    unsigned long randomIndex
      = this->m_RandomGenerator->GetIntegerVariate( numberOfValidSamples - 1 );
    sampleContainer->push_back( allValidSamples->ElementAt( randomIndex ) );
  }

} // end GenerateData()


/**
 * ******************* BeforeThreadedGenerateData *******************
 */

template< class TInputImage >
void
ImageRandomSamplerSparseMask< TInputImage >
::BeforeThreadedGenerateData( void )
{
  /** Clear the random number list. */
  this->m_RandomNumberList.resize( 0 );
  this->m_RandomNumberList.reserve( this->m_NumberOfSamples );

  /** Get a handle to the full sampler output size. */
  const unsigned long numberOfValidSamples
    = this->m_InternalFullSampler->GetOutput()->Size();

  /** Fill the list with random numbers. */
  for ( unsigned int i = 0; i < this->GetNumberOfSamples(); ++i )
  {
    unsigned long randomIndex
      = this->m_RandomGenerator->GetIntegerVariate( numberOfValidSamples - 1 );
    this->m_RandomNumberList.push_back( randomIndex );
  }

  /** Initialize variables needed for threads. */
  this->m_ThreaderSampleContainer.clear();
  this->m_ThreaderSampleContainer.resize( this->GetNumberOfThreads() );
  for ( std::size_t i = 0; i < this->GetNumberOfThreads(); i++ )
  {
    this->m_ThreaderSampleContainer[ i ] = ImageSampleContainerType::New();
  }

} // end BeforeThreadedGenerateData()


/**
 * ******************* ThreadedGenerateData *******************
 */

template< class TInputImage >
void
ImageRandomSamplerSparseMask< TInputImage >
::ThreadedGenerateData( const InputImageRegionType &, ThreadIdType threadId )
{
  /** Get a handle to the full sampler output. */
  typename ImageSampleContainerType::Pointer allValidSamples
    = this->m_InternalFullSampler->GetOutput();

  /** Figure out which samples to process. */
  unsigned long chunkSize = this->GetNumberOfSamples() / this->GetNumberOfThreads();
  unsigned long sampleStart = threadId * chunkSize;
  if ( threadId == this->GetNumberOfThreads() - 1 )
  {
    chunkSize = this->GetNumberOfSamples()
      - ( ( this->GetNumberOfThreads() - 1 ) * chunkSize );
  }

  /** Get a reference to the output and reserve memory for it. */
  ImageSampleContainerPointer & sampleContainerThisThread
    = this->m_ThreaderSampleContainer[ threadId ];
  sampleContainerThisThread->Reserve( chunkSize );

  /** Setup an iterator over the sampleContainerThisThread. */
  typename ImageSampleContainerType::Iterator iter;
  typename ImageSampleContainerType::ConstIterator end = sampleContainerThisThread->End();

  /** Take random samples from the allValidSamples-container. */
  unsigned long sampleId = sampleStart;
  for ( iter = sampleContainerThisThread->Begin(); iter != end; ++iter, sampleId++ )
  {
    unsigned long randomIndex = static_cast<unsigned long>( this->m_RandomNumberList[ sampleId ] );
    (*iter).Value() = allValidSamples->ElementAt( randomIndex );
  }

} // end ThreadedGenerateData()


/**
 * ******************* PrintSelf *******************
 */

template< class TInputImage >
void
ImageRandomSamplerSparseMask< TInputImage >
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "InternalFullSampler: " << this->m_InternalFullSampler.GetPointer() << std::endl;
  os << indent << "RandomGenerator: " << this->m_RandomGenerator.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef __ImageRandomSamplerSparseMask_txx

