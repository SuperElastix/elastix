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
    this->m_RandomGenerator = RandomGeneratorType::New();
    //this->m_RandomGenerator->Initialize();

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
    /** Get handles to the input image and output sample container. */
    InputImageConstPointer inputImage = this->GetInput();
    typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();

    /** Clear the container. */
    sampleContainer->Initialize();

    /** Make sure the internal full sampler is up-to-date. */
    this->m_InternalFullSampler->SetInput( inputImage );
    this->m_InternalFullSampler->SetMask( this->GetMask() );
    this->m_InternalFullSampler->SetInputImageRegion( this->GetCroppedInputImageRegion() );

    /** Use try/catch, since the full sampler may crash, due to insufficient
     * memory.
     */
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

