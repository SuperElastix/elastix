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
    this->m_NumberOfSamples = 100;
    /** Setup random generator */
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
    typename MaskType::Pointer mask = const_cast<MaskType *>( this->GetMask() );

    /** Make sure the internal full sampler is up-to-date */
    this->m_InternalFullSampler->SetInput(inputImage);
    this->m_InternalFullSampler->SetMask(mask);
    this->m_InternalFullSampler->Update();
    typename ImageSampleContainerType::Pointer allValidSamples =
      this->m_InternalFullSampler->GetOutput();
    unsigned long numberOfValidSamples = allValidSamples->Size();

    /** Take random samples from the allValidSamples-container */
    for ( unsigned int i = 0; i < this->GetNumberOfSamples(); ++i )
    {
      unsigned long randomIndex = 
        this->m_RandomGenerator->GetIntegerVariate( numberOfValidSamples );
      sampleContainer->push_back( allValidSamples->ElementAt( randomIndex ) );
    }  

  } // end GenerateData


  /**
	 * ******************* PrintSelf *******************
	 */
  
  template< class TInputImage >
    void
    ImageRandomSamplerSparseMask< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );
  } // end PrintSelf



} // end namespace itk

#endif // end #ifndef __ImageRandomSamplerSparseMask_txx

