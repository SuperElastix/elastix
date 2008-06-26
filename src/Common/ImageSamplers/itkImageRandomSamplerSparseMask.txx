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
    this->m_InternalFullSampler->SetMask(  this->GetMask() );
    this->m_InternalFullSampler->SetInputImageRegion( this->GetInputImageRegion() );
    this->m_InternalFullSampler->Update();
    typename ImageSampleContainerType::Pointer allValidSamples =
      this->m_InternalFullSampler->GetOutput();
    unsigned long numberOfValidSamples = allValidSamples->Size();

    /** Take random samples from the allValidSamples-container. */
    for ( unsigned int i = 0; i < this->GetNumberOfSamples(); ++i )
    {
      unsigned long randomIndex = 
        this->m_RandomGenerator->GetIntegerVariate( numberOfValidSamples - 1 );
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

    os << indent << "InternalFullSampler: " << this->m_InternalFullSampler.GetPointer() << std::endl;
    os << indent << "RandomGenerator: " << this->m_RandomGenerator.GetPointer() << std::endl;

  } // end PrintSelf


} // end namespace itk

#endif // end #ifndef __ImageRandomSamplerSparseMask_txx

