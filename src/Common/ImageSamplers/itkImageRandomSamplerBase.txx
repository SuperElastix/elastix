/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageRandomSamplerBase_txx
#define __ImageRandomSamplerBase_txx

#include "itkImageRandomSamplerBase.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkImageRandomConstIteratorWithIndex.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TInputImage >
ImageRandomSamplerBase< TInputImage >
::ImageRandomSamplerBase()
{
  this->m_NumberOfSamples = 1000;

} // end Constructor


/**
 * ******************* BeforeThreadedGenerateData *******************
 */

template< class TInputImage >
void
ImageRandomSamplerBase< TInputImage >
::BeforeThreadedGenerateData( void )
{
  /** Create a random number generator. Also used in the ImageRandomConstIteratorWithIndex. */
  typedef typename Statistics::MersenneTwisterRandomVariateGenerator::Pointer GeneratorPointer;
  GeneratorPointer localGenerator = Statistics::MersenneTwisterRandomVariateGenerator::GetInstance();
  // \todo: should probably be global?

  /** Clear the random number list. */
  this->m_RandomNumberList.resize( 0 );
  this->m_RandomNumberList.reserve( this->m_NumberOfSamples );

  /** Fill the list with random numbers. */
  const double numPixels = static_cast<double>( this->GetCroppedInputImageRegion().GetNumberOfPixels() );
  localGenerator->GetVariateWithOpenRange( numPixels - 0.5 ); // dummy jump
  for ( unsigned long i = 0; i < this->m_NumberOfSamples; i++ )
  {
    const double randomPosition
      = localGenerator->GetVariateWithOpenRange( numPixels - 0.5 );
    this->m_RandomNumberList.push_back( randomPosition );
  }
  localGenerator->GetVariateWithOpenRange( numPixels - 0.5 ); // dummy jump

  /** Initialize variables needed for threads. */
  Superclass::BeforeThreadedGenerateData();

} // end BeforeThreadedGenerateData()


/**
 * ******************* PrintSelf *******************
 */

template< class TInputImage >
void
ImageRandomSamplerBase< TInputImage >
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  
  os << indent << "NumberOfSamples: " << this->m_NumberOfSamples << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef __ImageRandomSamplerBase_txx
