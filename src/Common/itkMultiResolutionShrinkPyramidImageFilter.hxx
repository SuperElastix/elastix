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
#ifndef __itkMultiResolutionShrinkPyramidImageFilter_hxx
#define __itkMultiResolutionShrinkPyramidImageFilter_hxx

#include "itkMultiResolutionShrinkPyramidImageFilter.h"

#include "itkShrinkImageFilter.h"
#include "vnl/vnl_math.h"

namespace itk
{

/*
 * GenerateData
 */
template< class TInputImage, class TOutputImage >
void
MultiResolutionShrinkPyramidImageFilter< TInputImage, TOutputImage >
::GenerateData( void )
{
  /** Create the shrinking filter. */
  typedef ShrinkImageFilter< TInputImage, TOutputImage > ShrinkerType;
  typename ShrinkerType::Pointer shrinker = ShrinkerType::New();
  shrinker->SetInput( this->GetInput() );

  /** Loop over all resolution levels. */
  unsigned int factors[ ImageDimension ];
  for( unsigned int ilevel = 0; ilevel < this->m_NumberOfLevels; ilevel++ )
  {
    this->UpdateProgress( static_cast< float >( ilevel )
      / static_cast< float >( this->m_NumberOfLevels ) );

    // Allocate memory for each output
    OutputImagePointer outputPtr = this->GetOutput( ilevel );
    outputPtr->SetBufferedRegion( outputPtr->GetRequestedRegion() );
    outputPtr->Allocate();

    // compute and set shrink factors
    for( unsigned int idim = 0; idim < ImageDimension; idim++ )
    {
      factors[ idim ] = this->m_Schedule[ ilevel ][ idim ];
    }
    shrinker->SetShrinkFactors( factors );
    shrinker->GraftOutput( outputPtr );

    // force to always update in case shrink factors are the same
    shrinker->Modified();
    shrinker->UpdateLargestPossibleRegion();
    this->GraftNthOutput( ilevel, shrinker->GetOutput() );
  }
} // end GenerateData()


/**
 * GenerateInputRequestedRegion
 */
template< class TInputImage, class TOutputImage >
void
MultiResolutionShrinkPyramidImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion( void )
{
  // call the superclass' implementation of this method
  Superclass::Superclass::GenerateInputRequestedRegion();
}


} // namespace itk

#endif
