/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMultiResolutionShrinkPyramidImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2009-04-07 13:14:19 $
  Version:   $Revision: 1.32 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
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
