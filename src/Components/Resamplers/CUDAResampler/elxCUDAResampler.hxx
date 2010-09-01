/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxCUDAResampler_hxx
#define __elxCUDAResampler_hxx

#include "elxCUDAResampler.h"


namespace elastix
{

/**
 * ******************* BeforeAll ***********************
 */

template <class TElastix>
int
CUDAResampler<TElastix>
::BeforeAll( void )
{
  int res = Superclass1::CudaResampleImageFilterType::checkExecutionParameters();
  if ( res != 0 )
  {
    itkExceptionMacro( "ERROR: no valid CUDA devices found!" );
  }
  return res;

  // implement checks for CUDA cards available.

} // end BeforeAll()


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
CUDAResampler<TElastix>
::BeforeRegistration( void )
{
  /** Are we using a CUDA enabled GPU for resampling? */
  bool useCUDA = false;
  this->m_Configuration->ReadParameter( useCUDA, "UseCUDA", 0 );
  this->SetUseCuda( useCUDA );

} // end BeforeRegistration()

} /* namespace elastix */

#endif

