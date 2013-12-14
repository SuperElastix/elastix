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

template< class TElastix >
int
CUDAResampler< TElastix >
::BeforeAll( void )
{
  int res = Superclass1::CudaResampleImageFilterType::checkExecutionParameters();
  if( res != 0 )
  {
    itkExceptionMacro( "ERROR: no valid CUDA devices found!" );
  }
  return res;

  // implement checks for CUDA cards available.

} // end BeforeAll()


/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
CUDAResampler< TElastix >
::BeforeRegistration( void )
{
  /** Are we using a CUDA enabled GPU for resampling? */
  bool useCUDA = false;
  this->m_Configuration->ReadParameter( useCUDA, "UseCUDA", 0 );
  this->SetUseCuda( useCUDA );

  /** Are we using the fast CUDA kernel for resampling,
   * or the accurate kernel? Default = accurate.
   */
  bool useFastCUDAKernel = false;
  this->m_Configuration->ReadParameter( useFastCUDAKernel, "UseFastCUDAKernel", 0 );
  this->SetUseFastCUDAKernel( useFastCUDAKernel );

} // end BeforeRegistration()


/*
 * ******************* ReadFromFile  ****************************
 */

template< class TElastix >
void
CUDAResampler< TElastix >
::ReadFromFile( void )
{
  /** Call ReadFromFile of the ResamplerBase. */
  this->Superclass2::ReadFromFile();

  /** CUDAResampler specific. */

  /** Are we using a CUDA enabled GPU for resampling? */
  bool useCUDA = false;
  this->m_Configuration->ReadParameter( useCUDA, "UseCUDA", 0 );
  this->SetUseCuda( useCUDA );

  /** Are we using the fast CUDA kernel for resampling,
   * or the accurate kernel? Default = accurate.
   */
  bool useFastCUDAKernel = false;
  this->m_Configuration->ReadParameter( useFastCUDAKernel, "UseFastCUDAKernel", 0 );
  this->SetUseFastCUDAKernel( useFastCUDAKernel );

} // end ReadFromFile()


/**
 * ************************* WriteToFile ************************
 */

template< class TElastix >
void
CUDAResampler< TElastix >
::WriteToFile( void ) const
{
  /** Call the WriteToFile from the ResamplerBase. */
  this->Superclass2::WriteToFile();

  /** Add some CUDAResampler specific lines. */
  xout[ "transpar" ] << std::endl << "// CUDAResampler specific" << std::endl;

  /** Is CUDA used or not? */
  std::string useCUDA = "false";
  if( this->GetUseCuda() ) { useCUDA = "true"; }
  xout[ "transpar" ] << "(UseCUDA \"" << useCUDA << "\")" << std::endl;

  /** Are we using the fast CUDA kernel for resampling,
   * or the accurate kernel? Default = accurate.
   */
  std::string useFastCUDAKernel = "false";
  if( this->GetUseFastCUDAKernel() ) { useFastCUDAKernel = "true"; }
  xout[ "transpar" ] << "(UseFastCUDAKernel \""
                     << useFastCUDAKernel << "\")" << std::endl;

} // end WriteToFile()


/**
 * ************************* CheckForValidConfiguration ************************
 */

template< class TElastix >
void
CUDAResampler< TElastix >
::CheckForValidConfiguration( ValidTransformPointer & bSplineTransform )
{
  this->Superclass1::CheckForValidConfiguration( bSplineTransform );

  elxout << this->Superclass1::GetWarningReport().GetWarningReportAsString()
         << std::endl;

} // end CheckForValidConfiguration()


} // end namespace elastix

#endif // end #ifndef __elxCUDAResampler_hxx
