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
