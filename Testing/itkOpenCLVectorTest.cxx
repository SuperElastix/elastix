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
#include "itkTestHelper.h"
#include "itkOpenCLVectorTest.h"
#include "itkOpenCLVector.h"
#include <algorithm>

//------------------------------------------------------------------------------
template< class type >
bool
std_all_of( const typename std::vector< type > & v, const type value )
{
  for( typename std::vector< type >::const_iterator it = v.begin(); it != v.end(); ++it )
  {
    if( *it != value )
    {
      return false;
    }
  }
  return true;
}


//------------------------------------------------------------------------------
template< class type >
bool
std_first_of( const typename std::vector< type > & v, const std::size_t n, const type value )
{
  for( std::size_t i = 0; i < n; i++ )
  {
    if( v[ i ] != value )
    {
      return false;
    }
  }
  return true;
}


//------------------------------------------------------------------------------
int
main( int argc, char * argv[] )
{
  typedef unsigned char                        VectorTypeUChar;
  typedef float                                VectorTypeFloat;
  typedef itk::OpenCLVector< VectorTypeUChar > OCLVectorTypeUChar;
  typedef itk::OpenCLVector< VectorTypeFloat > OCLVectorTypeFloat;
  OCLVectorTypeUChar vectorNull;

  if( !vectorNull.IsNull() )
  {
    return EXIT_FAILURE;
  }

  try
  {
    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
    context->Create( itk::OpenCLContext::DevelopmentSingleMaximumFlopsDevice );
    const std::list< itk::OpenCLDevice > devices = context->GetDevices();

    itk::OpenCLProgram program = context->BuildProgramFromSourceCode( devices,
      itk::OpenCLVectorTestKernel::GetOpenCLSource() );

    if( program.IsNull() )
    {
      if( context->GetDefaultDevice().HasCompiler() )
      {
        itkGenericExceptionMacro( << "Could not compile the OpenCL test program" );
      }
      else
      {
        itkGenericExceptionMacro( << "OpenCL implementation does not have a compiler" );
      }
    }

    // Local variables
    const std::size_t     bufferSize = 4096;
    const VectorTypeUChar valueUChar = 5;
    const VectorTypeFloat valueFloat = 5.1f;

    const std::size_t     bufferUCharSizeBytes = sizeof( VectorTypeUChar ) * bufferSize;
    const std::size_t     bufferFloatSizeBytes = sizeof( VectorTypeFloat ) * bufferSize;
    const itk::OpenCLSize workSize( bufferSize );
    const itk::OpenCLSize smallSize( 8 );

    // Tests the OpenCLVector
    OCLVectorTypeUChar deviceUCharVector
      = context->CreateVector< VectorTypeUChar >( itk::OpenCLMemoryObject::WriteOnly, bufferUCharSizeBytes );
    ITK_OPENCL_COMPARE( deviceUCharVector.IsNull(), false );
    std::cout << deviceUCharVector << std::endl;

    OCLVectorTypeFloat deviceFloatVector
      = context->CreateVector< VectorTypeFloat >( itk::OpenCLMemoryObject::WriteOnly, bufferFloatSizeBytes );
    ITK_OPENCL_COMPARE( deviceFloatVector.IsNull(), false );
    std::cout << deviceFloatVector << std::endl;

    // Create kernels
    itk::OpenCLKernel setUCharKernel = program.CreateKernel( "SetUChar" );
    ITK_OPENCL_COMPARE( setUCharKernel.IsNull(), false );
    setUCharKernel.SetGlobalWorkSize( smallSize );
    setUCharKernel( deviceUCharVector, valueUChar );

    itk::OpenCLKernel setFloatKernel = program.CreateKernel( "SetFloat" );
    ITK_OPENCL_COMPARE( setFloatKernel.IsNull(), false );
    setFloatKernel.SetGlobalWorkSize( smallSize );
    setFloatKernel( deviceFloatVector, valueFloat );

    // Compare results from kernel setUChar
    std::vector< VectorTypeUChar > hostBufferUChar( bufferSize );
    deviceUCharVector.Read( &hostBufferUChar[ 0 ], bufferSize );
    const bool resultUChar0 = std_first_of< VectorTypeUChar >( hostBufferUChar, smallSize.GetSizes()[ 0 ], valueUChar );
    ITK_OPENCL_COMPARE( resultUChar0, true );

    // Run again kernel setUChar and compare results
    setUCharKernel.SetGlobalWorkSize( workSize );
    itk::OpenCLEvent eventUChar = setUCharKernel.LaunchKernel();
    ITK_OPENCL_COMPARE( eventUChar.IsNull(), false );
    eventUChar.WaitForFinished();

    deviceUCharVector.Read( &hostBufferUChar[ 0 ], bufferSize );
    const bool resultUChar1 = std_all_of< VectorTypeUChar >( hostBufferUChar, valueUChar );

    // Only in the latest compilers
    //const bool resultUChar = std::all_of( hostBufferUChar.begin(), hostBufferUChar.end(),
    //  [ valueUChar ]( VectorTypeUChar i ){
    //    return i == valueUChar;
    //  } );
    ITK_OPENCL_COMPARE( resultUChar1, true );

    // Compare results from kernel setFloat
    std::vector< VectorTypeFloat > hostBufferFloat( bufferSize );
    deviceFloatVector.Read( &hostBufferFloat[ 0 ], bufferSize );
    const bool resultFloat0 = std_first_of< VectorTypeFloat >( hostBufferFloat, smallSize.GetSizes()[ 0 ], valueFloat );
    ITK_OPENCL_COMPARE( resultFloat0, true );

    // Run again kernel setFloat and compare results
    setFloatKernel.SetGlobalWorkSize( workSize );
    itk::OpenCLEvent eventFloat = setFloatKernel.LaunchKernel();
    ITK_OPENCL_COMPARE( eventFloat.IsNull(), false );
    eventFloat.WaitForFinished();

    deviceFloatVector.Read( &hostBufferFloat[ 0 ], bufferSize );
    const bool resultFloat1 = std_all_of< VectorTypeFloat >( hostBufferFloat, valueFloat );

    // Only in the latest compilers
    //const bool resultFloat = std::all_of( hostBufferFloat.begin(), hostBufferFloat.end(),
    //  [ valueFloat ]( VectorTypeFloat i ){
    //    return i == valueFloat;
    //  } );
    ITK_OPENCL_COMPARE( resultFloat1, true );
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "Caught ITK exception: " << e << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  itk::ReleaseContext();
  return EXIT_SUCCESS;
}
