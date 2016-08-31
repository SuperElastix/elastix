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
#include "itkOpenCLBufferTest.h"
#include "itkOpenCLBuffer.h"
#include <algorithm>

//------------------------------------------------------------------------------
template< class type >
bool
std_all_of( const std::vector< type > & v, const type value )
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
int
main( int argc, char * argv[] )
{
  itk::OpenCLBuffer bufferNull;

  if( !bufferNull.IsNull() )
  {
    return EXIT_FAILURE;
  }

  try
  {
    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
    context->Create( itk::OpenCLContext::DevelopmentSingleMaximumFlopsDevice );
    const std::list< itk::OpenCLDevice > devices = context->GetDevices();

    itk::OpenCLProgram program = context->BuildProgramFromSourceCode( devices,
      itk::OpenCLBufferTestKernel::GetOpenCLSource() );

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
    const std::size_t     bufferSize      = 16;
    const std::size_t     bufferSizeBytes = sizeof( float ) * bufferSize;
    const float           value           = 5.1f;
    const itk::OpenCLSize workSize( bufferSize );

    // Tests the OpenCLBuffer
    itk::OpenCLBuffer deviceBuffer = context->CreateBufferDevice( itk::OpenCLMemoryObject::WriteOnly, bufferSizeBytes );
    ITK_OPENCL_COMPARE( deviceBuffer.IsNull(), false );
    std::cout << deviceBuffer << std::endl;
    itk::OpenCLKernel setFloatKernel = program.CreateKernel( "SetFloat" );
    ITK_OPENCL_COMPARE( setFloatKernel.IsNull(), false );
    setFloatKernel( deviceBuffer, value );

    // Compare results test1
    std::vector< float > hostBuffer( bufferSize );
    deviceBuffer.Read( &hostBuffer[ 0 ], sizeof( float ) );
    ITK_OPENCL_COMPARE( hostBuffer[ 0 ], value );
    ITK_OPENCL_COMPARE( hostBuffer[ 1 ], 0.0f );

    // Compare results test2
    setFloatKernel.SetGlobalWorkSize( workSize );
    itk::OpenCLEvent event = setFloatKernel.LaunchKernel();
    ITK_OPENCL_COMPARE( event.IsNull(), false );
    event.WaitForFinished();

    deviceBuffer.Read( &hostBuffer[ 0 ], bufferSizeBytes );

    const bool result = std_all_of< float >( hostBuffer, value );

    // Only in the latest compilers
    //const bool result = std::all_of( hostBuffer.begin(), hostBuffer.end(),
    //  [ value ]( float i ){
    //    return i == value;
    //  } );

    ITK_OPENCL_COMPARE( result, true );
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
