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
#include "itkOpenCLKernelToImageBridgeTest.h"
#include "itkOpenCLKernelToImageBridge.h"

int
main( int argc, char * argv[] )
{
  try
  {
    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
    context->Create( itk::OpenCLContext::DevelopmentSingleMaximumFlopsDevice );
    const std::list< itk::OpenCLDevice > devices = context->GetDevices();

    itk::OpenCLProgram program = context->BuildProgramFromSourceCode( devices,
      itk::OpenCLKernelToImageBridgeTestKernel::GetOpenCLSource() );

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

    // Create ITK Image
    typedef itk::Image< float, 2 > ImageType;
    ImageType::Pointer image = ImageType::New();

    ImageType::SizeType size;
    size[ 0 ] = 64;
    size[ 1 ] = 64;

    ImageType::SpacingType   spacing; spacing.Fill( 1.1 );
    ImageType::PointType     origin; origin.Fill( 3.2 );
    ImageType::DirectionType direction;
    direction[ 0 ][ 0 ] = .5;
    direction[ 0 ][ 1 ] = .7;
    direction[ 1 ][ 0 ] = .7;
    direction[ 1 ][ 1 ] = .5;
    image->SetSpacing( spacing );
    image->SetOrigin( origin );
    image->SetDirection( direction );

    image->SetRegions( size );
    image->Allocate();
    image->FillBuffer( 11 );
    image->Print( std::cout );

    // Check the setting image information
    const std::size_t bufferSize      = 4;
    const std::size_t bufferSizeBytes = sizeof( float ) * bufferSize;
    itk::OpenCLBuffer directionBuffer = context->CreateBufferDevice( itk::OpenCLMemoryObject::WriteOnly, bufferSizeBytes );
    ITK_OPENCL_COMPARE( directionBuffer.IsNull(), false );

    itk::OpenCLKernel directionKernel = program.CreateKernel( "SetDirection" );
    ITK_OPENCL_COMPARE( directionKernel.IsNull(), false );
    directionKernel( image->GetDirection(), directionBuffer );

    // Check setting the direction
    std::vector< float > hostDirectionBuffer( bufferSize );
    directionBuffer.Read( &hostDirectionBuffer[ 0 ], bufferSizeBytes );
    ITK_OPENCL_COMPARE( hostDirectionBuffer[ 0 ], 0.5f );
    ITK_OPENCL_COMPARE( hostDirectionBuffer[ 1 ], 0.7f );
    ITK_OPENCL_COMPARE( hostDirectionBuffer[ 2 ], 0.7f );
    ITK_OPENCL_COMPARE( hostDirectionBuffer[ 3 ], 0.5f );
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
