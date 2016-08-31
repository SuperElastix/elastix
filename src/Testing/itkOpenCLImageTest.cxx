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
#include "itkOpenCLImageTest.h"
#include "itkOpenCLImage.h"
#include <algorithm>

int
main( int argc, char * argv[] )
{
  itk::OpenCLImage imageNull;

  if( !imageNull.IsNull() )
  {
    return EXIT_FAILURE;
  }

  try
  {
    // Create context with most simple method
    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
    context->Create();

    if( !context->IsCreated() )
    {
      std::cerr << "OpenCL-enabled device is not present." << std::endl;
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }

    const std::list< itk::OpenCLDevice > devices = context->GetDevices();

    itk::OpenCLProgram program = context->BuildProgramFromSourceCode( devices,
      itk::OpenCLImageTestKernel::GetOpenCLSource() );

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
    const std::size_t     imageSize1D = 32;
    const std::size_t     imageSize2D = imageSize1D * imageSize1D;
    const int             value       = 11;
    const itk::OpenCLSize origin2D( 0, 0 );
    const itk::OpenCLSize workSize2D( imageSize1D, imageSize1D );

    // Test the OpenCLImage 2D
    const itk::OpenCLImageFormat format2d(
      itk::OpenCLImageFormat::IMAGE2D,
      itk::OpenCLImageFormat::RGBA,
      itk::OpenCLImageFormat::UNSIGNED_INT8 );

    std::cout << format2d;
    itk::OpenCLSize size2D( imageSize1D, imageSize1D );

    // Create input 2D image
    std::vector< unsigned int > input_buf_2D( imageSize2D, value );

    itk::OpenCLImage input_image_2D = context->CreateImageDevice( format2d, itk::OpenCLMemoryObject::ReadOnly, size2D );
    ITK_OPENCL_COMPARE( input_image_2D.IsNull(), false );
    ITK_OPENCL_COMPARE( input_image_2D.GetElementSizeInBytes(), std::size_t( 4 ) );
    std::cout << input_image_2D;
    const bool write2D = input_image_2D.Write( &input_buf_2D[ 0 ], origin2D, workSize2D );
    ITK_OPENCL_COMPARE( write2D, true );

    // Create output 2D image
    itk::OpenCLImage output_image_2D = context->CreateImageDevice( format2d, itk::OpenCLMemoryObject::WriteOnly, size2D );
    ITK_OPENCL_COMPARE( output_image_2D.IsNull(), false );
    ITK_OPENCL_COMPARE( output_image_2D.GetElementSizeInBytes(), std::size_t( 4 ) );
    std::cout << output_image_2D;

    itk::OpenCLKernel image2DCopyKernel = program.CreateKernel( "Image2DCopy" );
    ITK_OPENCL_COMPARE( image2DCopyKernel.IsNull(), false );
    image2DCopyKernel.SetGlobalWorkSize( workSize2D );

    image2DCopyKernel.SetArg( 0, input_image_2D );
    image2DCopyKernel.SetArg( 1, output_image_2D );

    // Execute kernel
    itk::OpenCLEvent event = image2DCopyKernel.LaunchKernel();
    ITK_OPENCL_COMPARE( event.IsNull(), false );
    event.WaitForFinished();

    // Compare results
    std::vector< unsigned int > output_buf_2D( imageSize2D );
    output_image_2D.Read( &output_buf_2D[ 0 ], origin2D, workSize2D );
    const bool result = std::equal( input_buf_2D.begin(), input_buf_2D.end(), output_buf_2D.begin() );
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
