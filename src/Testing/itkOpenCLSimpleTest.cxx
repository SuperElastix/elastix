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
#include "itkOpenCLSimpleTest.h"
#include "itkOpenCLProgram.h"

//------------------------------------------------------------------------------
// This test is mainly to test CMake generating process when two kernels are
// merged into one source code see OpenCLSimpleTestKernel.cxx
int
main( int argc, char * argv[] )
{
  try
  {
    std::list< itk::OpenCLDevice > devices;
    itk::OpenCLContext::Pointer    context = itk::OpenCLContext::GetInstance();
    context->Create( itk::OpenCLContext::DevelopmentMultipleMaximumFlopsDevices );
    devices = context->GetDevices();

    itk::OpenCLProgram programAllGPU = context->BuildProgramFromSourceCode( devices,
      itk::OpenCLSimpleTest1Kernel::GetOpenCLSource() );

    if( context.IsNull() )
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
    context->Release();

    // Create context with maximum flops
    context->Create( itk::OpenCLContext::SingleMaximumFlopsDevice );
    devices = context->GetDevices();
    itk::OpenCLProgram programMaxFlops = context->BuildProgramFromSourceCode( devices,
      itk::OpenCLSimpleTest2Kernel::GetOpenCLSource() );

    if( programMaxFlops.IsNull() )
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
    context->Release();
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
