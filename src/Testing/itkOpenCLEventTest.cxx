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
#include "itkOpenCLEventTest.h"

int
main( int argc, char * argv[] )
{
  itk::OpenCLEvent eventNull;

  if( !eventNull.IsNull() )
  {
    return EXIT_FAILURE;
  }

  try
  {
    // Create and check OpenCL context
    if( !itk::CreateContext() )
    {
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }

    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();

    // Setup for OpenCL profiling
#ifdef OPENCL_PROFILING
    if( !context->GetDefaultCommandQueue().IsProfilingEnabled() )
    {
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }
#else
    itk::OpenCLCommandQueue queue
      = context->CreateCommandQueue( CL_QUEUE_PROFILING_ENABLE );
    if( !queue.IsProfilingEnabled() )
    {
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }
    context->SetCommandQueue( queue );
#endif

    // Create program
    itk::OpenCLProgram program = context->BuildProgramFromSourceCode(
      context->GetDevices(), itk::OpenCLEventTestKernel::GetOpenCLSource() );
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

    // Create vector
    itk::OpenCLVector< float > oclVector
      = context->CreateVector< float >( itk::OpenCLMemoryObject::ReadWrite, 1000 );
    for( std::size_t index = 0; index < oclVector.GetSize(); ++index )
    {
      oclVector[ index ] = float(index);
    }

    itk::OpenCLKernel kernel = program.CreateKernel( "AddToVector" );
    ITK_OPENCL_COMPARE( kernel.IsNull(), false );

    kernel.SetGlobalWorkSize( oclVector.GetSize() );
    itk::OpenCLEvent event = kernel( oclVector, 1567.4f );

    // Wait to finish
    event.WaitForFinished();

    // Check the event execution times
    if( event.GetFinishTime() == 0 )
    {
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }

    if( event.GetSubmitTime() <= event.GetQueueTime() )
    {
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }

    if( event.GetRunTime() <= event.GetSubmitTime() )
    {
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }

    if( event.GetFinishTime() <= event.GetRunTime() )
    {
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }

    context->SetCommandQueue( context->GetDefaultCommandQueue() );
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
