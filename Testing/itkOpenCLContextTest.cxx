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
#include "itkOpenCLContext.h"
#include "itkTestHelper.h"

int
main( int argc, char * argv[] )
{
  itk::OpenCLContext::Pointer contextNull = itk::OpenCLContext::New();

  if( contextNull->IsCreated() )
  {
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();

  // Check the pointers, they should be the same
  if( contextNull.GetPointer() != context.GetPointer() )
  {
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  //context->Create(itk::OpenCLContext::Default);
  context->Create( itk::OpenCLContext::DevelopmentSingleMaximumFlopsDevice );
  //context->Create(itk::OpenCLContext::DevelopmentMultipleMaximumFlopsDevices);
  //context->Create(itk::OpenCLContext::SingleMaximumFlopsDevice);
  //context->Create(itk::OpenCLContext::MultipleMaximumFlopsDevices);

  if( !context->IsCreated() )
  {
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  std::list< itk::OpenCLDevice > devices = context->GetDevices();
  for( std::list< itk::OpenCLDevice >::const_iterator dev = devices.begin(); dev != devices.end(); ++dev )
  {
    std::cout << ( *dev ) << std::endl;
  }

  // Release and exit
  itk::ReleaseContext();
  return EXIT_SUCCESS;
}
