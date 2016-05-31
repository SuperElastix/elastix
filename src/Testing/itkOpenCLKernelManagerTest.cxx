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
#include "itkOpenCLKernelManager.h"
#include "itkTestHelper.h"

int
main( int argc, char * argv[] )
{
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  context->Create( itk::OpenCLContext::DevelopmentSingleMaximumFlopsDevice );
  if( !context->IsCreated() )
  {
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  itk::OpenCLKernelManager::Pointer kernelManager = itk::OpenCLKernelManager::New();
  if( kernelManager.IsNull() )
  {
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  itk::ReleaseContext();
  return EXIT_SUCCESS;
}
