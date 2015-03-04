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
#include "itkOpenCLPlatform.h"

int
main( int argc, char * argv[] )
{
  itk::OpenCLPlatform platform;

  if( !platform.IsNull() )
  {
    return EXIT_FAILURE;
  }

  // Check null platform
  ITK_OPENCL_COMPARE( platform.GetPlatformId(), cl_platform_id( 0 ) );
  ITK_OPENCL_COMPARE( platform == platform, true );
  ITK_OPENCL_COMPARE( !( platform != platform ), true );
  ITK_OPENCL_COMPARE( platform.GetName().empty(), true );

  // Get all platforms and print it
  const std::list< itk::OpenCLPlatform > platforms = itk::OpenCLPlatform::GetAllPlatforms();
  for( std::list< itk::OpenCLPlatform >::const_iterator it = platforms.begin(); it != platforms.end(); ++it )
  {
    std::cout << *it;
  }

  return EXIT_SUCCESS;
}
