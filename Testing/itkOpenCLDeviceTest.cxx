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
#include "itkOpenCLDevice.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLProgram.h"

int
main( int argc, char * argv[] )
{
  itk::OpenCLDevice device;

  if( !device.IsNull() )
  {
    return EXIT_FAILURE;
  }

  // Get all devices
  std::list< itk::OpenCLDevice >       gpus;
  const std::list< itk::OpenCLDevice > devices = itk::OpenCLDevice::GetAllDevices();
  for( std::list< itk::OpenCLDevice >::const_iterator dev = devices.begin(); dev != devices.end(); ++dev )
  {
    if( ( ( *dev ).GetDeviceType() & itk::OpenCLDevice::GPU ) != 0 )
    {
      gpus.push_back( *dev );
      std::cout << ( *dev );
    }
  }

  return EXIT_SUCCESS;
}
