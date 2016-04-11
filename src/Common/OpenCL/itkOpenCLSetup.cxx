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
#include "itkOpenCLSetup.h"

#include "itkOpenCLLogger.h"
#include "itkOpenCLContext.h"
#include <sstream>

namespace itk
{
//------------------------------------------------------------------------------
bool
CreateOpenCLContext( std::string & errorMessage,
  const std::string openCLDeviceType,
  const int openCLDeviceID )
{
  /** Get a handle to an existing OpenCL context. */
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();

  /** If it already existed, then do nothing. */
  if( context->IsCreated() ) { return true; }

  /** The default behavior is that the device ID and type are not supplied.
   * In that case we estimate which is the best performing device and select it.
   */
  if( openCLDeviceType == "GPU" && openCLDeviceID == -1 )
  {
#if defined( OPENCL_USE_INTEL_CPU ) || defined( OPENCL_USE_AMD_CPU )
    return context->Create( itk::OpenCLContext::DevelopmentSingleMaximumFlopsDevice );
#else
    return context->Create( itk::OpenCLContext::SingleMaximumFlopsDevice );
#endif
  }

  /** Convert device type string to enum. */
  const std::string             indent( "  " );
  itk::OpenCLDevice::DeviceType deviceType = itk::OpenCLDevice::Default;
  if( openCLDeviceType == "GPU" )
  {
    deviceType = itk::OpenCLDevice::GPU;
  }
  else if( openCLDeviceType == "CPU" )
  {
    deviceType = itk::OpenCLDevice::CPU;
  }
  else
  {
    std::stringstream errorMessageStream;
    errorMessageStream << "ERROR: You have selected the OpenCL device type: " << openCLDeviceType << std::endl
                       << indent << "This type is not supported by the elastix. " << std::endl
                       << indent << "Please provide the correct OpenCL device type (GPU, CPU) "
                       << "using the (OpenCLDeviceType \"\") option." << std::endl;
    errorMessage = errorMessageStream.str();

    return false;
  }

  /** Get a list of all OpenCL devices. */
  std::list< itk::OpenCLDevice >       devicesByType;
  const std::list< itk::OpenCLDevice > allDevices = itk::OpenCLDevice::GetAllDevices();
  for( std::list< itk::OpenCLDevice >::const_iterator device = allDevices.begin(); device != allDevices.end(); ++device )
  {
    if( ( ( *device ).GetDeviceType() & deviceType ) != 0 )
    {
      devicesByType.push_back( *device );
    }
  }

  /** Check if user provided the correct OpenCL device ID. */
  if( ( openCLDeviceID < 0 ) || ( openCLDeviceID > static_cast< int >( devicesByType.size() ) - 1 ) )
  {
    const std::string s = ( devicesByType.size() > 1 ) ? "s" : "";
    std::stringstream errorMessageStream;
    errorMessageStream << "ERROR: You have selected the OpenCL device ID: " << openCLDeviceID
                       << ", with (OpenCLDeviceID \"" << openCLDeviceID << "\") option." << std::endl
                       << indent << "There are only " << devicesByType.size() << " "
                       << openCLDeviceType << " OpenCL-enabled device" << s << " present on this system:" << std::endl;

    unsigned int deviceID = 0;
    for( std::list< itk::OpenCLDevice >::const_iterator device = devicesByType.begin(); device != devicesByType.end(); ++device )
    {
      errorMessageStream << indent << "OpenCL device ID: " << deviceID << std::endl;
      errorMessageStream << indent << indent << "Name: " << ( *device ).GetName() << std::endl;
      errorMessageStream << indent << indent << "Vendor: " << ( *device ).GetVendor() << std::endl;
      errorMessageStream << indent << indent << "Has double support: " << ( ( *device ).HasDouble() ? "Yes" : "No" ) << std::endl;
      errorMessageStream << indent << indent << "Device type: ";
      switch( ( *device ).GetDeviceType() )
      {
        case OpenCLDevice::Default:
          errorMessageStream << "Default"; break;
        case OpenCLDevice::CPU:
          errorMessageStream << "CPU"; break;
        case OpenCLDevice::GPU:
          errorMessageStream << "GPU"; break;
        case OpenCLDevice::Accelerator:
          errorMessageStream << "Accelerator"; break;
        case OpenCLDevice::All:
          errorMessageStream << "All"; break;
        default:
          errorMessageStream << "Unknown"; break;
      }
      errorMessageStream << std::endl << indent << "elastix option: "
                         << "(OpenCLDeviceID \"" << deviceID << "\")" << std::endl;
      ++deviceID;
    }

    errorMessageStream << std::endl << indent << "Please provide the correct "
                       << openCLDeviceType << " OpenCL device ID using the (OpenCLDeviceID \"\") option." << std::endl;
    errorMessage = errorMessageStream.str();

    return false;
  }

  /** Select the OpenCL device ID. The operator[] does not exist for std::list.
   * We have to loop over devicesByType and select it. */
  std::list< itk::OpenCLDevice > selected;
  int                            deviceID = 0;
  for( std::list< OpenCLDevice >::const_iterator device = devicesByType.begin();
    device != devicesByType.end(); ++device )
  {
    {
      if( deviceID == openCLDeviceID )
      {
        selected.push_back( *device );
        break;
      }
      ++deviceID;
    }
  }

  /** Create OpenCL context that matches selected device. */
  if( selected.size() == 1 )
  {
    context->Create( selected );
  }

  /** Check that OpenCL context has been created. */
  if( !context->IsCreated() )
  {
    std::stringstream errorMessageStream;
    errorMessageStream << "ERROR: You have requested (OpenCLDeviceType \"" << openCLDeviceType << "\") option, but "
                       << openCLDeviceType << " OpenCL-enabled device is not present on this system!" << std::endl;

    // Add recommendations where to get OpenCL drivers
    errorMessageStream << indent << "For NVIDIA graphical cards (OpenCLDeviceType \"GPU\") option, download OpenCL drivers from:" << std::endl
                       << indent << "http://www.nvidia.com/Download/index.aspx" << std::endl;

    errorMessageStream << indent << "For AMD processors (OpenCLDeviceType \"CPU\") option or "
                       << "graphical cards (OpenCLDeviceType \"GPU\") option, download OpenCL drivers from:" << std::endl
                       << indent << "http://support.amd.com/en-us/download" << std::endl;

    errorMessageStream << indent << "For Intel processors (OpenCLDeviceType \"CPU\") option or "
                       << "HD graphical cards (OpenCLDeviceType \"GPU\") option, download OpenCL drivers from:" << std::endl
                       << indent << "https://software.intel.com/en-us/intel-opencl/download" << std::endl;

    errorMessage = errorMessageStream.str();
    return false;
  }

  /** Check for the device 'double' support. The check has 'double' ensures that
   * the OpenCL device is from around 2009 with some decent support for the OpenCL.
   * For NVIDIA that is generation since GT200 and later.
   * for AMD that is generation since HD 4730, 5830, 6930, 7730, R7 240 and later.
   * We are making it minimum requirements for elastix with OpenCL for now.
   * Although this check is too strict for the Intel HD GPU which only supports float (2014).
   * The support for the Intel HD has to be investigated, while Intel OpenCL CPU should work. */
  if( !context->GetDefaultDevice().HasDouble() )
  {
    std::stringstream errorMessageStream;
    errorMessageStream << "ERROR: OpenCL device: " << context->GetDefaultDevice().GetName()
                       << ", does not support 'double' computations." << std::endl
                       << indent << "OpenCL processing in elastix is disabled, since 'double' support is currently required. "
                       << "Processing will be performed on the CPU instead." << std::endl
                       << indent << "You may consider upgrading your graphical card (hardware)." << std::endl;
    errorMessage = errorMessageStream.str();

    return false;
  }

  return true;
} // end CreateOpenCLContext()


//------------------------------------------------------------------------------
void
CreateOpenCLLogger( const std::string & prefixFileName, const std::string & outputDirectory )
{
  /** Create the OpenCL logger */
  itk::OpenCLLogger::Pointer logger = itk::OpenCLLogger::GetInstance();
  logger->SetLogFileNamePrefix( prefixFileName );
  logger->SetOutputDirectory( outputDirectory );
} // end CreateOpenCLLogger()


} // end namespace itk
