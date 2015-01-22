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

namespace itk
{
//------------------------------------------------------------------------------
bool
CreateOpenCLContext( std::string & errorMessage )
{
  /** Create the OpenCL context */
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  if( !context->IsCreated() )
  {
#if defined( OPENCL_USE_INTEL_CPU ) || defined( OPENCL_USE_AMD_CPU )
    context->Create( itk::OpenCLContext::DevelopmentSingleMaximumFlopsDevice );
#else
    context->Create( itk::OpenCLContext::SingleMaximumFlopsDevice );
#endif

    if( !context->IsCreated() )
    {
      errorMessage = "ERROR: OpenCL-enabled device is not present!";
      return true;
    }

    /** Check for the device 'double' support. The check has 'double' ensures that
     * the OpenCL device is from around 2009 with some decent support for the OpenCL.
     * For NVIDIA that is generation since GT200 and later.
     * for ATI that is generation since HD 4730, 5830, 6930, 7730, R7 240 and later.
     * We are making it minimum requirements for elastix with OpenCL for now.
     * Although this check is too strict for the Intel HD GPU which only supports float (2014).
     * The support for the Intel HD has to be investigated, while Intel OpenCL CPU should work. */
    if( !context->GetDefaultDevice().HasDouble() )
    {
      errorMessage = "ERROR: OpenCL device: " + context->GetDefaultDevice().GetName()
        + ", does not support 'double' computations. Consider updating it.";
      return true;
    }
  }
  return false;
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
