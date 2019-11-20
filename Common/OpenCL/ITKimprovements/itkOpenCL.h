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
#ifndef itkOpenCL_h
#define itkOpenCL_h

#include "itkOpenCLExport.h"

#ifndef CL_TARGET_OPENCL_VERSION
#  define CL_TARGET_OPENCL_VERSION 120
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#  include <OpenCL/cl_platform.h>
#  include <OpenCL/cl.h>
#else
#  include <CL/cl_platform.h>
#  include <CL/cl.h>
#endif

namespace itk
{
/** \enum OpenCLVersion
 * This enum defines bits corresponding to OpenCL versions.
 * \value VERSION_1_0 OpenCL 1.0 is supported.
 * \value VERSION_1_1 OpenCL 1.1 is supported.
 * \value VERSION_1_2 OpenCL 1.2 is supported.
 * \value VERSION_2_0 OpenCL 2.0 is supported.
 * \value VERSION_2_1 OpenCL 2.1 is supported.
 */
enum OpenCLVersion
{
  VERSION_1_0 = 0x0001,
  VERSION_1_1 = 0x0002,
  VERSION_1_2 = 0x0003,
  VERSION_2_0 = 0x0004,
  VERSION_2_1 = 0x0005
};
} // end namespace itk

#endif /* itkOpenCL_h */
