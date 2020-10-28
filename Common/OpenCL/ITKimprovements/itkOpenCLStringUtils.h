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
#ifndef itkOpenCLStringUtils_h
#define itkOpenCLStringUtils_h

#include "itkOpenCL.h"

#include <list>
#include <string>

namespace itk
{
// C-style support functions
std::string
opencl_simplified(const std::string & str);

bool
opencl_has_extension(const std::string & list, const std::string & name);

// OpenCL C-style support functions
std::string
opencl_get_platform_info_string(const cl_platform_id id, const cl_platform_info name);

std::string
opencl_get_device_info_string(const cl_device_id id, const cl_device_info name);

bool
opencl_is_platform(cl_platform_id id, cl_platform_info name, const char * str);

int
opencl_version_flags(const std::string & version);

std::list<std::string>
opencl_split_string(const std::string & str, const char separator);

// OpenCL support functions to retrieve information about an OpenCL device.
unsigned int
opencl_get_device_info_uint(const cl_device_id id, const cl_device_info name);

int
opencl_get_device_info_int(const cl_device_id id, const cl_device_info name);

unsigned long
opencl_get_device_info_ulong(const cl_device_id id, const cl_device_info name);

std::size_t
opencl_get_device_info_size(const cl_device_id id, const cl_device_info name);

bool
opencl_get_device_info_bool(const cl_device_id id, const cl_device_info name);

bool
opencl_get_device_info_is_string(const cl_device_id id, const cl_device_info name, const char * str);

} // end of namespace itk

#endif /* itkOpenCLStringUtils_h */
