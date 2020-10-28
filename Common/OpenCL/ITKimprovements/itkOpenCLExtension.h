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
#ifndef itkOpenCLExtension_h
#define itkOpenCLExtension_h

#include "itkOpenCL.h"

// This file provides standard OpenCL definitions
// that may not be present in the system headers.
// See the "cl.h" file on your system.

// OpenCL 1.1 extension definitions
// cl_khr_fp64
#ifndef CL_DEVICE_DOUBLE_FP_CONFIG
#  define CL_DEVICE_DOUBLE_FP_CONFIG 0x1032
#endif

// cl_khr_fp16
#ifndef CL_DEVICE_HALF_FP_CONFIG
#  define CL_DEVICE_HALF_FP_CONFIG 0x1033
#endif

// OpenCL 1.2 extension definitions
#ifndef CL_COMMAND_BARRIER
#  define CL_COMMAND_BARRIER 0x1205
#endif

#ifndef CL_COMMAND_MIGRATE_MEM_OBJECTS
#  define CL_COMMAND_MIGRATE_MEM_OBJECTS 0x1206
#endif

#ifndef CL_COMMAND_FILL_BUFFER
#  define CL_COMMAND_FILL_BUFFER 0x1207
#endif

#ifndef CL_COMMAND_FILL_IMAGE
#  define CL_COMMAND_FILL_IMAGE 0x1208
#endif

#ifndef CL_MEM_OBJECT_IMAGE2D_ARRAY
#  define CL_MEM_OBJECT_IMAGE2D_ARRAY 0x10F3
#endif

#ifndef CL_MEM_OBJECT_IMAGE1D
#  define CL_MEM_OBJECT_IMAGE1D 0x10F4
#endif

#ifndef CL_MEM_OBJECT_IMAGE1D_ARRAY
#  define CL_MEM_OBJECT_IMAGE1D_ARRAY 0x10F5
#endif

#ifndef CL_MEM_OBJECT_IMAGE1D_BUFFER
#  define CL_MEM_OBJECT_IMAGE1D_BUFFER 0x10F6
#endif

// cl_khr_icd
// http://www.khronos.org/registry/cl/extensions/khr/cl_khr_icd.txt
#ifndef CL_PLATFORM_ICD_SUFFIX_KHR
#  define CL_PLATFORM_ICD_SUFFIX_KHR 0x0920
#endif
#ifndef CL_PLATFORM_NOT_FOUND_KHR
#  define CL_PLATFORM_NOT_FOUND_KHR -1001
#endif

#endif /* itkOpenCLExtension_h */
