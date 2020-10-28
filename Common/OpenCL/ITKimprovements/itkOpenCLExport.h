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
#ifndef itkOpenCLExport_h
#define itkOpenCLExport_h

#include "itkConfigure.h"
#include "itkMacro.h"

// Setup symbol export
#define ITKOpenCL_HIDDEN ITK_ABI_HIDDEN

#if !defined(ITKSTATIC)
#  ifdef ITKOpenCL_EXPORTS
#    define ITKOpenCL_EXPORT ITK_ABI_EXPORT
#  else
#    define ITKOpenCL_EXPORT ITK_ABI_IMPORT
#  endif /* ITKOpenCL_EXPORTS */
#else
/* ITKOpenCL is build as a static lib */
#  if __GNUC__ >= 4
// Don't hide symbols in the static ITKOpenCL library in case
// -fvisibility=hidden is used
#    define ITKOpenCL_EXPORT ITK_ABI_EXPORT
#  else
#    define ITKOpenCL_EXPORT
#  endif
#endif

#endif /* itkOpenCLExport_h */
