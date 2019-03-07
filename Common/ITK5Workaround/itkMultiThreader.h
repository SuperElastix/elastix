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

#ifndef elxITK5Workaround_itkMultiThreader_h
#define elxITK5Workaround_itkMultiThreader_h

// This header file provides a workaround to still allow doing
// #include "itkMultiThreader.h" with ITK >= 5.

#include "itkPlatformMultiThreader.h"

using itk::ITK_THREAD_RETURN_TYPE;

const ITK_THREAD_RETURN_TYPE ITK_THREAD_RETURN_VALUE = ITK_THREAD_RETURN_DEFAULT_VALUE;

namespace itk
{
  typedef PlatformMultiThreader MultiThreader;
}

#endif
