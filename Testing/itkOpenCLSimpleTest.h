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
#ifndef itkOpenCLSimpleTest_h
#define itkOpenCLSimpleTest_h

#include "itkMacro.h"

namespace itk
{
/** Returns the OpenCL source code for OpenCLSimpleTest1.cl kernel */
const char *
GetOpenCLSourceOfOpenCLSimpleTest1Kernel();

/** Returns the OpenCL source code for OpenCLSimpleTest2.cl kernel */
const char *
GetOpenCLSourceOfOpenCLSimpleTest2Kernel();
} // end namespace itk

#endif /* itkOpenCLSimpleTest_h */
