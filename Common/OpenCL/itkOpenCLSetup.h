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
#ifndef itkOpenCLSetup_h
#define itkOpenCLSetup_h

#include <string>

/** This file contains helper functionality to enable
 * OpenCL support within elastix and transformix.
 */
namespace itk
{
/** Method that is used to create OpenCL context within elastix and transformix. */
bool
CreateOpenCLContext(std::string & errorMessage, const std::string openCLDeviceType, const int openCLDeviceID);

/** Method that is used to create OpenCL logger within elastix and transformix. */
void
CreateOpenCLLogger(const std::string & prefixFileName, const std::string & outputDirectory);

} // end namespace itk

#endif
