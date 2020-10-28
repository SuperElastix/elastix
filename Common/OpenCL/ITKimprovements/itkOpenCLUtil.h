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
/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkOpenCLUtil_h
#define itkOpenCLUtil_h

#include <string.h>
#include <iostream>
#include <sstream>
#include <typeinfo>

namespace itk
{
/** Get the local block size based on the desired Image Dimension
 * currently set as follows:
 * OpenCL workgroup (block) size for 1/2/3D - needs to be tuned based on the GPU architecture
 * 1D : 256
 * 2D : 16x16 = 256
 * 3D : 4x4x4 = 64
 *
 * \note This file was taken from ITK 4.1.0.
 * It was modified by Denis P. Shamonin and Marius Staring.
 * Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands.
 * Added functionality is described in the Insight Journal paper:
 * http://hdl.handle.net/10380/3393
 *
 */
int
OpenCLGetLocalBlockSize(unsigned int ImageDim);

/** Get Typename */
std::string
GetTypename(const std::type_info & intype);

/** Get 64-bit pragma */
std::string
Get64BitPragma();

/** Get Typename in String */
void
GetTypenameInString(const std::type_info & intype, std::ostringstream & ret);

} // end of namespace itk

#endif
