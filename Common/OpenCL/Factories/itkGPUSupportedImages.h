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
#ifndef itkGPUSupportedImages_h
#define itkGPUSupportedImages_h

#include "itkMacro.h"
#include "TypeList.h"

namespace itk
{
/** \struct OpenCLDefaultImageDimentions
 * \brief Default OpenCL image dimensions support struct.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
struct OpenCLDefaultImageDimentions
{
  itkStaticConstMacro(Support1D, bool, true);
  itkStaticConstMacro(Support2D, bool, true);
  itkStaticConstMacro(Support3D, bool, true);
};

// Default OpenCL supported image types
typedef typelist::MakeTypeList<unsigned char, char, unsigned short, short, unsigned int, int, float, double>::Type
  OpenCLDefaultImageTypes;
} // namespace itk

#endif // end #ifndef itkGPUSupportedImages_h
