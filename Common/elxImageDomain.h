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

#ifndef itkImageDomain_h
#define itkImageDomain_h

#include <itkImageBase.h>

namespace elastix
{
template <unsigned int VImageDimension>
bool
HaveSameImageDomain(const itk::ImageBase<VImageDimension> & image1, const itk::ImageBase<VImageDimension> & image2)
{
  return image1.GetLargestPossibleRegion() == image2.GetLargestPossibleRegion() &&
         image1.GetOrigin() == image2.GetOrigin() && image1.GetSpacing() == image2.GetSpacing() &&
         image1.GetDirection() == image2.GetDirection();
}
} // namespace elastix

#endif
