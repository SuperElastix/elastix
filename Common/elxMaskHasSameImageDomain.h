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
#include <itkImageMaskSpatialObject.h>
#include "elxDeref.h"

namespace elastix
{
/** Enum to indicate that a mask is null, has the same domain as the input image, or has a different image domain. */
enum class MaskCondition
{
  IsNull,
  HasSameImageDomain,
  HasDifferentImageDomain
};

/** Returns true, if and only if the mask has exactly the same image domain (image region, origin, spacing, direction)
 * as the specified input image. */
template <unsigned int VImageDimension>
bool
MaskHasSameImageDomain(const itk::ImageMaskSpatialObject<VImageDimension> & mask,
                       const itk::ImageBase<VImageDimension> &              inputImage)
{
  const auto & maskImage = Deref(mask.GetImage());
  return maskImage.GetLargestPossibleRegion() == inputImage.GetLargestPossibleRegion() &&
         maskImage.GetOrigin() == inputImage.GetOrigin() && maskImage.GetSpacing() == inputImage.GetSpacing() &&
         maskImage.GetDirection() == inputImage.GetDirection();
}
} // namespace elastix

#endif
