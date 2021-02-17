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

#ifndef elxCoreMainGTestUtilities_h
#define elxCoreMainGTestUtilities_h

#include <itkImage.h>
#include <itkImageRegionRange.h>
#include <itkIndex.h>
#include <itkSize.h>

#include <algorithm> // For fill.
#include <iterator>  // For begin and end.

namespace itk
{
namespace Experimental
{
// Workaround to allow using things that may be either in itk or in itk::Experimental.
}
} // namespace itk


namespace elastix
{
namespace CoreMainGTestUtilities
{

/// Fills the specified image region with pixel values 1.
template <typename TPixel, unsigned int VImageDimension>
void
FillImageRegion(itk::Image<TPixel, VImageDimension> & image,
                const itk::Index<VImageDimension> &   regionIndex,
                const itk::Size<VImageDimension> &    regionSize)
{
  // ImageRegionRange is to be moved from namespace itk::Experimental
  // to namespace itk with ITK version 5.2.
  using namespace itk;
  using namespace itk::Experimental;

  const ImageRegionRange<Image<TPixel, VImageDimension>> imageRegionRange{
    image, ImageRegion<VImageDimension>{ regionIndex, regionSize }
  };
  std::fill(std::begin(imageRegionRange), std::end(imageRegionRange), 1);
}


} // namespace CoreMainGTestUtilities
} // namespace elastix


#endif
