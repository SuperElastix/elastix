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
#ifndef elxPrepareImageTypeSupport_h
#define elxPrepareImageTypeSupport_h

#include "elxElastixTemplate.h"

// ITK header files:
#include <itkImage.h>
#include <itkObject.h>

namespace elastix
{

/** Class template, intended to be specialized for all supported image types, by `elxSupportedImageTypeMacro` calls in
 * the CMake generated file "elxSupportedImageTypes.h"
 */
template <unsigned VIndex>
class ElastixTypedef
{
public:
  /** In the specialisations of this template class */
  /** this typedef will make sense */
  using ElastixType = ::itk::Object;
  constexpr static const char * FixedPixelTypeString{ "" };
  constexpr static const char * MovingPixelTypeString{ "" };
  constexpr static unsigned int FixedDimension{ 0 };
  constexpr static unsigned int MovingDimension{ 0 };
  /** In the specialisations of this template class, this value will be 'true' */
  constexpr static bool IsDefined{ false };
};

} // namespace elastix


/**
 * Macro for installing support for new ImageTypes.
 * Used in elxSupportedImageTypes.cxx .
 *
 * Example of usage:
 *
 * namespace elastix {
 * elxSupportedImageTypeMacro(unsigned short, 2, float, 3, 1);
 * elxSupportedImageTypeMacro(unsigned short, 3, float, 3, 2);
 * etc.
 * } //end namespace elastix
 *
 * The first line adds support for the following combination of ImageTypes:
 * fixedImage: 2D unsigned short
 * movingImage 3D float
 * The Index (last argument) of this combination of ImageTypes is 1.
 *
 * The Index should not be 0. This value is reserved for errormessages.
 * Besides, duplicate indices are not allowed.
 *
 * IMPORTANT: the macro must be invoked in namespace elastix!
 *
 * Details: the macro adds a class template specialization for the class
 * ElastixTypedef<VIndex>.
 */

#define elxSupportedImageTypeMacro(_fPixelType, _fDim, _mPixelType, _mDim, _VIndex)                                    \
  template <>                                                                                                          \
  class ElastixTypedef<_VIndex>                                                                                        \
  {                                                                                                                    \
  public:                                                                                                              \
    using FixedImageType = ::itk::Image<_fPixelType, _fDim>;                                                           \
    using MovingImageType = ::itk::Image<_mPixelType, _mDim>;                                                          \
    using ElastixType = ::elx::ElastixTemplate<FixedImageType, MovingImageType>;                                       \
    constexpr static const char * FixedPixelTypeString{ #_fPixelType };                                                \
    constexpr static const char * MovingPixelTypeString{ #_mPixelType };                                               \
    constexpr static unsigned int FixedDimension{ _fDim };                                                             \
    constexpr static unsigned int MovingDimension{ _mDim };                                                            \
    constexpr static bool         IsDefined{ true };                                                                   \
  }

#endif // end #ifndef elxPrepareImageTypeSupport_h
