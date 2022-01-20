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
#ifndef elxSupportedImageDimensions_h
#define elxSupportedImageDimensions_h

#include <elxSupportedImageTypes.h>
#include <utility> // For index_sequence


namespace elastix
{

/** The minimum possible supported image dimension (for either fixed or moving images). */
constexpr unsigned minSupportedImageDimension{ 2 };

/** The maximum possible supported image dimension (for either fixed or moving images). */
constexpr unsigned maxSupportedImageDimension{ 4 };


template <unsigned VDimension, std::size_t... VIndex>
constexpr bool
SupportsFixedDimensionByImageTypeIndexSequence(std::index_sequence<VIndex...>)
{
  const bool foundEntries[] = { (ElastixTypedef<VIndex + 1>::FixedDimension == VDimension)... };

  for (const bool isDimensionFound : foundEntries)
  {
    if (isDimensionFound)
    {
      return true;
    }
  }
  return false;
}


// Tells whether any of the supported fixed image types has the specified dimension.
template <unsigned VDimension>
constexpr bool
SupportsFixedDimension()
{
  return SupportsFixedDimensionByImageTypeIndexSequence<VDimension>(
    std::make_index_sequence<NrOfSupportedImageTypes>());
}


template <unsigned VDimension = minSupportedImageDimension>
struct FixedImageDimensionSupport
{
  // Adds those dimensions from the specified `dimensionSequence` that are supported.
  template <std::size_t... VIndex>
  constexpr static auto
  AddSupportedDimensions(std::index_sequence<VIndex...> dimensionSequence)
  {
    using AddDimensionIfSupported = std::conditional_t<SupportsFixedDimension<VDimension>(),
                                                       std::index_sequence<VDimension, VIndex...>,
                                                       std::index_sequence<VIndex...>>;

    return FixedImageDimensionSupport<VDimension + 1>::AddSupportedDimensions(AddDimensionIfSupported());
  }
};


template <>
struct FixedImageDimensionSupport<maxSupportedImageDimension + 1>
{
  template <std::size_t... VIndex>
  constexpr static auto
  AddSupportedDimensions(std::index_sequence<VIndex...> dimensionSequence)
  {
    return dimensionSequence;
  }
};


/** A sequence of the dimensions of supported fixed images */
const auto SupportedFixedImageDimensionSequence =
  FixedImageDimensionSupport<>::AddSupportedDimensions(std::index_sequence<>());

} // namespace elastix

#endif
