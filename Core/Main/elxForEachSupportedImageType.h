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

#ifndef elxForEachSupportedImageType_h
#define elxForEachSupportedImageType_h

#include <elxSupportedImageTypes.h>

#include <utility> // For index_sequence.


namespace elastix
{

template <typename TFunction, std::size_t... VIndexSequence>
void
ForEachSupportedImageType(const TFunction & func, const std::index_sequence<VIndexSequence...> &)
{
  struct Empty
  {};

  const auto supportImageTypeWithIndex = [&func](const auto index) {
    // Use `index + 1` instead of `index`, because the indices from SupportedImageTypes start with 1, while the sequence
    // returned by `std::make_index_sequence()` starts with zero.
    func(elx::ElastixTypedef<index + 1>{});
    return Empty{};
  };

  // Expand the "variadic template" index sequence by the initialization of a dummy array.
  const Empty dummyArray[] = { supportImageTypeWithIndex(std::integral_constant<std::size_t, VIndexSequence>{})... };
  (void)dummyArray;
}


/** Runs a function `func(ElastixTypedef<VIndex>{})`, for each supported image type from "elxSupportedImageTypes.h". */
template <typename TFunction>
void
ForEachSupportedImageType(const TFunction & func)

{
  ForEachSupportedImageType(func, std::make_index_sequence<elx::NrOfSupportedImageTypes>());
}

} // namespace elastix


#endif
