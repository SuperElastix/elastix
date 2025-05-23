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
#ifndef elxPixelType_h
#define elxPixelType_h

#include <climits> // For CHAR_BIT.
#include <cstdint> // For int8_t, uint64_t, etc.
#include <string>
#include <type_traits> // For is_floating_point_v and is_unsigned_v.

namespace elastix
{
// Converts the pixel type specified by its template argument to a string, which can be used as parameter value of the
// elastix/transformix parameter ResultImagePixelType. Uses a "fixed width" format, returning a string of the form
// "intN", "uintN", or "floatN", with N being the number of bits of the specified type.
template <typename TPixel>
std::string
PixelTypeToFixedWidthString()
{
  // Check that `TPixel` can be represented losslessly as fixed-width type:
  static_assert(std::get<TPixel>(std::tuple<std::int8_t,
                                            std::uint8_t,
                                            std::int16_t,
                                            std::uint16_t,
                                            std::int32_t,
                                            std::uint32_t,
                                            std::int64_t,
                                            std::uint64_t,
                                            float,
                                            double>()) == 0);

  return (std::is_floating_point_v<TPixel> ? "float"
          : std::is_unsigned_v<TPixel>     ? "uint"
                                           : "int") +
         std::to_string(sizeof(TPixel) * CHAR_BIT);
}


// Converts the pixel type specified by its template argument to a string, which can be used as parameter value of the
// elastix/transformix parameter ResultImagePixelType.
template <typename TPixel>
std::string
PixelTypeToString()
{
  if constexpr (std::is_same_v<TPixel, char> && !std::is_same_v<char, std::int8_t> &&
                !std::is_same_v<char, std::uint8_t>)
  {
    // `char` appears different from both `int8_t` and `uint8_t`, so it must have its own string representation.
    return "char";
  }
  else
  {
    return PixelTypeToFixedWidthString<TPixel>();
  }
}

} // namespace elastix

#endif
