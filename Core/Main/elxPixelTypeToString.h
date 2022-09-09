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

namespace elastix
{
// Helper function (template) for writing pixel types as strings to parameter files
// `PixelTypeToString()` is only supported for the template specializations below here.
template <typename>
constexpr const char *
PixelTypeToString() = delete;


template <>
constexpr const char *
PixelTypeToString<char>()
{
  return "char";
}

template <>
constexpr const char *
PixelTypeToString<unsigned char>()
{
  return "unsigned char";
}

template <>
constexpr const char *
PixelTypeToString<short>()
{
  return "short";
}

template <>
constexpr const char *
PixelTypeToString<unsigned short>()
{
  return "unsigned short";
}

template <>
constexpr const char *
PixelTypeToString<int>()
{
  return "int";
}

template <>
constexpr const char *
PixelTypeToString<unsigned int>()
{
  return "unsigned int";
}

template <>
constexpr const char *
PixelTypeToString<long>()
{
  return "long";
}

template <>
constexpr const char *
PixelTypeToString<unsigned long>()
{
  return "unsigned long";
}

template <>
constexpr const char *
PixelTypeToString<float>()
{
  return "float";
}

template <>
constexpr const char *
PixelTypeToString<double>()
{
  return "double";
}

} // namespace elastix

#endif // elxPixelType_h
