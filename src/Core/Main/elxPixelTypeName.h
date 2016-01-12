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
#ifndef elxPixelTypeName_h
#define elxPixelTypeName_h

namespace elastix {
// PixelType traits for writing types as strings to parameter files

// Default implementation
template < typename T >
struct PixelTypeName
{
  static const char* ToString()
  {
     itkGenericExceptionMacro( "Pixel type \"" << typeid( T ).name() << "\" is not supported." )
  }
};

template <>
struct PixelTypeName< char >
{
  static const char* ToString()
  {
    return "char";
  }
};

template <>
struct PixelTypeName< unsigned char >
{
  static const char* ToString()
  {
    return "unsigned char";
  }
};

template <>
struct PixelTypeName< short >
{
  static const char* ToString()
  {
    return "short";
  }
};

template <>
struct PixelTypeName< unsigned short >
{
  static const char* ToString()
  {
    return "unsigned short";
  }
};

template <>
struct PixelTypeName< int >
{
  static const char* ToString()
  {
    return "int";
  }
};

template <>
struct PixelTypeName< unsigned int >
{
  static const char* ToString()
  {
    return "unsigned int";
  }
};

template <>
struct PixelTypeName< long >
{
  static const char* ToString()
  {
    return "long";
  }
};

template <>
struct PixelTypeName< unsigned long >
{
  static const char* ToString()
  {
    return "unsigned long";
  }
};

template <>
struct PixelTypeName< float >
{
  static const char* ToString()
  {
    return "float";
  }
};

template <>
struct PixelTypeName< double >
{
  static const char* ToString()
  {
    return "double";
  }
};

}

#endif // elxPixelTypeName_h
