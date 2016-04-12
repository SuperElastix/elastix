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

namespace elastix {
// PixelType traits for writing types as strings to parameter files

// Default implementation
template < typename T >
struct PixelType
{
  static const char* ToString()
  {
     itkGenericExceptionMacro( "Pixel type \"" << typeid( T ).name() << "\" is not supported." )
  }
};

template <>
struct PixelType< char >
{
  static const char* ToString()
  {
    return "char";
  }
};

template <>
struct PixelType< unsigned char >
{
  static const char* ToString()
  {
    return "unsigned char";
  }
};

template <>
struct PixelType< short >
{
  static const char* ToString()
  {
    return "short";
  }
};

template <>
struct PixelType< unsigned short >
{
  static const char* ToString()
  {
    return "unsigned short";
  }
};

template <>
struct PixelType< int >
{
  static const char* ToString()
  {
    return "int";
  }
};

template <>
struct PixelType< unsigned int >
{
  static const char* ToString()
  {
    return "unsigned int";
  }
};

template <>
struct PixelType< long >
{
  static const char* ToString()
  {
    return "long";
  }
};

template <>
struct PixelType< unsigned long >
{
  static const char* ToString()
  {
    return "unsigned long";
  }
};

template <>
struct PixelType< float >
{
  static const char* ToString()
  {
    return "float";
  }
};

template <>
struct PixelType< double >
{
  static const char* ToString()
  {
    return "double";
  }
};

}

#endif // elxPixelType_h
