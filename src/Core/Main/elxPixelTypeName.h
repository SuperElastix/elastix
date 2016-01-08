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