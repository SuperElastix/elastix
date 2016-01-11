#ifndef elxParameterObject_h
#define elxParameterObject_h

#include "itkObjectFactory.h"
#include "itkDataObject.h"

#include "itkParameterFileParser.h"

namespace elastix {

class ParameterObject : public itk::DataObject
{
public:
  typedef ParameterObject                 Self;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;
  itkNewMacro( Self );
  itkTypeMacro( Self, itk::DataObject );

  typedef std::string                                             ParameterKeyType;
  typedef std::string                                             ParameterValueType;
  typedef std::vector< ParameterValueType >                       ParameterValueVectorType;
  typedef std::map< ParameterKeyType, ParameterValueVectorType >  ParameterMapType;
  typedef ParameterMapType::iterator                              ParameterMapIterator;
  typedef ParameterMapType::const_iterator                        ParameterMapConstIterator;
  typedef std::vector< ParameterMapType >                         ParameterMapVectorType;
  typedef std::string                                             ParameterFileNameType;
  typedef std::vector< ParameterFileNameType >                    ParameterFileNameVectorType;
  typedef ParameterFileNameVectorType::iterator                   ParameterFileNameVectorIterator;
  typedef ParameterFileNameVectorType::const_iterator             ParameterFileNameVectorConstIterator;

  typedef itk::ParameterFileParser                                ParameterFileParserType;
  typedef ParameterFileParserType::Pointer                        ParameterFileParserPointer;                    

  void SetParameterMap( const ParameterMapType parameterMap );
  void SetParameterMap( const ParameterMapVectorType parameterMapVector );
  void AddParameterMap( const ParameterMapType parameterMap );
  
  ParameterMapType& GetParameterMap( unsigned int index );
  ParameterMapVectorType& GetParameterMap( void );
  const ParameterMapVectorType& GetParameterMap( void ) const;  

  void ReadParameterFile( const ParameterFileNameType parameterFileName );
  void ReadParameterFile( const ParameterFileNameVectorType parameterFileNameVector );
  void AddParameterFile( const ParameterFileNameType parameterFileName );

  void WriteParameterFile( const ParameterMapType parameterMap, const ParameterFileNameType parameterFileName );
  void WriteParameterFile( const ParameterFileNameType parameterFileName );
  void WriteParameterFile( const ParameterFileNameVectorType parameterFileNameVector );

  // Default parameter maps
  void SetParameterMap( const std::string transformName, const unsigned int numberOfResolutions = 3u, const double finalGridSpacingInPhysicalUnits = 10.0 );
  void AddParameterMap( const std::string transformName, const unsigned int numberOfResolutions = 3u, const double finalGridSpacingInPhysicalUnits = 10.0 );
  ParameterMapType GetParameterMap( const std::string transformName, const unsigned int numberOfResolutions = 3u, const double finalGridSpacingInPhysicalUnits = 10.0 );

  // C++11 has a to_string in the standard library, but it is undesirable to 
  // introduce a C++11 dependency for a simple number to string helper function
  template < typename T > 
  static std::string ToString( const T& n )
  {
    std::ostringstream stm;
    stm << n;
    return stm.str();
  }

private:

  ParameterMapVectorType  m_ParameterMapVector;

};

} // namespace elx

#endif // elxParameterObject_h