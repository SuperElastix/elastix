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
#ifndef elxParameterObject_h
#define elxParameterObject_h

#include "itkObjectFactory.h"
#include "itkDataObject.h"
#include "elxMacro.h"

#include "itkParameterFileParser.h"

namespace elastix
{

// TODO: Why does the compiler not see ELASTIXLIB_API declspec in elxMacro.h?
//   error: variable has incomplete type 'class ELASTIXLIB_API'
// with class ELASTIXLIB_API ParameterObject : public itk::DataObject

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
  typedef ParameterValueVectorType::iterator                      ParameterValueVectorIterator;
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
  void SetParameterMap( const std::string transformName, const unsigned int numberOfResolutions = 4u, const double finalGridSpacingInPhysicalUnits = 10.0 );
  void AddParameterMap( const std::string transformName, const unsigned int numberOfResolutions = 4u, const double finalGridSpacingInPhysicalUnits = 10.0 );
  ParameterMapType GetParameterMap( const std::string transformName, const unsigned int numberOfResolutions = 4u, const double finalGridSpacingInPhysicalUnits = 10.0 );

  // C++11 has a to_string in the standard library, but we avoid
  // introducing C++11 dependency for this function
  template< typename T >
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
