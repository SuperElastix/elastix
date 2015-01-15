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
#ifndef __transformixlib_h
#define __transformixlib_h

/**
 *  Includes
 */
#include <itkDataObject.h>
#include "itkParameterFileParser.h"

/********************************************************************************
 *                    *
 *      Dll export    *
 *                    *
 ********************************************************************************/
#if ( defined( _WIN32 ) || defined( WIN32 ) )
#  ifdef _ELASTIX_BUILD_LIBRARY
#    ifdef _ELASTIX_BUILD_SHARED_LIBRARY
#      define TRANSFORMIXLIB_API __declspec( dllexport )
#    else
#      define TRANSFORMIXLIB_API __declspec( dllimport )
#    endif
#  else
#    define TRANSFORMIXLIB_API __declspec( dllexport )
#  endif
#else
#  if __GNUC__ >= 4
#    define TRANSFORMIXLIB_API __attribute__ ( ( visibility( "default" ) ) )
#  else
#    define TRANSFORMIXLIB_API
#  endif
#endif

/********************************************************************************
 *                          *
 *      Class definition    *
 *                          *
 ********************************************************************************/
namespace transformix
{

class TRANSFORMIXLIB_API TRANSFORMIX
{
public:

  //typedefs for images
  typedef itk::DataObject Image;
  typedef Image::Pointer  ImagePointer;

  //typedefs for parameter map
  typedef itk::ParameterFileParser::ParameterValuesType             ParameterValuesType;
  typedef itk::ParameterFileParser::ParameterMapType                ParameterMapType;
  typedef std::vector< itk::ParameterFileParser::ParameterMapType > ParameterMapListType;

  /** Constructor and destructor. */
  TRANSFORMIX();
  virtual ~TRANSFORMIX();

  /** Return value: 0 is success in case not 0 an error occurred
   *    0 = success
   *    1 = error
   *   -2 = output folder does not exist
   */
  int TransformImage( ImagePointer inputImage,
    ParameterMapType & parameterMap,
    std::string outputPath,
    bool performLogging,
    bool performCout );

  /** Return value: 0 is success in case not 0 an error occurred
   *    0 = success
   *    1 = error
   *   -2 = output folder does not exist
   */
  int TransformImage( ImagePointer inputImage,
    std::vector< ParameterMapType > & parameterMaps,
    std::string outputPath,
    bool performLogging,
    bool performCout );

  /** Getter for result image. */
  ImagePointer GetResultImage( void );

private:

  ImagePointer m_ResultImage;

};

// end class TRANSFORMIX

} // namespace transformix

#endif // end #ifndef __transformixlib_h
