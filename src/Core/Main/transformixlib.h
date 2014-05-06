/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
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
