/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elastixlib_h
#define __elastixlib_h

/*
 *  Includes
 */
#include <itkDataObject.h>
#include "itkParameterFileParser.h"

/********************************************************************************
 *                    *
 *      Dll export    *
 *                    *
 ********************************************************************************/
#if (defined(_WIN32) || defined(WIN32) )
#  ifdef _ELASTIX_BUILD_LIBRARY
#    ifdef _ELASTIX_BUILD_SHARED_LIBRARY
#      define ELASTIXLIB_API __declspec(dllexport)
#    else
#      define ELASTIXLIB_API __declspec(dllimport)
#    endif
#  else
#    define ELASTIXLIB_API __declspec(dllexport)
#  endif
#else
#  if __GNUC__ >= 4
#    define ELASTIXLIB_API __attribute__ ((visibility ("default")))
#  else
#    define ELASTIXLIB_API
#  endif
#endif

/********************************************************************************
 *                          *
 *      Class definition    *
 *                          *
 ********************************************************************************/

namespace elastix
{

class ELASTIXLIB_API ELASTIX
{
public:
  //typedefs for images
  typedef itk::DataObject   Image;
  typedef Image::Pointer    ImagePointer;

  //typedefs for parameter map
  typedef itk::ParameterFileParser::ParameterValuesType ParameterValuesType;
  typedef itk::ParameterFileParser::ParameterMapType    ParameterMapType;

  /**
   *  Constructor and destructor
   */
  ELASTIX();
  ~ELASTIX();

  /**
   *  The image registration interface functionality
   *  Note:
   *    - itk::Image::PixelType must be the same as specified in ParameterMap
   *      ('Fixed/MovingInternalImagePixelType')
   *    - Direction cosines are taken from fixed image (always set UseDirectionCosines TRUE)
   *  Params:
   *    fixedImage  itk::Image note type should be the same as specified in the Parameterfile
   *      FixedInternalImagePixelType and dimensions!
   *    movingImage itk::Image note type should be the same as specified in the Parameterfile
   *      MovingInternalImagePixelType and dimensions!
   *    ParameterMap
   *    outputPath
   *    performLogging  boolean indicating wether logging should be performed.
   *      NOTE: in case of logging also give a valid outputPath!
   *    performCout boolean indicating wether output should be send to command window
   *    fixedMask default no Mask present
   *    movingMask  default no Mask present
   *  return value: 0 is success in case not 0 an error occurred
   *     0 = success
   *     1 = error
   *    -2 = output folder does not exist
   *    \todo generate file elastix_errors.h containing error codedefines
   *      (e.g. #define ELASTIX_NO_ERROR 0)
   */
  int RegisterImages( ImagePointer fixedImage,
    ImagePointer movingImage,
    ParameterMapType& parameterMap,
    std::string   outputPath,
    bool performLogging,
    bool performCout,
    ImagePointer fixedMask = 0,
    ImagePointer movingMask = 0);

  int RegisterImages( ImagePointer fixedImage,
    ImagePointer movingImage,
    std::vector< ParameterMapType >& parameterMaps,
    std::string   outputPath,
    bool performLogging,
    bool performCout,
    ImagePointer fixedMask = 0,
    ImagePointer movingMask = 0);

  /** Getter for result image. */
  ImagePointer GetResultImage( void );

  /** Get transformparametermap */
  ParameterMapType GetTransformParameterMap( void );

private:
  /* the result images */
  ImagePointer     m_ResultImage;

  /* Final transformation*/
  ParameterMapType m_TransformParameters;

}; // end class ELASTIX

}// end namespace elastix

#endif
