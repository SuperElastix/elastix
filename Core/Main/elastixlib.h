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
#ifndef __elastixlib_h
#define __elastixlib_h

/*
 *  Includes
 */
#include <itkDataObject.h>
#include "itkParameterFileParser.h"
#include "elxMacro.h"
#include "elxElastixMain.h"


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
  typedef itk::DataObject Image;
  typedef Image::Pointer  ImagePointer;

  //typedefs for parameter map
  typedef itk::ParameterFileParser::ParameterValuesType             ParameterValuesType;
  typedef itk::ParameterFileParser::ParameterMapType                ParameterMapType;
  typedef std::vector< itk::ParameterFileParser::ParameterMapType > ParameterMapListType;

  //typedefs for ObjectPointer
  typedef elastix::ElastixMain::ObjectPointer              ObjectPointer;

  /**
   *  Constructor and destructor
   */
  ELASTIX();
  virtual ~ELASTIX();

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
    const ParameterMapType & parameterMap,
    const std::string & outputPath,
    bool performLogging,
    bool performCout,
    ImagePointer fixedMask = nullptr,
    ImagePointer movingMask = nullptr );

  int RegisterImages( ImagePointer fixedImage,
    ImagePointer movingImage,
    const std::vector< ParameterMapType > & parameterMaps,
    const std::string & outputPath,
    bool performLogging,
    bool performCout,
    ImagePointer fixedMask = nullptr,
    ImagePointer movingMask = nullptr,
    ObjectPointer transform = nullptr);

  /** Getter for result image. */
  ImagePointer GetResultImage( void );

  /** Get transform parameters of last registration step. */
  ParameterMapType GetTransformParameterMap( void );

  /** Get transform parameters of all registration steps. */
  ParameterMapListType GetTransformParameterMapList( void );

private:

  /* the result images */
  ImagePointer m_ResultImage;

  /* Final transformation*/
  ParameterMapListType m_TransformParametersList;

};

// end class ELASTIX

} // end namespace elastix

#endif
