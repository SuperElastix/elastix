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
#ifndef elastixlib_h
#define elastixlib_h

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
  // typedefs for images
  using Image = itk::DataObject;
  using ImagePointer = Image::Pointer;
  using ConstImagePointer = Image::ConstPointer;

  // typedefs for parameter map
  using ParameterValuesType = itk::ParameterFileParser::ParameterValuesType;
  using ParameterMapType = itk::ParameterFileParser::ParameterMapType;
  using ParameterMapListType = std::vector<itk::ParameterFileParser::ParameterMapType>;

  // typedefs for ObjectPointer
  using ObjectPointer = elastix::ElastixMain::ObjectPointer;

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
   *      (e.g. \#define ELASTIX_NO_ERROR 0)
   */
  int
  RegisterImages(ImagePointer             fixedImage,
                 ImagePointer             movingImage,
                 const ParameterMapType & parameterMap,
                 const std::string &      outputPath,
                 bool                     performLogging,
                 bool                     performCout,
                 ImagePointer             fixedMask = nullptr,
                 ImagePointer             movingMask = nullptr);

  int
  RegisterImages(ImagePointer                          fixedImage,
                 ImagePointer                          movingImage,
                 const std::vector<ParameterMapType> & parameterMaps,
                 const std::string &                   outputPath,
                 bool                                  performLogging,
                 bool                                  performCout,
                 ImagePointer                          fixedMask = nullptr,
                 ImagePointer                          movingMask = nullptr,
                 ObjectPointer                         transform = nullptr);

  /** Getter for result image. */
  ConstImagePointer
  GetResultImage() const;

  /** Getter for result image. Non-const overload */
  ImagePointer
  GetResultImage();

  /** Get transform parameters of last registration step. */
  ParameterMapType
  GetTransformParameterMap() const;

  /** Get transform parameters of all registration steps. */
  ParameterMapListType
  GetTransformParameterMapList() const;

private:
  /* the result images */
  ImagePointer m_ResultImage;

  /* Final transformation*/
  ParameterMapListType m_TransformParametersList;
};

// end class ELASTIX

} // end namespace elastix

#endif
