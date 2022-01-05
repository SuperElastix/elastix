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
#ifndef transformixlib_h
#define transformixlib_h

/**
 *  Includes
 */
#include <itkDataObject.h>
#include "itkParameterFileParser.h"
#include "elxMacro.h"

/********************************************************************************
 *                          *
 *      Class definition    *
 *                          *
 ********************************************************************************/
namespace transformix
{

class ELASTIXLIB_API TRANSFORMIX
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

  /** Constructor and destructor. */
  TRANSFORMIX();
  virtual ~TRANSFORMIX();

  /** Return value: 0 is success in case not 0 an error occurred
   *    0 = success
   *    1 = error
   *   -2 = output folder does not exist
   */
  int
  TransformImage(ImagePointer       inputImage,
                 ParameterMapType & parameterMap,
                 std::string        outputPath,
                 bool               performLogging,
                 bool               performCout);

  /** Return value: 0 is success in case not 0 an error occurred
   *    0 = success
   *    1 = error
   *   -2 = output folder does not exist
   */
  int
  TransformImage(ImagePointer                    inputImage,
                 std::vector<ParameterMapType> & parameterMaps,
                 const std::string &             outputPath,
                 bool                            performLogging,
                 bool                            performCout);

  /** Getter for result image. */
  ConstImagePointer
  GetResultImage() const;

  /** Getter for result image. Non-const overload */
  ImagePointer
  GetResultImage();

private:
  ImagePointer m_ResultImage;
};

// end class TRANSFORMIX

} // namespace transformix

#endif // end #ifndef transformixlib_h
