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

#include "itkTransformixFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkParameterFileParser.h"
#include "elxParameterObject.h"

//-------------------------------------------------------------------------------------

int
main(int argc, char * argv[])
{
  /** Some basic type definitions. */
  constexpr unsigned int Dimension = 3;
  using PixelType = float;

  /** Check. */
  if (argc != 4)
  {
    std::cerr << "ERROR: Usage: " << argv[0] << " <movingImage> <transformParameters> <transformedImage>" << std::endl;
    return 1;
  }
  const char * movingImageFile = argv[1];
  const char * transformParametersFile = argv[2];
  const char * transformedImageFile = argv[3];

  /** Other typedef. */
  using ImageType = itk::Image<PixelType, Dimension>;

  auto movingImage = itk::ReadImage<ImageType>(movingImageFile);

  auto parameterObject = elastix::ParameterObject::New();
  parameterObject->ReadParameterFile(transformParametersFile);
  parameterObject->Print(std::cout);

  using TransformixType = itk::TransformixFilter<ImageType>;
  auto transformix = TransformixType::New();
  transformix->SetMovingImage(movingImage);
  transformix->SetTransformParameterObject(parameterObject);
  transformix->SetLogToConsole(true);
  ImageType::Pointer resultImage = transformix->GetOutput();

  itk::WriteImage(resultImage, transformedImageFile);

  /** Return a value. */
  return 0;

} // end main
