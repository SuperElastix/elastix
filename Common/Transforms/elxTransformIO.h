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
#ifndef elxTransformIO_h
#define elxTransformIO_h

#include <itkTransformBase.h>
#include <string>

namespace elastix
{
class BaseComponent;

class TransformIO
{
public:
  static itk::OptimizerParameters<double>
  GetParameters(const bool fixed, const itk::TransformBaseTemplate<double> & transform)
  {
    return fixed ? transform.GetFixedParameters() : transform.GetParameters();
  }

  static void
  SetParameters(const bool                                 fixed,
                itk::TransformBaseTemplate<double> & transform,
                const itk::OptimizerParameters<double> & parameters)
  {
    fixed ? transform.SetFixedParameters(parameters) : transform.SetParameters(parameters);
  }




  template <typename TElastixTransform>
  static itk::TransformBaseTemplate<double>::Pointer
  CreateCorrespondingItkTransform(const TElastixTransform & elxTransform)
  {
    return CreateCorrespondingItkTransform(
      elxTransform, TElastixTransform::FixedImageDimension, TElastixTransform::MovingImageDimension);
  }

  static
  void
  Write(const itk::TransformBaseTemplate<double> & itkTransform, const std::string & fileName);

private:
  static itk::TransformBaseTemplate<double>::Pointer
  CreateCorrespondingItkTransform(const BaseComponent & elxTransform,
                                  const unsigned        fixedImageDimension,
                                  const unsigned        movingImageDimension);
};
} // namespace elastix

#endif