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
#ifndef _itkMultiResolutionImageRegistrationMethodWithFeatures_hxx
#define _itkMultiResolutionImageRegistrationMethodWithFeatures_hxx

#include "itkMultiResolutionImageRegistrationMethodWithFeatures.h"

#include "itkContinuousIndex.h"
#include <vnl/vnl_math.h>

namespace itk
{

/*
 * ****************** CheckPyramids ******************
 */

template <typename TFixedImage, typename TMovingImage>
void
MultiResolutionImageRegistrationMethodWithFeatures<TFixedImage, TMovingImage>::CheckPyramids()
{
  /** Check if at least one of the following are provided. */
  if (this->GetFixedImage() == nullptr)
  {
    itkExceptionMacro(<< "FixedImage is not present");
  }
  if (this->GetMovingImage() == nullptr)
  {
    itkExceptionMacro(<< "MovingImage is not present");
  }
  if (this->GetFixedImagePyramid() == nullptr)
  {
    itkExceptionMacro(<< "Fixed image pyramid is not present");
  }
  if (this->GetMovingImagePyramid() == nullptr)
  {
    itkExceptionMacro(<< "Moving image pyramid is not present");
  }

  /** Check if the number if fixed/moving pyramids == nr of fixed/moving images,
   * and whether the number of fixed image regions == the number of fixed images.
   */
  if (this->GetNumberOfFixedImagePyramids() != this->GetNumberOfFixedImages())
  {
    itkExceptionMacro(<< "The number of fixed image pyramids should equal the number of fixed images");
  }
  if (this->GetNumberOfMovingImagePyramids() != this->GetNumberOfMovingImages())
  {
    itkExceptionMacro(<< "The number of moving image pyramids should equal the number of moving images");
  }
  if (this->GetNumberOfFixedImageRegions() != this->GetNumberOfFixedImages())
  {
    itkExceptionMacro(<< "The number of fixed image regions should equal the number of fixed image");
  }

} // end CheckPyramids()


} // end namespace itk

#endif // end #ifndef _itkMultiResolutionImageRegistrationMethodWithFeatures_hxx
