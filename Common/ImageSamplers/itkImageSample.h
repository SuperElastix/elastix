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
#ifndef itkImageSample_h
#define itkImageSample_h

#include "itkNumericTraits.h"

namespace itk
{

/** \class ImageSample
 *
 * \brief A class that defines an image sample, which is
 * the coordinates of a point and its value.
 *
 */

template <class TImage>
class ITK_TEMPLATE_EXPORT ImageSample
{
public:
  // ImageSample():m_ImageValue(0.0){};
  ImageSample() = default;
  ~ImageSample() = default;

  /** Typedef's. */
  typedef TImage                                      ImageType;
  typedef typename ImageType::PointType               PointType;
  typedef typename ImageType::PixelType               PixelType;
  typedef typename NumericTraits<PixelType>::RealType RealType;

  /** Member variables. */
  PointType m_ImageCoordinates;
  RealType  m_ImageValue;
};

} // end namespace itk

#endif // end #ifndef itkImageSample_h
