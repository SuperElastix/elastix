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
#include "itkMath.h"

#include <algorithm> // For min.
#include <vector>

namespace itk
{

/** \class ImageSample
 *
 * \brief A class that defines an image sample, which is
 * the coordinates of a point and its value.
 *
 * Its constructors, assignment operators, and destructor are implicitly defaulted, following the C++ "Rule of Zero".
 */

template <typename TImage>
class ITK_TEMPLATE_EXPORT ImageSample
{
public:
  /** Typedef's. */
  using ImageType = TImage;
  using PointType = typename ImageType::PointType;
  using PixelType = typename ImageType::PixelType;
  using RealType = typename NumericTraits<PixelType>::RealType;

  /** Member variables. */
  PointType m_ImageCoordinates;
  RealType  m_ImageValue;

  friend bool
  operator==(const ImageSample & lhs, const ImageSample & rhs)
  {
    return lhs.m_ImageCoordinates == rhs.m_ImageCoordinates && Math::ExactlyEquals(lhs.m_ImageValue, rhs.m_ImageValue);
  }

  friend bool
  operator!=(const ImageSample & lhs, const ImageSample & rhs)
  {
    return !(lhs == rhs);
  }


  /** Retrieves a subrange of samples from the specified container, for the specified work unit. */
  static auto
  GetRangeOfSamples(const std::vector<ImageSample> & sampleContainer,
                    const size_t                     numberOfWorkUnits,
                    const size_t                     workUnitID)
  {
    const size_t sampleContainerSize{ sampleContainer.size() };

    // Equivalent to `ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(numberOfWorkUnits)))`.
    const size_t numberOfSamplesPerWorkUnit{ (sampleContainerSize / numberOfWorkUnits) +
                                             (sampleContainerSize % numberOfWorkUnits == 0 ? 0 : 1) };

    using IteratorType = decltype(sampleContainer.cbegin());

    struct Range
    {
      IteratorType Begin{};
      IteratorType End{};

      auto
      begin() const
      {
        return Begin;
      }

      auto
      end() const
      {
        return End;
      }
    };

    const auto beginOfSampleContainer = sampleContainer.cbegin();
    return Range{ beginOfSampleContainer + std::min(numberOfSamplesPerWorkUnit * workUnitID, sampleContainerSize),
                  beginOfSampleContainer +
                    std::min(numberOfSamplesPerWorkUnit * (workUnitID + 1), sampleContainerSize) };
  }
};


} // end namespace itk

#endif // end #ifndef itkImageSample_h
