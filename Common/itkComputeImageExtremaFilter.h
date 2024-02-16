/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkComputeImageExtremaFilter_h
#define itkComputeImageExtremaFilter_h

#include "itkImageSink.h"
#include "itkImageMaskSpatialObject.h"

namespace itk
{
/** \class ComputeImageExtremaFilter
 * \brief Compute minimum and maximum pixel value of an Image.
 *
 * \ingroup MathematicalStatisticsImageFilters
 * \ingroup ITKImageStatistics
 */
template <typename TInputImage>
class ITK_TEMPLATE_EXPORT ComputeImageExtremaFilter : public ImageSink<TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ComputeImageExtremaFilter);

  /** Standard Self typedef */
  using Self = ComputeImageExtremaFilter;
  using Superclass = ImageSink<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ComputeImageExtremaFilter, StatisticsImageFilter);

  /** Image related typedefs. */
  using InputImagePointer = typename TInputImage::Pointer;

  using Superclass::InputImageDimension;
  using typename Superclass::InputImageRegionType;
  using PixelType = typename Superclass::InputImagePixelType;

  /** Image related typedefs. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  using ImageSpatialMaskType = ImageMaskSpatialObject<Self::ImageDimension>;
  using ImageSpatialMaskPointer = typename ImageSpatialMaskType::Pointer;
  using ImageSpatialMaskConstPointer = typename ImageSpatialMaskType::ConstPointer;
  itkSetConstObjectMacro(ImageSpatialMask, ImageSpatialMaskType);
  itkGetConstObjectMacro(ImageSpatialMask, ImageSpatialMaskType);

  PixelType
  GetMinimum() const
  {
    return m_ThreadMin;
  }

  PixelType
  GetMaximum() const
  {
    return m_ThreadMax;
  }

protected:
  ComputeImageExtremaFilter() = default;
  ~ComputeImageExtremaFilter() override = default;

  /** Initialize minimum and maximum before the threads run. */
  void
  BeforeStreamedGenerateData() override;

  /** Multi-thread version GenerateData. */
  void
  ThreadedStreamedGenerateData(const InputImageRegionType &) override;

private:
  struct MinMaxResult
  {
    PixelType Min;
    PixelType Max;
  };

  static MinMaxResult
  RetrieveMinMax(const TInputImage &, const InputImageRegionType &, const ImageSpatialMaskType *, bool);

  ImageSpatialMaskConstPointer m_ImageSpatialMask{};
  bool                         m_SameGeometry{ false };

  PixelType m_ThreadMin{ 1 };
  PixelType m_ThreadMax{ 1 };

  std::mutex m_Mutex{};
}; // end of class
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkComputeImageExtremaFilter.hxx"
#endif

#endif
