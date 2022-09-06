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

#include "itkStatisticsImageFilter.h"
#include "itkSpatialObject.h"
#include "itkImageMaskSpatialObject.h"

namespace itk
{
/** \class ComputeImageExtremaFilter
 * \brief Compute min. max, variance and mean of an Image.
 *
 * StatisticsImageFilter computes the minimum, maximum, sum, mean, variance
 * sigma of an image.  The filter needs all of its input image.  It
 * behaves as a filter with an input and output. Thus it can be inserted
 * in a pipline with other filters and the statistics will only be
 * recomputed if a downstream filter changes.
 *
 * The filter passes its input through unmodified.  The filter is
 * threaded. It computes statistics in each thread then combines them in
 * its AfterThreadedGenerate method.
 *
 * \ingroup MathematicalStatisticsImageFilters
 * \ingroup ITKImageStatistics
 *
 * \wiki
 * \wikiexample{Statistics/StatisticsImageFilter,Compute min\, max\, variance and mean of an Image.}
 * \endwiki
 */
template <typename TInputImage>
class ITK_TEMPLATE_EXPORT ComputeImageExtremaFilter : public StatisticsImageFilter<TInputImage>
{
public:
  /** Standard Self typedef */
  using Self = ComputeImageExtremaFilter;
  using Superclass = StatisticsImageFilter<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ComputeImageExtremaFilter, StatisticsImageFilter);

  /** Image related typedefs. */
  using InputImagePointer = typename TInputImage::Pointer;

  using typename Superclass::RegionType;
  using typename Superclass::SizeType;
  using typename Superclass::IndexType;
  using typename Superclass::PixelType;
  using PointType = typename TInputImage::PointType;

  /** Image related typedefs. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Type to use for computations. */
  using typename Superclass::RealType;

  itkSetMacro(ImageRegion, RegionType);
  itkSetMacro(UseMask, bool);

  using ImageMaskType = SpatialObject<Self::ImageDimension>;
  using ImageMaskPointer = typename ImageMaskType::Pointer;
  using ImageMaskConstPointer = typename ImageMaskType::ConstPointer;
  itkSetConstObjectMacro(ImageMask, ImageMaskType);
  itkGetConstObjectMacro(ImageMask, ImageMaskType);

  using ImageSpatialMaskType = ImageMaskSpatialObject<Self::ImageDimension>;
  using ImageSpatialMaskPointer = typename ImageSpatialMaskType::Pointer;
  using ImageSpatialMaskConstPointer = typename ImageSpatialMaskType::ConstPointer;
  itkSetConstObjectMacro(ImageSpatialMask, ImageSpatialMaskType);
  itkGetConstObjectMacro(ImageSpatialMask, ImageSpatialMaskType);

protected:
  ComputeImageExtremaFilter() = default;
  ~ComputeImageExtremaFilter() override = default;

  /** Initialize some accumulators before the threads run. */
  void
  BeforeStreamedGenerateData() override;

  /** Do final mean and variance computation from data accumulated in threads.
   */
  void
  AfterStreamedGenerateData() override;

  /** Multi-thread version GenerateData. */
  void
  ThreadedStreamedGenerateData(const RegionType &) override;
  virtual void
  ThreadedGenerateDataImageSpatialMask(const RegionType &);
  virtual void
  ThreadedGenerateDataImageMask(const RegionType &);
  virtual void
                               SameGeometry();
  RegionType                   m_ImageRegion;
  ImageMaskConstPointer        m_ImageMask;
  ImageSpatialMaskConstPointer m_ImageSpatialMask;
  bool                         m_UseMask{ false };
  bool                         m_SameGeometry{ false };

private:
  ComputeImageExtremaFilter(const Self &);
  void
  operator=(const Self &);

  CompensatedSummation<RealType> m_ThreadSum{ 1 };
  CompensatedSummation<RealType> m_SumOfSquares{ 1 };
  SizeValueType                  m_Count{ 1 };
  PixelType                      m_ThreadMin{ 1 };
  PixelType                      m_ThreadMax{ 1 };

  std::mutex m_Mutex;
}; // end of class
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkComputeImageExtremaFilter.hxx"
#endif

#endif
