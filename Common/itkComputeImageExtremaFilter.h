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
#include "itkImageMaskSpatialObject2.h"

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
template< typename TInputImage >
class ComputeImageExtremaFilter :
  public StatisticsImageFilter< TInputImage >
{
public:
  /** Standard Self typedef */
  typedef ComputeImageExtremaFilter                     Self;
  typedef StatisticsImageFilter< TInputImage >          Superclass;
  typedef SmartPointer< Self >                          Pointer;
  typedef SmartPointer< const Self >                    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ComputeImageExtremaFilter, StatisticsImageFilter);

  /** Image related typedefs. */
  typedef typename TInputImage::Pointer InputImagePointer;

  typedef typename Superclass::RegionType RegionType;
  typedef typename Superclass::SizeType   SizeType;
  typedef typename Superclass::IndexType  IndexType;
  typedef typename Superclass::PixelType  PixelType;
  typedef typename TInputImage::PointType  PointType;

  /** Image related typedefs. */
  itkStaticConstMacro( ImageDimension, unsigned int,
                      TInputImage::ImageDimension );

  /** Type to use for computations. */
  typedef typename Superclass::RealType RealType;

  itkSetMacro( ImageRegion, RegionType );
  itkSetMacro( UseMask, bool );

  typedef SpatialObject< itkGetStaticConstMacro(ImageDimension) > ImageMaskType;
  typedef typename ImageMaskType::Pointer ImageMaskPointer;
  typedef typename ImageMaskType::ConstPointer ImageMaskConstPointer;
  itkSetConstObjectMacro( ImageMask, ImageMaskType );
  itkGetConstObjectMacro( ImageMask, ImageMaskType );

  typedef ImageMaskSpatialObject2< itkGetStaticConstMacro(ImageDimension) > ImageSpatialMaskType;
  typedef typename ImageSpatialMaskType::Pointer ImageSpatialMaskPointer;
  typedef typename ImageSpatialMaskType::ConstPointer ImageSpatialMaskConstPointer;
  itkSetConstObjectMacro( ImageSpatialMask, ImageSpatialMaskType );
  itkGetConstObjectMacro( ImageSpatialMask, ImageSpatialMaskType );

protected:
  ComputeImageExtremaFilter();
  virtual ~ComputeImageExtremaFilter() {}

  /** Initialize some accumulators before the threads run. */
  virtual void BeforeThreadedGenerateData();

  /** Do final mean and variance computation from data accumulated in threads.
   */
  virtual void AfterThreadedGenerateData();

  /** Multi-thread version GenerateData. */
  virtual void ThreadedGenerateData( const RegionType &
                             outputRegionForThread,
                             ThreadIdType threadId );
  virtual void ThreadedGenerateDataImageSpatialMask( const RegionType &
                              outputRegionForThread,
                              ThreadIdType threadId );
  virtual void ThreadedGenerateDataImageMask( const RegionType &
                              outputRegionForThread,
                              ThreadIdType threadId );
  virtual void SameGeometry();
  RegionType            m_ImageRegion;
  ImageMaskConstPointer m_ImageMask;
  ImageSpatialMaskConstPointer  m_ImageSpatialMask;
  bool                  m_UseMask;
  bool                  m_SameGeometry;

private:
  ComputeImageExtremaFilter( const Self & );
  void operator = ( const Self & );
  Array< RealType >       m_ThreadSum;
  Array< RealType >       m_SumOfSquares;
  Array< SizeValueType >  m_Count;
  Array< PixelType >      m_ThreadMin;
  Array< PixelType >      m_ThreadMax;

}; // end of class
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkComputeImageExtremaFilter.hxx"
#endif

#endif
