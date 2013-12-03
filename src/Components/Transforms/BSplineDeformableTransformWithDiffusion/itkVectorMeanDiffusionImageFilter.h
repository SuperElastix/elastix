/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkVectorMeanDiffusionImageFilter_H__
#define __itkVectorMeanDiffusionImageFilter_H__

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkNumericTraits.h"

#include "itkRescaleIntensityImageFilter.h"

namespace itk
{
/**
 * \class VectorMeanDiffusionImageFilter
 * \brief Applies an averaging filter to an image
 *
 * Computes an image where a given pixel is the mean value of the
 * the pixels in a neighborhood about the corresponding input pixel.
 *
 * A mean filter is one of the family of linear filters.
 *
 * \sa Image
 * \sa Neighborhood
 * \sa NeighborhoodOperator
 * \sa NeighborhoodIterator
 *
 * \ingroup IntensityImageFilters
 */

template <class TInputImage, class TGrayValueImage >
class VectorMeanDiffusionImageFilter
  : public ImageToImageFilter< TInputImage, TInputImage >
{
public:

  /** Convenient typedefs for simplifying declarations. */
  typedef TInputImage         InputImageType;
  typedef TGrayValueImage     GrayValueImageType;
  typedef typename GrayValueImageType::Pointer    GrayValueImagePointer;

  /** Standard class typedefs. */
  typedef VectorMeanDiffusionImageFilter      Self;
  typedef ImageToImageFilter<
    InputImageType, InputImageType >          Superclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  /** Extract dimension from input image. */
  itkStaticConstMacro( InputImageDimension, unsigned int,
    TInputImage::ImageDimension );

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( VectorMeanDiffusionImageFilter, ImageToImageFilter );

  /** Image typedef support. */
  typedef typename InputImageType::PixelType        InputPixelType;
  typedef typename InputPixelType::ValueType        ValueType;
  //typedef typename NumericTraits<InputPixelType>::RealType    InputRealType;
  typedef typename InputImageType::RegionType       InputImageRegionType;
  typedef typename InputImageType::SizeType         InputSizeType;
  typedef typename InputImageType::IndexType        IndexType;
  typedef Vector< double,
    itkGetStaticConstMacro( InputImageDimension ) > VectorRealType;
  typedef Image< double,
    itkGetStaticConstMacro( InputImageDimension ) > DoubleImageType;
  typedef typename DoubleImageType::Pointer         DoubleImagePointer;
  typedef typename GrayValueImageType::PixelType    GrayValuePixelType;

  /** Typedef for the rescale intensity filter. */
  typedef RescaleIntensityImageFilter<
    GrayValueImageType, DoubleImageType >           RescaleImageFilterType;
  typedef typename RescaleImageFilterType::Pointer  RescaleImageFilterPointer;

  /** Set the radius of the neighborhood used to compute the mean. */
  itkSetMacro( Radius, InputSizeType );

  /** Get the radius of the neighborhood used to compute the mean */
  itkGetConstReferenceMacro( Radius, InputSizeType );

  /** MeanImageFilter needs a larger input requested region than
   * the output requested region.  As such, MeanImageFilter needs
   * to provide an implementation for GenerateInputRequestedRegion()
   * in order to inform the pipeline execution model.
   *
   * \sa ImageToImageFilter::GenerateInputRequestedRegion().
   */
  virtual void GenerateInputRequestedRegion() throw( InvalidRequestedRegionError );

  /** Set & Get the NumberOfIterations. */
  itkSetMacro( NumberOfIterations, unsigned int );
  itkGetConstMacro( NumberOfIterations, unsigned int );

  /** Set- and GetObjectMacro's for the GrayValueImage. */
  void SetGrayValueImage( GrayValueImageType * _arg );
  typename GrayValueImageType::Pointer GetGrayValueImage( void )
  {
    return this->m_GrayValueImage.GetPointer();
  }

protected:

  VectorMeanDiffusionImageFilter();
  virtual ~VectorMeanDiffusionImageFilter() {}

  void PrintSelf( std::ostream& os, Indent indent ) const;

  /** MeanImageFilter can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData()
   * routine which is called for each processing thread. The output
   * image data is allocated automatically by the superclass prior to
   * calling ThreadedGenerateData().  ThreadedGenerateData can only
   * write to the portion of the output image specified by the
   * parameter "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData().
   */
  void GenerateData( void );

private:

  VectorMeanDiffusionImageFilter( const Self& );  // purposely not implemented
  void operator=( const Self& );                  // purposely not implemented

  /** Declare member variables. */
  InputSizeType   m_Radius;
  unsigned int    m_NumberOfIterations;

  /** Declare member images. */
  GrayValueImagePointer    m_GrayValueImage;
  DoubleImagePointer       m_Cx;

  RescaleImageFilterPointer  m_RescaleFilter;

  /** For calculating a feature image from the input m_GrayValueImage. */
  void FilterGrayValueImage( void );

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVectorMeanDiffusionImageFilter.hxx"
#endif

#endif // end #ifndef __itkVectorMeanDiffusionImageFilter_H__

