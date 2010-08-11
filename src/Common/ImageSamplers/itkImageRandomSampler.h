/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageRandomSampler_h
#define __ImageRandomSampler_h

#include "itkImageRandomSamplerBase.h"


namespace itk
{

  /** \class ImageRandomSampler
   *
   * \brief Samples randomly some voxels of an image.
   *
   * This image sampler randomly samples 'NumberOfSamples' voxels in
   * the InputImageRegion. Voxels may be selected multiple times.
   * If a mask is given, the sampler tries to find samples within the
   * mask. If the mask is very sparse, this may take some time. In this case,
   * consider using the ImageRandomSamplerSparseMask.
   *
	 * \ingroup ImageSamplers
	 * */

  template < class TInputImage >
  class ImageRandomSampler :
    public ImageRandomSamplerBase< TInputImage >
  {
  public:

    /** Standard ITK-stuff. */
    typedef ImageRandomSampler                     Self;
    typedef ImageRandomSamplerBase< TInputImage >  Superclass;
    typedef SmartPointer<Self>                     Pointer;
    typedef SmartPointer<const Self>               ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( ImageRandomSampler, ImageRandomSamplerBase );

    /** Typedefs inherited from the superclass. */
    typedef typename Superclass::DataObjectPointer            DataObjectPointer;
    typedef typename Superclass::OutputVectorContainerType    OutputVectorContainerType;
    typedef typename Superclass::OutputVectorContainerPointer OutputVectorContainerPointer;
    typedef typename Superclass::InputImageType               InputImageType;
    typedef typename Superclass::InputImagePointer            InputImagePointer;
    typedef typename Superclass::InputImageConstPointer       InputImageConstPointer;
    typedef typename Superclass::InputImageRegionType         InputImageRegionType;
    typedef typename Superclass::InputImagePixelType          InputImagePixelType;
    typedef typename Superclass::ImageSampleType              ImageSampleType;
    typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
    typedef typename Superclass::MaskType                     MaskType;

    /** The input image dimension. */
    itkStaticConstMacro( InputImageDimension, unsigned int,
      Superclass::InputImageDimension );

    /** Other typedefs. */
    typedef typename InputImageType::IndexType    InputImageIndexType;
    typedef typename InputImageType::PointType    InputImagePointType;

  protected:

    /** The constructor. */
    ImageRandomSampler(){};
    /** The destructor. */
    virtual ~ImageRandomSampler() {};

    /** Function that does the work. */
    virtual void GenerateData( void );

  private:

    /** The private constructor. */
    ImageRandomSampler( const Self& );          // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );            // purposely not implemented

  }; // end class ImageRandomSampler


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageRandomSampler.txx"
#endif

#endif // end #ifndef __ImageRandomSampler_h

