/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageRandomSamplerBase_h
#define __ImageRandomSamplerBase_h

#include "itkImageSamplerBase.h"

namespace itk
{

/** \class ImageRandomSamplerBase
 *
 * \brief This class is a base class for any image sampler that randomly picks samples.
 *
 * It adds the Set/GetNumberOfSamples function.
 *
 * \ingroup ImageSamplers
 */

  template < class TInputImage >
  class ImageRandomSamplerBase :
    public ImageSamplerBase< TInputImage >
  {
  public:

    /** Standard ITK-stuff. */
    typedef ImageRandomSamplerBase            Self;
    typedef ImageSamplerBase< TInputImage >   Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( ImageRandomSamplerBase, ImageSamplerBase );

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
    typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;
    typedef typename Superclass::MaskType                     MaskType;

    /** The input image dimension. */
    itkStaticConstMacro( InputImageDimension, unsigned int,
      Superclass::InputImageDimension );

    /** Set the number of samples. */
    itkSetClampMacro( NumberOfSamples, unsigned long, 1, NumericTraits<unsigned long>::max() );

  protected:

    /** The constructor. */
    ImageRandomSamplerBase();

    /** The destructor. */
    virtual ~ImageRandomSamplerBase() {};

    /** Multi-threaded function that does the work. */
    virtual void BeforeThreadedGenerateData( void );

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Member variable used when threading. */
    std::vector<double>                       m_RandomNumberList;

  private:

    /** The private constructor. */
    ImageRandomSamplerBase( const Self& );    // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );            // purposely not implemented

  }; // end class ImageRandomSamplerBase


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageRandomSamplerBase.txx"
#endif

#endif // end #ifndef __ImageRandomSamplerBase_h
