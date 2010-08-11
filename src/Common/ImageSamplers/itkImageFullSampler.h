/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageFullSampler_h
#define __ImageFullSampler_h

#include "itkImageSamplerBase.h"

namespace itk
{

  /** \class ImageFullSampler
   *
   * \brief Samples all voxels in the InputImageRegion.
   *
   * This ImageSampler samples all voxels in the InputImageRegion.
   * If a mask is given: only those voxels within the mask AND the
   * InputImageRegion.
   *
	 * \ingroup ImageSamplers
   */

  template < class TInputImage >
  class ImageFullSampler :
    public ImageSamplerBase< TInputImage >
  {
  public:

    /** Standard ITK-stuff. */
    typedef ImageFullSampler                  Self;
    typedef ImageSamplerBase< TInputImage >   Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( ImageFullSampler, ImageSamplerBase );

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

    /** Other typdefs. */
    typedef typename InputImageType::IndexType    InputImageIndexType;
    typedef typename InputImageType::PointType    InputImagePointType;

    /** Selecting new samples makes no sense if nothing changed.
     * The same samples would be selected anyway.
     */
    virtual bool SelectNewSamplesOnUpdate(void)
    {
      return false;
    };

    /** Returns whether the sampler supports SelectNewSamplesOnUpdate(). */
    virtual bool SelectingNewSamplesOnUpdateSupported( void ) const
    {
      return false;
    }

  protected:

    /** The constructor. */
    ImageFullSampler() {};
    /** The destructor. */
    virtual ~ImageFullSampler() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Function that does the work. */
    virtual void GenerateData( void );

  private:

    /** The private constructor. */
    ImageFullSampler( const Self& );          // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );            // purposely not implemented

  }; // end class ImageFullSampler


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageFullSampler.txx"
#endif

#endif // end #ifndef __ImageFullSampler_h

