/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageSamplerBase_h
#define __ImageSamplerBase_h

#include "itkImageToVectorContainerFilter.h"
#include "itkImageSample.h"
#include "itkVectorDataContainer.h"
#include "itkSpatialObject.h"


namespace itk
{

  /** \class ImageSamplerBase
   *
   * \brief This class is a base class for any image sampler.
   *
   * \parameter ImageSampler: The way samples are taken from the fixed image in
   *    order to compute the metric value and its derivative in each iteration.
   *    Can be given for each resolution. Select one of {Random, Full, Grid, RandomCoordinate}.\n
   *    example: <tt>(ImageSampler "Random")</tt> \n
   *    The default is Random.
	 *
	 * \ingroup ImageSamplers
   */

  template < class TInputImage >
  class ImageSamplerBase :
    public ImageToVectorContainerFilter< TInputImage,
      VectorDataContainer< unsigned long, ImageSample< TInputImage > > >
  {
  public:

    /** Standard ITK-stuff. */
    typedef ImageSamplerBase                  Self;
    typedef ImageToVectorContainerFilter<
      TInputImage,
      VectorDataContainer<
        unsigned long,
        ImageSample< TInputImage > > >        Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( ImageSamplerBase, ImageToVectorContainerFilter );

    /** Typedefs inherited from the superclass. */
    typedef typename Superclass::DataObjectPointer            DataObjectPointer;
    typedef typename Superclass::OutputVectorContainerType    OutputVectorContainerType;
    typedef typename Superclass::OutputVectorContainerPointer OutputVectorContainerPointer;
    typedef typename Superclass::InputImageType               InputImageType;
    typedef typename Superclass::InputImagePointer            InputImagePointer;
    typedef typename Superclass::InputImageConstPointer       InputImageConstPointer;
    typedef typename Superclass::InputImageRegionType         InputImageRegionType;
    typedef typename Superclass::InputImagePixelType          InputImagePixelType;

    /** The input image dimension. */
    itkStaticConstMacro( InputImageDimension, unsigned int,
      InputImageType::ImageDimension );

    /** Other typdefs. */
    typedef ImageSample< InputImageType >               ImageSampleType;
    typedef VectorDataContainer< unsigned long,
      ImageSampleType >                                 ImageSampleContainerType;
    typedef typename InputImageType::SizeType           InputImageSizeType;
    typedef typename InputImageType::IndexType          InputImageIndexType;
    typedef typename InputImageType::PointType          InputImagePointType;
    typedef typename InputImagePointType::ValueType     InputImagePointValueType;
    typedef typename ImageSampleType::RealType          ImageSampleValueType;
    typedef SpatialObject<
      itkGetStaticConstMacro( InputImageDimension ) >   MaskType;
    typedef typename MaskType::Pointer                  MaskPointer;
    typedef typename MaskType::ConstPointer             MaskConstPointer;
    typedef std::vector< MaskConstPointer >             MaskVectorType;
    typedef std::vector< InputImageRegionType >         InputImageRegionVectorType;

    /** ******************** Masks ******************** */

    /** Set the masks. */
    virtual void SetMask( const MaskType *_arg, unsigned int pos );

    /** Set the first mask. NB: the first mask is used to
     * compute a bounding box in which samples are considered. */
    virtual void SetMask( const MaskType *_arg )
    {
      this->SetMask( _arg, 0 );
    }

    /** Get the masks. */
    virtual const MaskType * GetMask( unsigned int pos ) const;

    /** Get the first mask. */
    virtual const MaskType * GetMask( void ) const
    {
      return this->GetMask( 0 );
    };

    /** Set the number of masks. */
    virtual void SetNumberOfMasks( const unsigned int _arg );

    /** Get the number of masks. */
    itkGetConstMacro( NumberOfMasks, unsigned int );

    /** ******************** Regions ******************** */

    /** Set the region over which the samples will be taken. */
    virtual void SetInputImageRegion( const InputImageRegionType _arg, unsigned int pos );

    /** Set the region over which the samples will be taken. */
    virtual void SetInputImageRegion( const InputImageRegionType _arg )
    {
      this->SetInputImageRegion( _arg, 0 );
    }

    /** Get the input image regions. */
    virtual const InputImageRegionType & GetInputImageRegion( unsigned int pos ) const;

    /** Get the first input image region. */
    virtual const InputImageRegionType & GetInputImageRegion( void ) const
    {
      return this->GetInputImageRegion( 0 );
    };

    /** Set the number of input image regions. */
    virtual void SetNumberOfInputImageRegions( const unsigned int _arg );

    /** Get the number of input image regions. */
    itkGetConstMacro( NumberOfInputImageRegions, unsigned int );

    /** ******************** Other ******************** */

    /** SelectNewSamplesOnUpdate. When this function is called, the sampler
     * will generate a new sample set after calling Update(). The return bool
     * is false when this feature is not supported by the sampler. */
    virtual bool SelectNewSamplesOnUpdate( void );

    /** Returns whether the sampler supports SelectNewSamplesOnUpdate() */
    virtual bool SelectingNewSamplesOnUpdateSupported( void ) const
    {
      return true;
    }

    /** Get a handle to the cropped InputImageregion. */
    itkGetConstReferenceMacro( CroppedInputImageRegion, InputImageRegionType );

  protected:

    /** The constructor. */
    ImageSamplerBase();

    /** The destructor. */
    virtual ~ImageSamplerBase() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** GenerateInputRequestedRegion. */
    virtual void GenerateInputRequestedRegion( void );

    /** IsInsideAllMasks. */
    virtual bool IsInsideAllMasks( const InputImagePointType & point ) const;

    /** UpdateAllMasks. */
    virtual void UpdateAllMasks( void );

    /** Checks if the InputImageRegions are a subregion of the
     * LargestPossibleRegions.
     */
    virtual bool CheckInputImageRegions( void );

    /** Compute the intersection of the InputImageRegion and the bounding box of the mask. */
    void CropInputImageRegion( void );

  private:

    /** The private constructor. */
    ImageSamplerBase( const Self& );          // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );            // purposely not implemented

    /** Member variables. */
    MaskConstPointer                  m_Mask;
    MaskVectorType                    m_MaskVector;
    unsigned int                      m_NumberOfMasks;
    InputImageRegionType              m_InputImageRegion;
    InputImageRegionVectorType        m_InputImageRegionVector;
    unsigned int                      m_NumberOfInputImageRegions;

    InputImageRegionType              m_CroppedInputImageRegion;
    InputImageRegionType              m_DummyInputImageRegion;

  }; // end class ImageSamplerBase


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageSamplerBase.txx"
#endif

#endif // end #ifndef __ImageSamplerBase_h

