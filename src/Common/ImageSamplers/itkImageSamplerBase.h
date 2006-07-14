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
    typedef SpatialObject<
      itkGetStaticConstMacro( InputImageDimension ) > 	MaskType;

    /** Set the number of samples. */
    itkSetClampMacro( NumberOfSamples, unsigned long,
			1, NumericTraits<unsigned long>::max() );

    /** Get the number of samples. */
		itkGetConstMacro( NumberOfSamples, unsigned long );

    /** Set the mask. */
    itkSetObjectMacro( Mask, MaskType );

    /** Get the mask. */
    itkGetConstObjectMacro( Mask, MaskType );
   
    /** Set the region over which the samples will be taken. */
    itkSetMacro( InputImageRegion, InputImageRegionType );

    /** Get the region over which the samples will be taken. */
    itkGetConstReferenceMacro( InputImageRegion, InputImageRegionType );

    /** SelectNewSamplesOnUpdate. */
    virtual bool SelectNewSamplesOnUpdate( void );

  protected:

    /** The constructor. */
    ImageSamplerBase();
    /** The destructor. */
    virtual ~ImageSamplerBase() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** GenerateInputRequestedRegion. */
    virtual void GenerateInputRequestedRegion( void );
    
  private:

		/** The private constructor. */
    ImageSamplerBase( const Self& );	        // purposely not implemented
		/** The private copy constructor. */
    void operator=( const Self& );				    // purposely not implemented

    /** Member variables. */
    unsigned long                 m_NumberOfSamples;
    typename MaskType::Pointer    m_Mask;
    InputImageRegionType          m_InputImageRegion;

  }; // end class ImageSamplerBase


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageSamplerBase.txx"
#endif

#endif // end #ifndef __ImageSamplerBase_h

