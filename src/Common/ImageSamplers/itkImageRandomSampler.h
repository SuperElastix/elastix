#ifndef __ImageRandomSampler_h
#define __ImageRandomSampler_h

#include "itkImageSamplerBase.h"


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
	 * \parameter NumberOfSpatialSamples: The number of image voxels used for computing the
	 *		metric value and its derivative in each iteration. Must be given for each resolution.\n
	 *		example: <tt>(NumberOfSpatialSamples 2048 2048 4000)</tt> \n
	 *		The default is 5000.
   *
   * 
   */

  template < class TInputImage >
  class ImageRandomSampler :
    public ImageSamplerBase< TInputImage >
  {
  public:

		/** Standard ITK-stuff. */
    typedef ImageRandomSampler                Self;
    typedef ImageSamplerBase< TInputImage >   Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

		/** Method for creation through the object factory. */
    itkNewMacro( Self );

		/** Run-time type information (and related methods). */
    itkTypeMacro( ImageRandomSampler, ImageSamplerBase );

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

    /** Set the number of samples. */
    itkSetClampMacro( NumberOfSamples, unsigned long, 1, NumericTraits<unsigned long>::max() );

    /** Get the number of samples. */
    itkGetConstMacro( NumberOfSamples, unsigned long );
    
  protected:

    /** The constructor. */
    ImageRandomSampler();
    /** The destructor. */
    virtual ~ImageRandomSampler() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Function that does the work. */
    virtual void GenerateData( void );
            
  private:

		/** The private constructor. */
    ImageRandomSampler( const Self& );	        // purposely not implemented
		/** The private copy constructor. */
    void operator=( const Self& );				    // purposely not implemented

    unsigned long m_NumberOfSamples;

  }; // end class ImageRandomSampler


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageRandomSampler.txx"
#endif

#endif // end #ifndef __ImageRandomSampler_h

