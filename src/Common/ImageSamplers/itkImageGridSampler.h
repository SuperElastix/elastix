#ifndef __ImageGridSampler_h
#define __ImageGridSampler_h

#include "itkImageSamplerBase.h"

namespace itk
{

  /** \class ImageGridSampler
   *
   * \brief Samples image voxels on a regular grid.
   *
   * This ImageSampler samples voxels that lie on a regular grid.
   * The grid can be specified by an integer downsampling factor for
   * each dimension.
   *
   * \parameter SampleGridSpacing: This parameter controls the spacing
   *    of the uniform grid in all dimensions. This should be given in
	 *		index coordinates. \n
	 *		example: <tt>(SampleGridSpacing 4 4 4)</tt> \n
	 *		Default is 2 in each dimension.
   */

  template < class TInputImage >
  class ImageGridSampler :
    public ImageSamplerBase< TInputImage >
  {
  public:

		/** Standard ITK-stuff. */
    typedef ImageGridSampler                Self;
    typedef ImageSamplerBase< TInputImage >   Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

		/** Method for creation through the object factory. */
    itkNewMacro( Self );

		/** Run-time type information (and related methods). */
    itkTypeMacro( ImageGridSampler, ImageSamplerBase );

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

    /** Typedefs for support of user defined grid spacing for the spatial samples. */
		typedef typename InputImageType::OffsetType							SampleGridSpacingType;
		typedef typename SampleGridSpacingType::OffsetValueType SampleGridSpacingValueType;
		typedef typename InputImageType::SizeType								SampleGridSizeType;
		typedef InputImageIndexType															SampleGridIndexType;
		typedef typename InputImageType::SizeType 							InputImageSizeType;
		   
    /** Set/Get the sample grid spacing for each dimension (only integer factors)
     * This function overrules and previous calls to SetNumberOfSamples. */
    itkSetMacro(SampleGridSpacing, SampleGridSpacingType);
    itkGetConstMacro(SampleGridSpacing, SampleGridSpacingType);

    /** Define an isotropic SampleGridSpacing such that the desired number
     * of samples is approximately realised. The following formula is used:
     *
     * spacing = max[ 1, round( (availablevoxels/nrofsamples)^(1/dimension) ) ],
     * with
     * availablevoxels = nr of voxels in input image region.
     * 
     * The InputImageRegion needs to be specified beforehand. A mask is ignored,
     * so the realised number of samples could be significantly lower than expected.
     * This function overrules any previous calls to SetSampleGridSpacing.  */
    virtual void SetNumberOfSamples( unsigned long nrofsamples );

    /** Selecting new samples makes no sense if nothing changed. The same
     * samples would be selected anyway. */
    virtual bool SelectNewSamplesOnUpdate(void)
    {
      return false;
    };

    /** Returns whether the sampler supports SelectNewSamplesOnUpdate() */
    virtual bool SelectingNewSamplesOnUpdateSupported( void ) const
    {
      return false;
    }
    
  protected:

    /** The constructor. */
    ImageGridSampler() {};
    /** The destructor. */
    virtual ~ImageGridSampler() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Function that does the work. */
    virtual void GenerateData( void );

    /** An array of integer spacing factors */
    SampleGridSpacingType m_SampleGridSpacing;
            
  private:

		/** The private constructor. */
    ImageGridSampler( const Self& );	        // purposely not implemented
		/** The private copy constructor. */
    void operator=( const Self& );				    // purposely not implemented

  }; // end class ImageGridSampler


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageGridSampler.txx"
#endif

#endif // end #ifndef __ImageGridSampler_h

