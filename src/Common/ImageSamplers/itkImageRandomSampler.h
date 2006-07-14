#ifndef __ImageRandomSampler_h
#define __ImageRandomSampler_h

#include "itkImageSamplerBase.h"
//#include "itkImageSample.h"
//#include "itkVectorContainer.h"


namespace itk
{

  /** \class ImageRandomSampler
   *
   * \brief 
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

    /** Other typdefs. */
    typedef typename InputImageType::IndexType    InputImageIndexType;
    typedef typename InputImageType::PointType    InputImagePointType;
    
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

  }; // end class ImageRandomSampler


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageRandomSampler.txx"
#endif

#endif // end #ifndef __ImageRandomSampler_h

