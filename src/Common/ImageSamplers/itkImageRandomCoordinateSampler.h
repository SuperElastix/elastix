#ifndef __ImageRandomCoordinateSampler_h
#define __ImageRandomCoordinateSampler_h

#include "itkImageSamplerBase.h"
#include "itkInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"


namespace itk
{

  /** \class ImageRandomCoordinateSampler
   *
   * \brief Samples an image by randomly composing a set of physical coordinates
   *
   * This image sampler generates not only samples that correspond with 
   * pixel locations, but selects points in physical space.
   */

  template < class TInputImage >
  class ImageRandomCoordinateSampler :
    public ImageSamplerBase< TInputImage >
  {
  public:

		/** Standard ITK-stuff. */
    typedef ImageRandomCoordinateSampler                Self;
    typedef ImageSamplerBase< TInputImage >   Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

		/** Method for creation through the object factory. */
    itkNewMacro( Self );

		/** Run-time type information (and related methods). */
    itkTypeMacro( ImageRandomCoordinateSampler, ImageSamplerBase );

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
    typedef typename InputImageType::IndexType      InputImageIndexType;
    typedef typename InputImageType::PointType      InputImagePointType;
    typedef typename InputImagePointType::ValueType InputImagePointValueType;
    typedef typename ImageSampleType::RealType      ImageSampleValueType;

    /** This image sampler samples the image on physical coordinates and thus
     * needs an interpolator */
    typedef double                                              CoordRepType;
    typedef InterpolateImageFunction< 
      InputImageType, CoordRepType >                            InterpolatorType;
    typedef BSplineInterpolateImageFunction<
      InputImageType, CoordRepType, double>                     DefaultInterpolatorType;

    /** The random number generator used to generate random coordinates */
    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;

    /** Set/Get the number of samples */
    itkGetConstMacro(NumberOfSamples, unsigned long);
    itkSetMacro(NumberOfSamples, unsigned long);

    /** Set/Get the interpolator. A 3rd order BSpline interpolator is used by default. */
    itkSetObjectMacro(Interpolator, InterpolatorType);
    itkGetObjectMacro(Interpolator, InterpolatorType);
    
  protected:

    /** The constructor. */
    ImageRandomCoordinateSampler();
    /** The destructor. */
    virtual ~ImageRandomCoordinateSampler() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Function that does the work. */
    virtual void GenerateData( void );

    /** Generate a point randomly in a bounding box.
     * This method can be overwritten in subclasses if a different distribution is desired. */
    virtual void GenerateRandomCoordinate(
      const InputImagePointType & smallestPoint,
      const InputImagePointType & largestPoint,
      InputImagePointType &       randomPoint);

    typename InterpolatorType::Pointer  m_Interpolator;
    typename RandomGeneratorType::Pointer m_RandomGenerator;
            
  private:

		/** The private constructor. */
    ImageRandomCoordinateSampler( const Self& );	        // purposely not implemented
		/** The private copy constructor. */
    void operator=( const Self& );				    // purposely not implemented

    unsigned long m_NumberOfSamples;

  }; // end class ImageRandomCoordinateSampler


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageRandomCoordinateSampler.txx"
#endif

#endif // end #ifndef __ImageRandomCoordinateSampler_h

