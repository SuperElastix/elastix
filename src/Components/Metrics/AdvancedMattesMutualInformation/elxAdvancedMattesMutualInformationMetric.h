#ifndef __elxAdvancedMattesMutualInformationMetric_H__
#define __elxAdvancedMattesMutualInformationMetric_H__

#include "elxIncludes.h"
#include "itkParzenWindowMutualInformationImageToImageMetric.h"

#include "elxTimer.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class AdvancedMattesMutualInformationMetric
	 * \brief A metric based on the itk::ParzenWindowMutualInformationImageToImageMetric.
	 *
	 * This metric is based on an adapted version of the
	 * itk::MattesMutualInformationImageToImageMetric.
	 *
	 * The parameters used in this class are:
	 * \parameter Metric: Select this metric as follows:\n
	 *		<tt>(Metric "AdvancedMattesMutualInformation")</tt>
	 * \parameter NumberOfHistogramBins: The size of the histogram. Must be given for each 
	 *		resolution, or for all resolutions at once. \n
	 *		example: <tt>(NumberOfHistogramBins 32 32 64)</tt> \n
	 *		The default is 32 for each resolution.
   * \parameter CheckNumberOfSamples: Whether the metric checks if at least 1/4 of the 
   *    samples map inside the moving image. Must be given for each resolution or for all
   *    resolutions at once. \n
   *    example: <tt>(CheckNumberOfSamples "false" "true" "false")</tt> \n
   *    The default is true. In general it is wise to set this to true, since it detects
   *    if the registration is going really bad.
   *
   * \sa ParzenWindowMutualInformationImageToImageMetric
	 * \ingroup Metrics
	 */
	
	template <class TElastix >	
		class AdvancedMattesMutualInformationMetric :
		public
			ParzenWindowMutualInformationImageToImageMetric<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef AdvancedMattesMutualInformationMetric					Self;
		typedef ParzenWindowMutualInformationImageToImageMetric<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( AdvancedMattesMutualInformationMetric,
			ParzenWindowMutualInformationImageToImageMetric );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific metric. \n
		 * example: <tt>(Metric "AdvancedMattesMutualInformation")</tt>\n
		 */
		elxClassNameMacro( "AdvancedMattesMutualInformation" );

		/** Typedefs from the superclass. */
    typedef typename 
      Superclass::CoordinateRepresentationType              CoordinateRepresentationType;
    typedef typename Superclass::MovingImageType            MovingImageType;
    typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
    typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
    typedef typename Superclass::FixedImageType             FixedImageType;
    typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
    typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
    typedef typename Superclass::TransformType              TransformType;
    typedef typename Superclass::TransformPointer           TransformPointer;
    typedef typename Superclass::InputPointType             InputPointType;
    typedef typename Superclass::OutputPointType            OutputPointType;
    typedef typename Superclass::TransformParametersType    TransformParametersType;
    typedef typename Superclass::TransformJacobianType      TransformJacobianType;
    typedef typename Superclass::InterpolatorType           InterpolatorType;
    typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
    typedef typename Superclass::RealType                   RealType;
    typedef typename Superclass::GradientPixelType          GradientPixelType;
    typedef typename Superclass::GradientImageType          GradientImageType;
    typedef typename Superclass::GradientImagePointer       GradientImagePointer;
    typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
    typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
    typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
    typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
    typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
    typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
    typedef typename Superclass::MeasureType                MeasureType;
    typedef typename Superclass::DerivativeType             DerivativeType;
    typedef typename Superclass::ParametersType             ParametersType;
    typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
    typedef typename Superclass::MovingImageRegionType      MovingImageRegionType;
    typedef typename Superclass::ImageSamplerType           ImageSamplerType;
    typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
    typedef typename Superclass::ImageSampleContainerType   ImageSampleContainerType;
    typedef typename 
      Superclass::ImageSampleContainerPointer               ImageSampleContainerPointer;
    typedef typename Superclass::InternalMaskPixelType      InternalMaskPixelType;
    typedef typename
      Superclass::InternalMovingImageMaskType               InternalMovingImageMaskType;
    typedef typename 
      Superclass::MovingImageMaskInterpolatorType           MovingImageMaskInterpolatorType;
    typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
    typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
    typedef typename
      Superclass::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
    typedef typename
      Superclass::MovingImageLimiterOutputType              MovingImageLimiterOutputType;
		
    /** The fixed image dimension. */
		itkStaticConstMacro( FixedImageDimension, unsigned int,
			FixedImageType::ImageDimension );

		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
		
		/** Typedef's inherited from Elastix. */
		typedef typename Superclass2::ElastixType								ElastixType;
		typedef typename Superclass2::ElastixPointer						ElastixPointer;
		typedef typename Superclass2::ConfigurationType					ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer			ConfigurationPointer;
		typedef typename Superclass2::RegistrationType					RegistrationType;
		typedef typename Superclass2::RegistrationPointer				RegistrationPointer;
		typedef typename Superclass2::ITKBaseType								ITKBaseType;
			
		/** Typedef for timer. */
		typedef tmr::Timer					TimerType;
		/** Typedef for timer. */
		typedef TimerType::Pointer	TimerPointer;

		/** Execute stuff before each new pyramid resolution:
		 * \li Set the number of histogram bins.
     * \li Set the CheckNumberOfSamples option.
     * \li Set the UseDifferentiableOverlap setting
     * \li Set the MovingMaskInterpolationOrder setting
     * \li Set the fixed/moving LimitRangeRatio
     * \li Set the fixed/moving limiter. */
		virtual void BeforeEachResolution( void );
	
		/** Set up a timer to measure the intialisation time and
		 * call the Superclass' implementation. */
		virtual void Initialize(void) throw (ExceptionObject);
	
	protected:

		/** The constructor. */
    AdvancedMattesMutualInformationMetric() {}; 
    /** The destructor. */ 
		virtual ~AdvancedMattesMutualInformationMetric() {}
	
	private:

		/** The private constructor. */
		AdvancedMattesMutualInformationMetric( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );								// purposely not implemented
		
	}; // end class AdvancedMattesMutualInformationMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAdvancedMattesMutualInformationMetric.hxx"
#endif

#endif // end #ifndef __elxAdvancedMattesMutualInformationMetric_H__
