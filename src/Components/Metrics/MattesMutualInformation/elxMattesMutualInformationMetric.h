#ifndef __elxMattesMutualInformationMetric_H__
#define __elxMattesMutualInformationMetric_H__

#include "elxIncludes.h"
#include "itkMattesMutualInformationImageToImageMetric2.h"

#include "elxTimer.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class MattesMutualInformationMetric
	 * \brief A metric based on the itk::MattesMutualInformationImageToImageMetric2.
	 *
	 * This metric is based on an adapted version of the
	 * itk::MattesMutualInformationImageToImageMetric.
	 *
	 * The parameters used in this class are:
	 * \parameter Metric: Select this metric as follows:\n
	 *		<tt>(Metric "MattesMutualInformation")</tt>
	 * \parameter NumberOfHistogramBins: The size of the histogram. Must be given for each 
	 *		resolution. \n
	 *		example: <tt>(NumberOfHistogramBins 32 32 64)</tt> \n
	 *		The default is 32 for each resolution.
   *
   * \sa MattesMutualInformationImageToImageMetric2
	 * \ingroup Metrics
	 */
	
	template <class TElastix >	
		class MattesMutualInformationMetric :
		public
			MattesMutualInformationImageToImageMetric2<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef MattesMutualInformationMetric									Self;
		typedef MattesMutualInformationImageToImageMetric2<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MattesMutualInformationMetric,
			MattesMutualInformationImageToImageMetric2 );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific metric. \n
		 * example: <tt>(Metric "MattesMutualInformation")</tt>\n
		 */
		elxClassNameMacro( "MattesMutualInformation" );

		/** Typedefs inherited from the superclass.*/
		typedef typename Superclass1::TransformType							TransformType;
		typedef typename Superclass1::TransformPointer 					TransformPointer;
		typedef typename Superclass1::TransformJacobianType			TransformJacobianType;
		typedef typename Superclass1::InterpolatorType 					InterpolatorType;
		typedef typename Superclass1::MeasureType								MeasureType;
		typedef typename Superclass1::DerivativeType 						DerivativeType;
		typedef typename Superclass1::ParametersType 						ParametersType;
		typedef typename Superclass1::FixedImageType 						FixedImageType;
		typedef typename Superclass1::MovingImageType						MovingImageType;
		typedef typename Superclass1::FixedImageConstPointer 		FixedImageConstPointer;
		typedef typename Superclass1::MovingImageConstPointer		MovingImageConstPointer;
		typedef typename Superclass1::FixedImageIndexType				FixedImageIndexType;
		typedef typename Superclass1::FixedImageIndexValueType 	FixedImageIndexValueType;
		typedef typename Superclass1::MovingImageIndexType 			MovingImageIndexType;
		typedef typename Superclass1::FixedImagePointType				FixedImagePointType;
		typedef typename Superclass1::MovingImagePointType 			MovingImagePointType;
		
		/** The fixed image dimension */
		itkStaticConstMacro (FixedImageDimension, unsigned int,
			FixedImageType::ImageDimension);
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
		 * \li Set the number of spatial samples.
		 * \li Set the flag to use all samples.
		 * \li Set the flag to calculate and show the exact metric value.
		 * \li Set the flag to take samples on a uniform grid.
		 * \li Set the grid spacing of the sampling grid.
		 */
		virtual void BeforeEachResolution(void);
	
		/** Sets up a timer to measure the intialisation time and
		 * calls the Superclass' implementation.
		 */
		virtual void Initialize(void) throw (ExceptionObject);

	
	protected:

		/** The constructor. */
    MattesMutualInformationMetric() {}; 
    /** The destructor. */ 
		virtual ~MattesMutualInformationMetric() {}
	
	private:

		/** The private constructor. */
		MattesMutualInformationMetric( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );								// purposely not implemented
		
	}; // end class MattesMutualInformationMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMattesMutualInformationMetric.hxx"
#endif

#endif // end #ifndef __elxMattesMutualInformationMetric_H__
