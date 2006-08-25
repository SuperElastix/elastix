#ifndef __elxAdvancedMeanSquaresMetric_H__
#define __elxAdvancedMeanSquaresMetric_H__

#include "elxIncludes.h"
#include "itkAdvancedMeanSquaresImageToImageMetric.h"

#include "elxTimer.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class AdvancedMeanSquaresMetric
	 * \brief An metric based on the itk::AdvancedMeanSquaresImageToImageMetric.
	 *
	 * The parameters used in this class are:
	 * \parameter Metric: Select this metric as follows:\n
	 *		<tt>(Metric "AdvancedMeanSquares")</tt>
	 *
	 * \ingroup Metrics
	 *
	 */

	template <class TElastix >
		class AdvancedMeanSquaresMetric:
		public
			AdvancedMeanSquaresImageToImageMetric<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef AdvancedMeanSquaresMetric															Self;
		typedef AdvancedMeanSquaresImageToImageMetric<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( AdvancedMeanSquaresMetric, AdvancedMeanSquaresImageToImageMetric );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific metric. \n
		 * example: <tt>(Metric "MeanSquares")</tt>\n
		 */
		elxClassNameMacro( "AdvancedMeanSquares" );

		/** Typedefs inherited from the superclass. */
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
		
		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
		
		/** Other typedef's. */
		typedef typename Superclass1::FixedImageMaskType 				FixedMaskImageType;
		typedef typename Superclass1::MovingImageMaskType				MovingMaskImageType;
		typedef typename Superclass1::FixedImageMaskPointer			FixedMaskImagePointer;
		typedef typename Superclass1::MovingImageMaskPointer		MovingMaskImagePointer;
		
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
	
		/** Sets up a timer to measure the initialisation time and
		 * calls the Superclass' implementation.
		 */
		virtual void Initialize(void) throw (ExceptionObject);

    /** 
     * Do some things before each resolution:
     * \li Set the UseDifferentiableOverlap setting
     */
    virtual void BeforeEachResolution(void);

	protected:

		/** The constructor. */
    AdvancedMeanSquaresMetric(){};
		/** The destructor. */
		virtual ~AdvancedMeanSquaresMetric() {}

	private:

		/** The private constructor. */
		AdvancedMeanSquaresMetric( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );							// purposely not implemented
		
	}; // end class AdvancedMeanSquaresMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAdvancedMeanSquaresMetric.hxx"
#endif

#endif // end #ifndef __elxAdvancedMeanSquaresMetric_H__

