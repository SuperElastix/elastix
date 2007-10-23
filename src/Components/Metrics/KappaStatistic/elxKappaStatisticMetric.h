#ifndef __elxKappaStatisticMetric_H__
#define __elxKappaStatisticMetric_H__

#include "elxIncludes.h"
#include "itkKappaStatisticImageToImageMetric2.h"

#include "elxTimer.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class KappaStatisticMetric
	 * \brief An metric based on the itk::KappaStatisticImageToImageMetric2.h.
	 *
	 * The parameters used in this class are:
	 * \parameter Metric: Select this metric as follows:\n
	 *		<tt>(Metric "KappaStatistic")</tt>
   * \parameter UseComplement: Bool to use the complement of the metric or not.\n
   *    If true, the 1 - KappaStatistic is returned.\n
   *    <tt>(UseComplement "true")</tt>\n
   *    The default value is false.
   * \parameter ForeGroundvalue: the overlap of structures with this value is
   *    calculated.\n
   *    <tt>(ForeGroundvalue 3.5)</tt>\n
   *    The default value is 1.0.
	 *
	 * \ingroup Metrics
	 *
	 */

	template <class TElastix >
		class KappaStatisticMetric:
		public
			KappaStatisticImageToImageMetric2<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff. */
		typedef KappaStatisticMetric													Self;
		typedef KappaStatisticImageToImageMetric2<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( KappaStatisticMetric, KappaStatisticImageToImageMetric2 );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific metric. \n
		 * example: <tt>(Metric "KappaStatistic")</tt>\n
		 */
		elxClassNameMacro( "KappaStatistic" );

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

		/** Sets up a timer to measure the intialisation time and
		 * calls the Superclass' implementation.
		 */
		virtual void Initialize(void) throw (ExceptionObject);

    /** 
     * Do some things before registration:
     * \li Set the UseComplement setting
     * \li Set the ForeGroundvalue setting
     */
    virtual void BeforeRegistration( void );

	protected:

		/** The constructor. */
    KappaStatisticMetric(){};
		/** The destructor. */
		virtual ~KappaStatisticMetric() {}

	private:

		/** The private constructor. */
		KappaStatisticMetric( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );							// purposely not implemented
		
	}; // end class KappaStatisticMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxKappaStatisticMetric.hxx"
#endif

#endif // end #ifndef __elxKappaStatisticMetric_H__

