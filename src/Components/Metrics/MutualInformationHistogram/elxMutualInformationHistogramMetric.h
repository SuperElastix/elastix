#ifndef __elxMutualInformationHistogramMetric_H__
#define __elxMutualInformationHistogramMetric_H__

#include "elxIncludes.h"
#include "itkMutualInformationHistogramImageToImageMetric.h"

#include "elxTimer.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class MutualInformationHistogramMetric
	 * \brief An metric based on mutual information...
	 *
	 * This metric ...
	 *
	 * \ingroup Metrics
	 */

	template <class TElastix >	
		class MutualInformationHistogramMetric :
		public
			MutualInformationHistogramImageToImageMetric<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff.*/
		typedef MutualInformationHistogramMetric							Self;
		typedef MutualInformationHistogramImageToImageMetric<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MutualInformationHistogramMetric,
			MutualInformationHistogramImageToImageMetric );
		
		/** Name of this class.*/
		elxClassNameMacro( "MutualInformationHistogram" );

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
		typedef typename Superclass1::MovingImageConstPointer		MovingImageCosntPointer;
		typedef typename Superclass1::ScalesType								ScalesType;
		
		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
		
		/** Typedef's inherited from Elastix.*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;
			
		/** Typedef's for timer.*/
		typedef tmr::Timer					TimerType;
		typedef TimerType::Pointer	TimerPointer;
		
		/** Methods that have to be present everywhere.*/
		virtual int BeforeAll(void);
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);

		virtual void Initialize(void) throw (ExceptionObject);
		
	protected:

		MutualInformationHistogramMetric(); 
		virtual ~MutualInformationHistogramMetric() {}

	private:

		MutualInformationHistogramMetric( const Self& );	// purposely not implemented
		void operator=( const Self& );										// purposely not implemented
		
	}; // end class MutualInformationHistogramMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMutualInformationHistogramMetric.hxx"
#endif

#endif // end #ifndef __elxMutualInformationHistogramMetric_H__
