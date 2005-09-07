#ifndef __elxViolaWellsMutualInformationMetric_H__
#define __elxViolaWellsMutualInformationMetric_H__

#include "elxIncludes.h"
#include "itkMutualInformationImageToImageMetricMoreRandom.h"

#include "elxTimer.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class ViolaWellsMutualInformationMetric
	 * \brief An metric based on mutual information...
	 *
	 * This metric ...
	 *
	 * \ingroup Metrics
	 */

	template <class TElastix >	
		class ViolaWellsMutualInformationMetric :
		public
			MutualInformationImageToImageMetricMoreRandom<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff.*/
		typedef ViolaWellsMutualInformationMetric							Self;
		typedef MutualInformationImageToImageMetricMoreRandom<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( ViolaWellsMutualInformationMetric,
			MutualInformationImageToImageMetricMoreRandom );
		
		/** Name of this class.*/
		elxClassNameMacro( "ViolaWellsMutualInformation" );

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
		typedef typename Superclass1::FixedImageIndexType				FixedImageIndexType;
		typedef typename Superclass1::FixedImageIndexValueType 	FixedImageIndexValueType;
		typedef typename Superclass1::MovingImageIndexType 			MovingImageIndexType;
		typedef typename Superclass1::FixedImagePointType				FixedImagePointType;
		typedef typename Superclass1::MovingImagePointType 			MovingImagePointType;
		
		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
		
		/** Other typedef's for masks. *
		typedef typename Superclass1::FixedMaskImageType 				FixedMaskImageType;
		typedef typename Superclass1::MovingMaskImageType				MovingMaskImageType;
		typedef typename Superclass1::FixedMaskImagePointer			FixedMaskImagePointer;
		typedef typename Superclass1::MovingMaskImagePointer		MovingMaskImagePointer;
		
		/** Typedef's inherited from Elastix. */
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;
			
		/** Typedef's for timer. */
		typedef tmr::Timer					TimerType;
		typedef TimerType::Pointer	TimerPointer;
		
		/** Methods that have to be present everywhere.*/
		virtual int BeforeAll(void);
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);

		virtual void Initialize(void) throw (ExceptionObject);
		
	protected:

		ViolaWellsMutualInformationMetric(); 
		virtual ~ViolaWellsMutualInformationMetric() {}

	private:

		ViolaWellsMutualInformationMetric( const Self& );	// purposely not implemented
		void operator=( const Self& );										// purposely not implemented
		
	}; // end class ViolaWellsMutualInformationMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxViolaWellsMutualInformationMetric.hxx"
#endif

#endif // end #ifndef __elxViolaWellsMutualInformationMetric_H__
