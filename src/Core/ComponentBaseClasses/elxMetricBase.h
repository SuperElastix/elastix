#ifndef __elxMetricBase_h
#define __elxMetricBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkImageToImageMetric.h"


namespace elastix
{
using namespace itk;


	/**
	 * ******************** MetricBase ******************************
	 *
	 * The MetricBase class ....
	 */

	template <class TElastix>
		class MetricBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard.*/
		typedef MetricBase									Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Typedef's inherited from Elastix.*/
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;

		/** Other typedef's.*/	
		typedef typename ElastixType::FixedInternalImageType FixedImageType;
		typedef typename ElastixType::MovingInternalImageType MovingImageType;
		
		/** ITKBaseType.*/
		typedef ImageToImageMetric<
			FixedImageType, MovingImageType >				ITKBaseType;

		/** Cast to ITKBaseType.*/
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}

	protected:

		MetricBase() {}
		virtual ~MetricBase() {}

	private:

		MetricBase( const Self& );			// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

	}; // end class MetricBase


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMetricBase.hxx"
#endif

#endif // end #ifndef __elxMetricBase_h

