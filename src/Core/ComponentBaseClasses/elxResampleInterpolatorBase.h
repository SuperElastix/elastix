#ifndef __elxResampleInterpolatorBase_h
#define __elxResampleInterpolatorBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkInterpolateImageFunction.h"


namespace elastix
{	
  using namespace itk;


	/**
	 * ***************** ResampleInterpolatorBase *******************
	 *
	 * This class 
	 */

	template <class TElastix>
		class ResampleInterpolatorBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard stuff.*/
		typedef ResampleInterpolatorBase		Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Typedef's from superclass.*/
		typedef typename Superclass::ElastixType					ElastixType;
		typedef typename Superclass::ElastixPointer				ElastixPointer;
		typedef typename Superclass::ConfigurationType		ConfigurationType;
		typedef typename Superclass::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass::RegistrationType			RegistrationType;
		typedef typename Superclass::RegistrationPointer	RegistrationPointer;

		/** Typedef's from elastix.*/
		typedef typename ElastixType::MovingImageType			InputImageType;
		typedef typename ElastixType::CoordRepType				CoordRepType;

		/** Other typedef's.*/
		typedef InterpolateImageFunction< 
			InputImageType, CoordRepType >									ITKBaseType;

		/** ...*/
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}

		/** Methods that have to be present everywhere.*/
		virtual int BeforeAllTransformix(void){ return 0;};

		/** Read/Write ResampleInterpolator specific things from/to file.*/
		virtual void WriteToFile(void);
		virtual void ReadFromFile(void);

	protected:

		ResampleInterpolatorBase() {}
		virtual ~ResampleInterpolatorBase() {}

	private:

		ResampleInterpolatorBase( const Self& );	// purposely not implemented
		void operator=( const Self& );						// purposely not implemented

	}; // end class ResampleInterpolatorBase


} //end namespace elastix



#ifndef ITK_MANUAL_INSTANTIATION
#include "elxResampleInterpolatorBase.hxx"
#endif

#endif // end #ifndef __elxResampleInterpolatorBase_h
