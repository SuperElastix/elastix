#ifndef __elxFixedImagePyramidBase_h
#define __elxFixedImagePyramidBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkObject.h"
#include "itkMultiResolutionPyramidImageFilter.h"


namespace elastix
{
using namespace itk;


	/**
	 * ******************** FixedImagePyramidBase *******************
	 *
	 * The FixedImagePyramidBase class ....
	 */

	template <class TElastix>
		class FixedImagePyramidBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard.*/
		typedef FixedImagePyramidBase				Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Typedefs inherited from the superclass.*/
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;

		/** Typedefs inherited from Elastix.*/
		typedef typename ElastixType::FixedInternalImageType	InputImageType;
		typedef typename ElastixType::FixedInternalImageType	OutputImageType;
			
		/** Used in the function GetFixedSchedule */
		typedef typename ElastixType::MovingInternalImageType MovingImageType;
		
		/** Other typedef's.*/
		typedef MultiResolutionPyramidImageFilter<
			InputImageType, OutputImageType >									ITKBaseType;

		/** Typedef's from ITKBaseType.*/
		typedef typename ITKBaseType::ScheduleType					ScheduleType;

		/** Cast to ITKBaseType.*/
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}

		/** Methods that have to be present everywhere.*/
		virtual void BeforeRegistrationBase(void);

		/** Method for setting the schedule.*/
		virtual void SetFixedSchedule(void);

	protected:

		FixedImagePyramidBase() {}
		virtual ~FixedImagePyramidBase() {}
		
	private:

		FixedImagePyramidBase( const Self& );	// purposely not implemented
		void operator=( const Self& );				// purposely not implemented

	}; // end class FixedImagePyramidBase


} // end namespace elastix



#ifndef ITK_MANUAL_INSTANTIATION
#include "elxFixedImagePyramidBase.hxx"
#endif


#endif // end #ifndef __elxFixedImagePyramidBase_h

