#ifndef __elxMovingImagePyramidBase_h
#define __elxMovingImagePyramidBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkObject.h"

#include "itkMultiResolutionPyramidImageFilter.h"


namespace elastix
{
using namespace itk;


	/**
	 * ******************** MovingImagePyramidBase ******************
	 *
	 * The MovingImagePyramidBase class ....
	 */

	template <class TElastix>
		class MovingImagePyramidBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard.*/
		typedef MovingImagePyramidBase			Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Typedefs inherited from the superclass.*/
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;

		/** Typedefs inherited from Elastix.*/
		typedef typename ElastixType::MovingInternalImageType		InputImageType;
		typedef typename ElastixType::MovingInternalImageType		OutputImageType;

		/** Typedef used by the function GetMovingSchedule */
		typedef typename ElastixType::FixedInternalImageType		FixedImageType;
		
		/** Other typedef's.*/
		typedef MultiResolutionPyramidImageFilter<
			InputImageType, OutputImageType >				ITKBaseType;

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
		virtual void SetMovingSchedule(void);

	protected:

		MovingImagePyramidBase() {}
		virtual ~MovingImagePyramidBase() {}
		
	private:

		MovingImagePyramidBase( const Self& );	// purposely not implemented
		void operator=( const Self& );					// purposely not implemented

	}; // end class MovingImagePyramidBase


} // end namespace elastix



#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMovingImagePyramidBase.hxx"
#endif

#endif // end #ifndef __elxMovingImagePyramidBase_h

