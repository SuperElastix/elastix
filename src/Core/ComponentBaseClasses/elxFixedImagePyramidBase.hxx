#ifndef __elxFixedImagePyramidBase_hxx
#define __elxFixedImagePyramidBase_hxx

#include "elxFixedImagePyramidBase.h"

namespace elastix
{
	using namespace itk;


	/**
	 * ******************* BeforeRegistrationBase *******************
	 */

	template <class TElastix>
		void FixedImagePyramidBase<TElastix>
		::BeforeRegistrationBase(void)
	{
		/** Call SetFixedSchedule.*/
		this->SetFixedSchedule();
		
	} // end BeforeRegistrationBase


	/**
	 * ********************** SetFixedSchedule **********************
	 */

	template <class TElastix>
		void FixedImagePyramidBase<TElastix>
		::SetFixedSchedule(void)
	{
		/** Get the ImageDimension.*/
		unsigned int FixedImageDimension = InputImageType::ImageDimension;
		unsigned int MovingImageDimension = MovingImageType::ImageDimension;

		/** Read numberOfResolutions.*/
		unsigned int numberOfResolutions = 0;
		m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0, true );
		if ( numberOfResolutions == 0 )
		{
			xl::xout["error"] << "ERROR: NumberOfResolutions not specified!" << std::endl;
		}
		// TO DO: quit program? Actually this check should be in the ::BeforeAll() method.

		/** Create a fixedSchedule.*/
		ScheduleType fixedSchedule( numberOfResolutions, FixedImageDimension );
		fixedSchedule.Fill( 0 );

		/** Are FixedPyramidSchedule and/or MovingPyramidSchedule available
		 * in the parameter-file?
		 */
		unsigned int temp_fix = 0;
		unsigned int temp_mov = 0;
		m_Configuration->ReadParameter( temp_fix, "FixedPyramidSchedule", 0, true );
		m_Configuration->ReadParameter( temp_mov, "MovingPyramidSchedule", 0, true );

		/** If the FixedPyramidSchedule exists:*/
		if ( temp_fix != 0 )
		{
			/** In this case set the fixedPyramidSchedule to the
			 * FixedPyramidSchedule given in the parameter-file.
			 */
			for ( unsigned int i = 0; i < numberOfResolutions; i++ )
			{
				for ( unsigned int j = 0; j < FixedImageDimension; j++ )
				{
					m_Configuration->ReadParameter(
						fixedSchedule[ j ][ i ],
						"FixedPyramidSchedule",
						i * numberOfResolutions + j );
				} // end for FixedImageDimension
			} // end for numberOfResolutions

			/** Set the schedule into this class.*/
			this->GetAsITKBaseType()->SetSchedule( fixedSchedule );

		}
		/** If only the MovingPyramidSchedule exists:*/
		else if ( temp_fix == 0 && temp_mov != 0 && FixedImageDimension == MovingImageDimension )
		{
			/** In this case set the fixedPyramidSchedule to the
			 * MovingPyramidSchedule given in the parameter-file.
			 */
			for ( unsigned int i = 0; i < numberOfResolutions; i++ )
			{
				for ( unsigned int j = 0; j < FixedImageDimension; j++ )
				{
					m_Configuration->ReadParameter(
						fixedSchedule[ j ][ i ],
						"MovingPyramidSchedule",
						i * numberOfResolutions + j );
				} // end for FixedImageDimension
			} // end for numberOfResolutions

			/** Set the schedule into this class.*/
			this->GetAsITKBaseType()->SetSchedule( fixedSchedule );

		}
		else if ( temp_fix == 0 && temp_mov != 0 && FixedImageDimension != MovingImageDimension )
		{
			xl::xout["warning"] << "WARNING: FixedImagePyramidSchedule is not specified!" << std::endl;
			xl::xout["warning"] << "A default schedule is assumed. " << std::endl;
		}
		/** If both PyramidSchedule's don't exist:*/
		else
		{
			/** In this case set the fixedPyramidSchedule to the
			 * default schedule.
			 */
			this->GetAsITKBaseType()->SetNumberOfLevels( numberOfResolutions );

		} // end if

	} // end SetFixedSchedule


} // end namespace elastix

#endif // end #ifndef __elxFixedImagePyramidBase_hxx

