#ifndef __elxMovingImagePyramidBase_hxx
#define __elxMovingImagePyramidBase_hxx

#include "elxMovingImagePyramidBase.h"

namespace elastix
{
	using namespace itk;


	/**
	 * ******************* BeforeRegistrationBase *******************
	 */

	template <class TElastix>
		void MovingImagePyramidBase<TElastix>
		::BeforeRegistrationBase(void)
	{
		/** Call SetFixedSchedule.*/
		this->SetMovingSchedule();
		
	} // end BeforeRegistrationBase


	/**
	 * ********************** SetMovingSchedule *********************
	 */

	template <class TElastix>
		void MovingImagePyramidBase<TElastix>
		::SetMovingSchedule(void)
	{
		/** Get the ImageDimension.*/
		unsigned int MovingImageDimension = InputImageType::ImageDimension;
		unsigned int FixedImageDimension = FixedImageType::ImageDimension;

		/** Read numberOfResolutions.*/
		unsigned int numberOfResolutions = 0;
		this->m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0, true );
		if ( numberOfResolutions == 0 )
		{
			xl::xout["error"] << "ERROR: NumberOfResolutions not specified!" << std::endl;
		}
		/** \todo quit program? */

		/** Create a movingSchedule.*/
		ScheduleType movingSchedule( numberOfResolutions, MovingImageDimension );
		movingSchedule.Fill( 0 );

		/** Are FixedPyramidSchedule and/or MovingPyramidSchedule available
		 * in the parameter-file?
		 */
		unsigned int temp_fix = 0;
		unsigned int temp_mov = 0;
		this->m_Configuration->ReadParameter( temp_fix, "FixedPyramidSchedule", 0, true );
		this->m_Configuration->ReadParameter( temp_mov, "MovingPyramidSchedule", 0, true );

		/** If the MovingPyramidSchedule exists:*/
		if ( temp_mov != 0 )
		{
			/** In this case set the movingPyramidSchedule to the
			 * MovingPyramidSchedule given in the parameter-file.
			 */
			for ( unsigned int i = 0; i < numberOfResolutions; i++ )
			{
				for ( unsigned int j = 0; j < MovingImageDimension; j++ )
				{
					this->m_Configuration->ReadParameter(
						movingSchedule[ j ][ i ],
						"MovingPyramidSchedule",
						i * numberOfResolutions + j );
				} // end for MovingImageDimension
			} // end for numberOfResolutions

			/** Set the schedule into this class.*/
			this->GetAsITKBaseType()->SetSchedule( movingSchedule );

		}
		/** If only the FixedPyramidSchedule exists:*/
		else if ( temp_fix != 0 && temp_mov == 0 && FixedImageDimension == MovingImageDimension )
		{
			/** In this case set the movingPyramidSchedule to the
			 * FixedPyramidSchedule given in the parameter-file.
			 */
			for ( unsigned int i = 0; i < numberOfResolutions; i++ )
			{
				for ( unsigned int j = 0; j < MovingImageDimension; j++ )
				{
					this->m_Configuration->ReadParameter(
						movingSchedule[ j ][ i ],
						"FixedPyramidSchedule",
						i * numberOfResolutions + j );
				} // end for MovingImageDimension
			} // end for numberOfResolutions

			/** Set the schedule into this class.*/
			this->GetAsITKBaseType()->SetSchedule( movingSchedule );

		}
		else if ( temp_fix != 0 && temp_mov == 0 && FixedImageDimension != MovingImageDimension )
		{
			xl::xout["warning"] << "WARNING: MovingImagePyramidSchedule is not specified!" << std::endl;
			xl::xout["warning"] << "A default schedule is assumed. " << std::endl;
			}		
		/** If both PyramidSchedule's don't exist:*/
		else
		{
			/** In this case set the movingPyramidSchedule to the
			 * default schedule.
			 */
			this->GetAsITKBaseType()->SetNumberOfLevels( numberOfResolutions );

		} // end if

	} // end SetMovingSchedule


} // end namespace elastix

#endif // end #ifndef __elxMovingImagePyramidBase_hxx

