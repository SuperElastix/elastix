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
		/** Call SetMovingSchedule.*/
		this->SetMovingSchedule();
		
	} // end BeforeRegistrationBase


  /**
	 * ********************** SetMovingSchedule **********************
	 */

	template <class TElastix>
		void MovingImagePyramidBase<TElastix>
		::SetMovingSchedule(void)
	{
		/** Get the ImageDimension. */
		const unsigned int MovingImageDimension = InputImageType::ImageDimension;
		
		/** Read numberOfResolutions. */
		unsigned int numberOfResolutions = 0;
		this->m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0, true );
		if ( numberOfResolutions == 0 )
		{
			xl::xout["error"] << "ERROR: NumberOfResolutions not specified!" << std::endl;
		}
		/** \todo quit program? Actually this check should be in the ::BeforeAll() method. */

		/** Create a movingSchedule. */
		//ScheduleType movingSchedule( numberOfResolutions, MovingImageDimension );
    this->GetAsITKBaseType()->SetNumberOfLevels( numberOfResolutions );
    ScheduleType movingSchedule = this->GetAsITKBaseType()->GetSchedule();
		
		/** Always set the numberOfLevels first. */
		this->GetAsITKBaseType()->SetNumberOfLevels( numberOfResolutions );

		/** Set the movingPyramidSchedule to the MovingImagePyramidSchedule given 
     * in the parameter-file.	The following parameter file fields can be used:
     * ImagePyramidSchedule
     * MovingImagePyramidSchedule
     * MovingImagePyramid<i>Schedule, for the i-th moving image pyramid used. 
     */
    int ret = 0;
		for ( unsigned int i = 0; i < numberOfResolutions; i++ )
		{
			for ( unsigned int j = 0; j < MovingImageDimension; j++ )
			{
        int ijret = 1;
        const unsigned int entrynr = i * MovingImageDimension + j;
				ijret &= this->m_Configuration->ReadParameter( movingSchedule[ i ][ j ],
					"ImagePyramidSchedule", entrynr, true );
        ijret &= this->m_Configuration->ReadParameter( movingSchedule[ i ][ j ],
					"MovingImagePyramidSchedule", entrynr, true );
        ijret &= this->m_Configuration->ReadParameter( movingSchedule[ i ][ j ],
					"Schedule", this->GetComponentLabel(), entrynr, -1, true );

        /** Remember if for at least one schedule element no value could be found. */
        ret |= ijret;
      } // end for MovingImageDimension
		} // end for numberOfResolutions
    if (ret && !this->GetConfiguration()->GetSilent() )
    {
      xl::xout["warning"] << "WARNING: the moving pyramid schedule is not fully specified!" << std::endl;
      xl::xout["warning"] << "A default pyramid schedule is used." << std::endl;
    }
    else
    {
      /** Set the schedule into this class. */
		  this->GetAsITKBaseType()->SetSchedule( movingSchedule );
    }
		
	} // end SetMovingSchedule()


} // end namespace elastix

#endif // end #ifndef __elxMovingImagePyramidBase_hxx

