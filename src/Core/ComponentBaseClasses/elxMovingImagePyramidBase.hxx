/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
void
MovingImagePyramidBase<TElastix>
::BeforeRegistrationBase( void )
{
  /** Call SetMovingSchedule.*/
  this->SetMovingSchedule();

} // end BeforeRegistrationBase()


/**
 * ********************** SetMovingSchedule **********************
 */

template <class TElastix>
void
MovingImagePyramidBase<TElastix>
::SetMovingSchedule( void )
{
  /** Get the ImageDimension. */
  const unsigned int MovingImageDimension = InputImageType::ImageDimension;

  /** Read numberOfResolutions. */
  unsigned int numberOfResolutions = 0;
  this->m_Configuration->ReadParameter( numberOfResolutions,
    "NumberOfResolutions", 0, true );
  if ( numberOfResolutions == 0 )
  {
    xl::xout["error"] << "ERROR: NumberOfResolutions not specified!" << std::endl;
  }
  /** \todo quit program? Actually this check should be in the ::BeforeAll() method. */

  /** Create a default movingSchedule. Set the numberOfLevels first. */
  this->GetAsITKBaseType()->SetNumberOfLevels( numberOfResolutions );
  ScheduleType movingSchedule = this->GetAsITKBaseType()->GetSchedule();

  /** Set the movingPyramidSchedule to the MovingImagePyramidSchedule given 
   * in the parameter-file. The following parameter file fields can be used:
   * ImagePyramidSchedule
   * MovingImagePyramidSchedule
   * MovingImagePyramid<i>Schedule, for the i-th moving image pyramid used. 
   */
  bool found = true;
  for ( unsigned int i = 0; i < numberOfResolutions; i++ )
  {
    for ( unsigned int j = 0; j < MovingImageDimension; j++ )
    {
      bool ijfound = false;
      const unsigned int entrynr = i * MovingImageDimension + j;
      ijfound |= this->m_Configuration->ReadParameter( movingSchedule[ i ][ j ],
        "ImagePyramidSchedule", entrynr, false );
      ijfound |= this->m_Configuration->ReadParameter( movingSchedule[ i ][ j ],
        "MovingImagePyramidSchedule", entrynr, false );
      ijfound |= this->m_Configuration->ReadParameter( movingSchedule[ i ][ j ],
        "Schedule", this->GetComponentLabel(), entrynr, -1, false );

      /** Remember if for at least one schedule element no value could be found. */
      found &= ijfound;

    } // end for MovingImageDimension
  } // end for numberOfResolutions

  if ( !found && this->GetConfiguration()->GetPrintErrorMessages() )
  {
    xl::xout["warning"] << "WARNING: the moving pyramid schedule is not fully specified!\n";
    xl::xout["warning"] << "  A default pyramid schedule is used." << std::endl;
  }
  else
  {
    /** Set the schedule into this class. */
    this->GetAsITKBaseType()->SetSchedule( movingSchedule );
  }

} // end SetMovingSchedule()


} // end namespace elastix

#endif // end #ifndef __elxMovingImagePyramidBase_hxx

