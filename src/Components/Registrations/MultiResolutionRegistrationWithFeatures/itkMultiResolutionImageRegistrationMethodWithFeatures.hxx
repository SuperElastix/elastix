/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef _itkMultiResolutionImageRegistrationMethodWithFeatures_hxx
#define _itkMultiResolutionImageRegistrationMethodWithFeatures_hxx

#include "itkMultiResolutionImageRegistrationMethodWithFeatures.h"

#include "itkContinuousIndex.h"
#include "vnl/vnl_math.h"


namespace itk
{

  /*
   * ****************** CheckPyramids ******************
   */

  template < typename TFixedImage, typename TMovingImage >
  void
  MultiResolutionImageRegistrationMethodWithFeatures<TFixedImage,TMovingImage>
  ::CheckPyramids( void ) throw (ExceptionObject)
  {
    /** Check if at least one of the following are provided. */
    if ( this->GetFixedImage() == 0 )
    {
      itkExceptionMacro( << "FixedImage is not present" );
    }
    if ( this->GetMovingImage() == 0 )
    {
      itkExceptionMacro( << "MovingImage is not present" );
    }
    if ( this->GetFixedImagePyramid() == 0 )
    {
      itkExceptionMacro( << "Fixed image pyramid is not present" );
    }
    if ( this->GetMovingImagePyramid() == 0 )
    {
      itkExceptionMacro( << "Moving image pyramid is not present" );
    }

    /** Check if the number if fixed/moving pyramids == nr of fixed/moving images,
     * and whether the number of fixed image regions == the number of fixed images.
     */
    if ( this->GetNumberOfFixedImagePyramids() != this->GetNumberOfFixedImages() )
    {
      itkExceptionMacro( << "The number of fixed image pyramids should equal the number of fixed images" );
    }
    if ( this->GetNumberOfMovingImagePyramids() != this->GetNumberOfMovingImages() )
    {
      itkExceptionMacro( << "The number of moving image pyramids should equal the number of moving images" );
    }
    if ( this->GetNumberOfFixedImageRegions() != this->GetNumberOfFixedImages() )
    {
      itkExceptionMacro( << "The number of fixed image regions should equal the number of fixed image" );
    }

  } // end CheckPyramids()


} // end namespace itk

#endif // end #ifndef _itkMultiResolutionImageRegistrationMethodWithFeatures_hxx
