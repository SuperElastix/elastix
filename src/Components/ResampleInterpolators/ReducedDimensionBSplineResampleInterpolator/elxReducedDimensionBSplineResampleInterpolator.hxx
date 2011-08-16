/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxReducedDimensionBSplineResampleInterpolator_hxx
#define __elxReducedDimensionBSplineResampleInterpolator_hxx

#include "elxReducedDimensionBSplineResampleInterpolator.h"

namespace elastix
{
  using namespace itk;

/*
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
ReducedDimensionBSplineResampleInterpolator<TElastix>
::BeforeRegistration( void )
{
  /** ReducedDimensionBSplineResampleInterpolator specific. */

  /** Set the SplineOrder, default = 3. */
  unsigned int splineOrder = 3;

  /** Read the desired splineOrder from the parameterFile. */
  this->m_Configuration->ReadParameter( splineOrder,
    "FinalReducedDimensionBSplineInterpolationOrder", 0 );

  /** Set the splineOrder in the superclass. */
  this->SetSplineOrder( splineOrder );

} // end BeforeRegistration()


/*
 * ******************* ReadFromFile  ****************************
 */

template <class TElastix>
void
ReducedDimensionBSplineResampleInterpolator<TElastix>
::ReadFromFile( void )
{
  /** Call ReadFromFile of the ResamplerBase. */
  this->Superclass2::ReadFromFile();

  /** ReducedDimensionBSplineResampleInterpolator specific. */

  /** Set the SplineOrder, default = 3. */
  unsigned int splineOrder = 3;

  /** Read the desired splineOrder from the parameterFile. */
  this->m_Configuration->ReadParameter( splineOrder,
    "FinalReducedDimensionBSplineInterpolationOrder", 0 );

  /** Set the splineOrder in the superclass. */
  this->SetSplineOrder( splineOrder );

} // end ReadFromFile()


/**
 * ******************* WriteToFile ******************************
 */

template <class TElastix>
void
ReducedDimensionBSplineResampleInterpolator<TElastix>
::WriteToFile( void ) const
{
  /** Call WriteToFile of the ResamplerBase. */
  this->Superclass2::WriteToFile();

  /** The ReducedDimensionBSplineResampleInterpolator adds: */

  /** Write the FinalBSplineInterpolationOrder. */
  xout["transpar"] << "(FinalReducedDimensionBSplineInterpolationOrder "
    << this->GetSplineOrder() << ")" << std::endl;

} // end WriteToFile()


} // end namespace elastix

#endif // end #ifndef __elxReducedDimensionBSplineResampleInterpolator_hxx

