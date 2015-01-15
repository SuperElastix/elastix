/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __elxReducedDimensionBSplineResampleInterpolator_hxx
#define __elxReducedDimensionBSplineResampleInterpolator_hxx

#include "elxReducedDimensionBSplineResampleInterpolator.h"

namespace elastix
{

/*
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
ReducedDimensionBSplineResampleInterpolator< TElastix >
::BeforeRegistration( void )
{
  /** ReducedDimensionBSplineResampleInterpolator specific. */

  /** Set the SplineOrder, default = 3. */
  unsigned int splineOrder = 3;

  /** Read the desired splineOrder from the parameterFile. */
  bool oldstyle = this->m_Configuration->ReadParameter( splineOrder,
    "FinalReducedDimensionBSplineInterpolationOrder", 0, false );
  if( oldstyle )
  {
    xout[ "warning" ] << "WARNING: FinalReducedDimensionBSplineInterpolator parameter is depecrated. "
                      << "Replace it by FinalBSplineInterpolationOrder" << std::endl;
  }
  this->m_Configuration->ReadParameter( splineOrder,
    "FinalBSplineInterpolationOrder", 0 );

  /** Set the splineOrder in the superclass. */
  this->SetSplineOrder( splineOrder );

} // end BeforeRegistration()


/*
 * ******************* ReadFromFile  ****************************
 */

template< class TElastix >
void
ReducedDimensionBSplineResampleInterpolator< TElastix >
::ReadFromFile( void )
{
  /** Call ReadFromFile of the ResamplerBase. */
  this->Superclass2::ReadFromFile();

  /** ReducedDimensionBSplineResampleInterpolator specific. */

  /** Set the SplineOrder, default = 3. */
  unsigned int splineOrder = 3;

  /** Read the desired splineOrder from the parameterFile. */
  bool oldstyle = this->m_Configuration->ReadParameter( splineOrder,
    "FinalReducedDimensionBSplineInterpolationOrder", 0, false );
  if( oldstyle )
  {
    xout[ "warning" ] << "WARNING: FinalReducedDimensionBSplineInterpolator parameter is depecrated. "
                      << "Replace it by FinalBSplineInterpolationOrder" << std::endl;
  }
  this->m_Configuration->ReadParameter( splineOrder,
    "FinalBSplineInterpolationOrder", 0 );

  /** Set the splineOrder in the superclass. */
  this->SetSplineOrder( splineOrder );

} // end ReadFromFile()


/**
 * ******************* WriteToFile ******************************
 */

template< class TElastix >
void
ReducedDimensionBSplineResampleInterpolator< TElastix >
::WriteToFile( void ) const
{
  /** Call WriteToFile of the ResamplerBase. */
  this->Superclass2::WriteToFile();

  /** The ReducedDimensionBSplineResampleInterpolator adds: */

  /** Write the FinalBSplineInterpolationOrder. */
  xout[ "transpar" ] << "(FinalBSplineInterpolationOrder "
                     << this->GetSplineOrder() << ")" << std::endl;

} // end WriteToFile()


} // end namespace elastix

#endif // end #ifndef __elxReducedDimensionBSplineResampleInterpolator_hxx
