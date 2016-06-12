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
#ifndef __elxBSplineResampleInterpolator_hxx
#define __elxBSplineResampleInterpolator_hxx

#include "elxBSplineResampleInterpolator.h"

namespace elastix
{

/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
BSplineResampleInterpolator< TElastix >
::BeforeRegistration( void )
{
  /** BSplineResampleInterpolator specific. */

  /** Set the SplineOrder, default = 3. */
  unsigned int splineOrder = 3;

  /** Read the desired splineOrder from the parameterFile. */
  this->m_Configuration->ReadParameter( splineOrder,
    "FinalBSplineInterpolationOrder", 0 );

  /** Set the splineOrder in the superclass. */
  this->SetSplineOrder( splineOrder );

} // end BeforeRegistration()


/**
 * ******************* ReadFromFile  ****************************
 */

template< class TElastix >
void
BSplineResampleInterpolator< TElastix >
::ReadFromFile( void )
{
  /** Call ReadFromFile of the ResamplerBase. */
  this->Superclass2::ReadFromFile();

  /** BSplineResampleInterpolator specific. */

  /** Set the SplineOrder, default = 3. */
  unsigned int splineOrder = 3;

  /** Read the desired splineOrder from the parameterFile. */
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
BSplineResampleInterpolator< TElastix >
::WriteToFile( void ) const
{
  /** Call WriteToFile of the ResamplerBase. */
  this->Superclass2::WriteToFile();

  /** The BSplineResampleInterpolator adds: */

  /** Write the FinalBSplineInterpolationOrder. */
  xout[ "transpar" ] << "(FinalBSplineInterpolationOrder "
                     << this->GetSplineOrder() << ")" << std::endl;

} // end WriteToFile()


/**
 * ******************* CreateTransformParametersMap ******************************
 */

template< class TElastix >
void
BSplineResampleInterpolator< TElastix >
::CreateTransformParametersMap( ParameterMapType * paramsMap ) const
{
  std::string                parameterName;
  std::vector< std::string > parameterValues;
  char                       tmpValue[ 256 ];

  /** Call CreateTransformParametersMap of the ResamplerBase. */
  this->Superclass2::CreateTransformParametersMap( paramsMap );

  /** The BSplineResampleInterpolator adds: */

  /** Write the FinalBSplineInterpolationOrder. */
  parameterName = "FinalBSplineInterpolationOrder";
  sprintf( tmpValue, "%d", this->GetSplineOrder() );
  parameterValues.push_back( tmpValue );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

} // end CreateTransformParametersMap()


} // end namespace elastix

#endif // end #ifndef __elxBSplineResampleInterpolator_hxx
