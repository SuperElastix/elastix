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

#ifndef __elxResampleInterpolatorBase_hxx
#define __elxResampleInterpolatorBase_hxx

#include "elxResampleInterpolatorBase.h"

namespace elastix
{

/**
 * ******************* ReadFromFile *****************************
 */

template< class TElastix >
void
ResampleInterpolatorBase< TElastix >
::ReadFromFile( void )
{
  // nothing, but must be here

} // end ReadFromFile


/**
 * ******************* WriteToFile ******************************
 */

template< class TElastix >
void
ResampleInterpolatorBase< TElastix >
::WriteToFile( void ) const
{
  /** Write ResampleInterpolator specific things. */
  xl::xout[ "transpar" ] << "\n// ResampleInterpolator specific" << std::endl;

  /** Write the name of the resample-interpolator. */
  xl::xout[ "transpar" ] << "(ResampleInterpolator \""
                         << this->elxGetClassName() << "\")" << std::endl;

} // end WriteToFile()


/**
 * ******************* CreateTransformParametersMap ****************
 */

template< class TElastix >
void
ResampleInterpolatorBase< TElastix >
::CreateTransformParametersMap( ParameterMapType * paramsMap ) const
{
  std::string                parameterName;
  std::vector< std::string > parameterValues;

  /** Write the name of this transform. */
  parameterName = "ResampleInterpolator";
  parameterValues.push_back( this->elxGetClassName() );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

} // end CreateTransformParametersMap()


} // end namespace elastix

#endif // end #ifndef __elxResampleInterpolatorBase_hxx
