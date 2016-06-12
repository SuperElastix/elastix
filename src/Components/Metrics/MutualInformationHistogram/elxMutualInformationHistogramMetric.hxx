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
#ifndef __elxMutualInformationHistogramMetric_HXX__
#define __elxMutualInformationHistogramMetric_HXX__

#include "elxMutualInformationHistogramMetric.h"
#include "itkTimeProbe.h"


namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
MutualInformationHistogramMetric< TElastix >
::MutualInformationHistogramMetric()
{
}  // end Constructor

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
MutualInformationHistogramMetric< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of MutualInformationHistogramMetric metric took: "
    << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
MutualInformationHistogramMetric< TElastix >
::BeforeRegistration( void )
{
  /** This exception can be removed once this class is fully implemented. */
  itkExceptionMacro( << "ERROR: This class is not yet fully implemented." );

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
MutualInformationHistogramMetric< TElastix >
::BeforeEachResolution( void )
{
  /** \todo adapt SecondOrderRegularisationMetric.
   * Set alpha, which balances the similarity and deformation energy
   * E_total = (1-alpha)*E_sim + alpha*E_def.
   * metric->SetAlpha( config.GetAlpha(level) );
   */

  const unsigned int nrOfParameters = this->m_Elastix->GetElxTransformBase()
    ->GetAsITKBaseType()->GetNumberOfParameters();
  ScalesType derivativeStepLengthScales( nrOfParameters );
  derivativeStepLengthScales.Fill( 1.0 );

  /** Read the parameters from the ParameterFile.*
  this->m_Configuration->ReadParameter( histogramSize, "HistogramSize", 0 );
  this->m_Configuration->ReadParameter( paddingValue, "PaddingValue", 0 );
  this->m_Configuration->ReadParameter( derivativeStepLength, "DerivativeStepLength", 0 );
  this->m_Configuration->ReadParameter( derivativeStepLengthScales, "DerivativeStepLengthScales", 0 );
  this->m_Configuration->ReadParameter( upperBoundIncreaseFactor, "UpperBoundIncreaseFactor", 0 );
  this->m_Configuration->ReadParameter( usePaddingValue, "UsePaddingValue", 0 );
  */
  /** Set them.*/
  //this->SetHistogramSize( ?? );
  //this->SetPaddingValue( ?? );
  //this->SetDerivativeStepLength( ?? );
  this->SetDerivativeStepLengthScales( derivativeStepLengthScales );
  //this->SetUpperBoundIncreaseFactor( ?? );
  //this->SetUsePaddingValue( ?? );

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxMutualInformationHistogramMetric_HXX__
