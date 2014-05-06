/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxTransformBendingEnergyPenaltyTerm_HXX__
#define __elxTransformBendingEnergyPenaltyTerm_HXX__

#include "elxTransformBendingEnergyPenaltyTerm.h"
#include "itkTimeProbe.h"


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
TransformBendingEnergyPenalty< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of TransformBendingEnergy metric took: "
    << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
TransformBendingEnergyPenalty< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Set the number of samples used to compute the SelfHessian */
  unsigned int numberOfSamplesForSelfHessian = 100000;
  this->GetConfiguration()->ReadParameter( numberOfSamplesForSelfHessian,
    "NumberOfSamplesForSelfHessian", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfSamplesForSelfHessian( numberOfSamplesForSelfHessian );

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxTransformBendingEnergyPenaltyTerm_HXX__
