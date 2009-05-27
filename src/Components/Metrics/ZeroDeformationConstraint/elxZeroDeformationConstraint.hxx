/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxZeroDeformationConstraint_HXX__
#define __elxZeroDeformationConstraint_HXX__

#include "elxZeroDeformationConstraint.h"


namespace elastix
{
using namespace itk;

  /**
   * ******************* Initialize ***********************
   */

  template <class TElastix>
    void ZeroDeformationConstraint<TElastix>
    ::Initialize(void) throw (ExceptionObject)
  {

    TimerPointer timer = TimerType::New();
    timer->StartTimer();
    this->Superclass1::Initialize();
    timer->StopTimer();
    elxout << "Initialization of ZeroDeformationConstraintMetric metric took: "
      << static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

  } // end Initialize


  /**
  * ***************** BeforeRegistration ***********************
  */

  template <class TElastix>
  void ZeroDeformationConstraint<TElastix>
    ::BeforeRegistration(void)
  {
    
    /** Add columns for the penalty term and lagrange multipliers. */
    xout["iteration"].AddTargetCell("5:Penalty");
    xout["iteration"].AddTargetCell("6:Lagrange");
    xout["iteration"].AddTargetCell("7:Infeasibility");
    xout["iteration"].AddTargetCell("8:MaxMagnitude");

    /** Format the metric and stepsize as floats */     
    xl::xout["iteration"]["5:Penalty"]  << std::showpoint << std::fixed;
    xl::xout["iteration"]["6:Lagrange"] << std::showpoint << std::fixed;
    xl::xout["iteration"]["7:Infeasibility"] << std::showpoint << std::fixed;
    xl::xout["iteration"]["8:MaxMagnitude"] << std::showpoint << std::fixed;

  }


  /**
   * ***************** BeforeEachResolution ***********************
   */

  template <class TElastix>
    void ZeroDeformationConstraint<TElastix>
    ::BeforeEachResolution(void)
  {
    /** Reset iteration number and num penalty term updates. */
    this->m_CurrentIteration = 0;
    this->m_NumPenaltyTermUpdates = 0;

    /** Reset previous maximum magnitude. */
    this->m_PreviousMaximumMagnitude = itk::NumericTraits< double >::max();

    /** Get the current resolution level. */
    unsigned int level = 
      ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    /** Get the initial lagrange multiplier. */
    double initialLagrangeMultiplier;
    this->GetConfiguration()->ReadParameter( initialLagrangeMultiplier,
      "InitialLagrangeMultiplier", this->GetComponentLabel(), level, 0 );
    this->SetInitialLangrangeMultiplier( initialLagrangeMultiplier );
    this->m_AverageLagrangeMultiplier = this->m_InitialLangrangeMultiplier;

    /** Get the initial penalty term multiplier. */
    this->GetConfiguration()->ReadParameter( this->m_InitialPenaltyTermMultiplier,
      "InitialPenaltyTermMultiplier", this->GetComponentLabel(), level, 0 );
    this->m_CurrentPenaltyTermMultiplier = this->m_InitialPenaltyTermMultiplier;
    
    /** Get the penalty term multiplier factor. */
    this->GetConfiguration()->ReadParameter( this->m_PenaltyTermMultiplierFactor,
      "PenaltyTermMultiplierFactor", this->GetComponentLabel(), level, 0 );

    /** Get the required decrease factor for the maximum magnitude. */
    this->GetConfiguration()->ReadParameter( this->m_RequiredConstraintDecreaseFactor,
      "RequiredConstraintDecreaseFactor", this->GetComponentLabel(), level, 0 );

    /** Get the number of subiterations after which the new lagrange multiplier
     * is determined.
     */
    this->GetConfiguration()->ReadParameter( this->m_NumSubIterations,
      "NumberOfSubIterations", this->GetComponentLabel(), level, 0 );

    /** Set moving image derivative scales. */
    this->SetUseMovingImageDerivativeScales( false );
    MovingImageDerivativeScalesType movingImageDerivativeScales;
    bool usescales = true;
    for ( unsigned int i = 0; i < MovingImageDimension; ++i )
    {
      usescales = usescales && this->GetConfiguration()->ReadParameter(
        movingImageDerivativeScales[ i ], "MovingImageDerivativeScales",
        this->GetComponentLabel(), i, -1, true );
    }
    if ( usescales )
    {
      this->SetUseMovingImageDerivativeScales( true );
      this->SetMovingImageDerivativeScales( movingImageDerivativeScales );
      elxout << "Multiplying moving image derivatives by: "
        << movingImageDerivativeScales << std::endl;
    }
    
  } // end BeforeEachResolution

  template <class TElastix>
  void ZeroDeformationConstraint<TElastix>
    ::AfterEachIteration(void)
  {
    this->m_CurrentIteration++;

    /** Print some information */
    xl::xout["iteration"]["5:Penalty"]  << this->m_CurrentPenaltyTermMultiplier;
    xl::xout["iteration"]["6:Lagrange"] << this->m_AverageLagrangeMultiplier;
    xl::xout["iteration"]["7:Infeasibility"] << this->GetCurrentInfeasibility();
    xl::xout["iteration"]["8:MaxMagnitude"] << this->GetCurrentMaximumMagnitude();

    if ( m_CurrentIteration % this->m_NumSubIterations == 0 )
    {
      this->DetermineNewLagrangeMultipliers();
      /** Check if maximum magnitude decreased enough. If not update penalty term multiplier. */
      if ( this->GetCurrentMaximumMagnitude() > this->m_RequiredConstraintDecreaseFactor * this->m_PreviousMaximumMagnitude )
      {
        this->m_CurrentPenaltyTermMultiplier = this->DetermineNewPenaltyTermMultiplier( this->m_NumPenaltyTermUpdates + 1 );
        this->m_NumPenaltyTermUpdates++;
      }
      this->m_PreviousMaximumMagnitude = this->GetCurrentMaximumMagnitude();
    }    
  } // end AfterEachIteration


  /**
  * ******************* DetermineNewLagrangeMultipliers *******************
  */

  template <class TElastix>
  void ZeroDeformationConstraint<TElastix>
    ::DetermineNewLagrangeMultipliers( )
  {

    m_AverageLagrangeMultiplier = 0.0;
    for ( std::vector< double >::size_type i = 0; i < this->m_CurrentLagrangeMultipliers.size(); ++i )
    {
      this->m_CurrentLagrangeMultipliers[ i ] = min( 0.0, this->m_CurrentLagrangeMultipliers[ i ] - 
        this->m_CurrentPenaltyTermValues[ i ] * this->GetCurrentPenaltyTermMultiplier() );
      m_AverageLagrangeMultiplier += this->m_CurrentLagrangeMultipliers[ i ];
    }
    m_AverageLagrangeMultiplier /= static_cast< double > ( this->m_CurrentLagrangeMultipliers.size() );

  } // end DetermineNewLagrangeMultipliers


  /**
  * ******************* DetermineNewPenaltyTermMultiplier *******************
  */

  template <class TElastix>
  double ZeroDeformationConstraint<TElastix>
    ::DetermineNewPenaltyTermMultiplier( const int iterationNumber ) const 
  {
    return static_cast< double > ( this->m_InitialPenaltyTermMultiplier * vcl_pow( 
      static_cast<double>( this->m_PenaltyTermMultiplierFactor ), static_cast<double> ( iterationNumber ) ) );
  } // end DetermineNewTermPenaltyMultiplier


} // end namespace elastix


#endif // end #ifndef __elxZeroDeformationConstraint_HXX__

