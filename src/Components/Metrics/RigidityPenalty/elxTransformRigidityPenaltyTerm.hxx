/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxTransformRigidityPenaltyTerm_HXX__
#define __elxTransformRigidityPenaltyTerm_HXX__

#include "elxTransformRigidityPenaltyTerm.h"


namespace elastix
{

 /**
  * ******************* BeforeRegistration ***********************
  */

template <class TElastix>
void
TransformRigidityPenalty<TElastix>
::BeforeRegistration( void )
{
  /** Read the fixed rigidity image if desired. */
  std::string fixedRigidityImageName = "";
  this->GetConfiguration()->ReadParameter( fixedRigidityImageName,
    "FixedRigidityImageName", this->GetComponentLabel(), 0, -1, false );
  
  typedef ImageFileReader<RigidityImageType> RigidityImageReaderType;
  typename RigidityImageReaderType::Pointer fixedRigidityReader;

  if ( fixedRigidityImageName != "" )
  {
    /** Use the FixedRigidityImage. */
    this->SetUseFixedRigidityImage( true );

    /** Create the reader and set the filename. */
    fixedRigidityReader = RigidityImageReaderType::New();
    fixedRigidityReader->SetFileName( fixedRigidityImageName.c_str() );

    /** Do the reading. */
    try
    {
      fixedRigidityReader->Update();
    }
    catch( ExceptionObject & excp )
    {
      /** Add information to the exception. */
      excp.SetLocation( "MattesMutualInformationWithRigidityPenalty - BeforeRegistration()" );
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while reading the fixed rigidity image.\n";
      excp.SetDescription( err_str );
      /** Pass the exception to an higher level. */
      throw excp;
    }

    /** Set the fixed rigidity image into the superclass. */
    this->SetFixedRigidityImage( fixedRigidityReader->GetOutput() );
  }
  else
  {
    this->SetUseFixedRigidityImage( false );
  }

  /** Read the moving rigidity image if desired. */
  std::string movingRigidityImageName = "";
  this->GetConfiguration()->ReadParameter( movingRigidityImageName,
    "MovingRigidityImageName", this->GetComponentLabel(), 0, -1, false );

  typename RigidityImageReaderType::Pointer movingRigidityReader;
  if ( movingRigidityImageName != "" )
  {
    /** Use the movingRigidityImage. */
    this->SetUseMovingRigidityImage( true );

    /** Create the reader and set the filename. */
    movingRigidityReader = RigidityImageReaderType::New();
    movingRigidityReader->SetFileName( movingRigidityImageName.c_str() );

    /** Do the reading. */
    try
    {
      movingRigidityReader->Update();
    }
    catch( ExceptionObject & excp )
    {
      /** Add information to the exception. */
      excp.SetLocation( "MattesMutualInformationWithRigidityPenalty - BeforeRegistration()" );
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while reading the moving rigidity image.\n";
      excp.SetDescription( err_str );
      /** Pass the exception to an higher level. */
      throw excp;
    }

    /** Set the moving rigidity image into the superclass. */
    this->SetMovingRigidityImage( movingRigidityReader->GetOutput() );
  }
  else
  {
    this->SetUseMovingRigidityImage( false );
  }

  /** Important check: at least one rigidity image must be given. */
  if ( fixedRigidityImageName == "" && movingRigidityImageName == "" )
  {
    xl::xout["warning"] << "WARNING: FixedRigidityImageName and "
      << "MovingRigidityImage are both not supplied.\n"
      << "  The rigidity penalty term is evaluated on entire input "
      << "transform domain." << std::endl;
  }

  /** Add target cells to xout["iteration"]. */
  xout["iteration"].AddTargetCell("Metric-LC");
  xout["iteration"].AddTargetCell("Metric-OC");
  xout["iteration"].AddTargetCell("Metric-PC");

  /** Format the metric as floats. */
  xl::xout["iteration"]["Metric-LC"] << std::showpoint << std::fixed
    << std::setprecision( 10 );
  xl::xout["iteration"]["Metric-OC"] << std::showpoint << std::fixed
    << std::setprecision( 10 );
  xl::xout["iteration"]["Metric-PC"] << std::showpoint << std::fixed
    << std::setprecision( 10 );

} // end BeforeRegistration()


/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
TransformRigidityPenalty<TElastix>
::Initialize( void ) throw (ExceptionObject)
{
  /** Create and start a timer. */
  TimerPointer timer = TimerType::New();
  timer->StartTimer();

  /** Initialize this class with the Superclass initializer. */
  this->Superclass1::Initialize();

  /** Check stuff. */
  this->CheckUseAndCalculationBooleans();

  /** Stop and print the timer. */
  timer->StopTimer();
  elxout << "Initialization of TransformRigidityPenalty term took: "
    << static_cast<long>( timer->GetElapsedClockSec() * 1000 )
    << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
TransformRigidityPenalty<TElastix>
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Get and set the dilateRigidityImages. */
  bool dilateRigidityImages = true;
  this->GetConfiguration()->ReadParameter( dilateRigidityImages, 
    "DilateRigidityImages", this->GetComponentLabel(), level, 0 );
  this->SetDilateRigidityImages( dilateRigidityImages );

  /** Get and set the dilationRadiusMultiplier. */
  double dilationRadiusMultiplier = 1.0;
  this->GetConfiguration()->ReadParameter( dilationRadiusMultiplier,
    "DilationRadiusMultiplier", this->GetComponentLabel(), level, 0 );
  this->SetDilationRadiusMultiplier( dilationRadiusMultiplier );

  /** Get and set the usage of the linearity condition part. */ 
  bool useLinearityCondition = true;
  this->GetConfiguration()->ReadParameter( useLinearityCondition,
    "UseLinearityCondition", this->GetComponentLabel(), level, 0 );
  this->SetUseLinearityCondition( useLinearityCondition );

  /** Get and set the usage of the orthonormality condition part. */
  bool useOrthonormalityCondition = true;
  this->GetConfiguration()->ReadParameter( useOrthonormalityCondition,
    "UseOrthonormalityCondition", this->GetComponentLabel(), level, 0 );
  this->SetUseOrthonormalityCondition( useOrthonormalityCondition );

  /** Set the usage of the properness condition part. */
  bool usePropernessCondition = true;
  this->GetConfiguration()->ReadParameter( usePropernessCondition,
    "UsePropernessCondition", this->GetComponentLabel(), level, 0 );
  this->SetUsePropernessCondition( usePropernessCondition );

  /** Set the calculation of the linearity condition part. */
  bool calculateLinearityCondition = true;
  this->GetConfiguration()->ReadParameter( calculateLinearityCondition,
    "CalculateLinearityCondition", this->GetComponentLabel(), level, 0 );
  this->SetCalculateLinearityCondition( calculateLinearityCondition );

  /** Set the calculation of the orthonormality condition part. */
  bool calculateOrthonormalityCondition = true;
  this->GetConfiguration()->ReadParameter( calculateOrthonormalityCondition,
    "CalculateOrthonormalityCondition", this->GetComponentLabel(), level, 0 );
  this->SetCalculateOrthonormalityCondition( calculateOrthonormalityCondition );

  /** Set the calculation of the properness condition part. */
  bool calculatePropernessCondition = true;
  this->GetConfiguration()->ReadParameter( calculatePropernessCondition,
    "CalculatePropernessCondition", this->GetComponentLabel(), level, 0 );
  this->SetCalculatePropernessCondition( calculatePropernessCondition );

  /** Set the LinearityConditionWeight of this level. */
  double linearityConditionWeight = 1.0;
  this->m_Configuration->ReadParameter( linearityConditionWeight,
    "LinearityConditionWeight", this->GetComponentLabel(), level, 0 );
  this->SetLinearityConditionWeight( linearityConditionWeight );

  /** Set the orthonormalityConditionWeight of this level. */
  double orthonormalityConditionWeight = 1.0;
  this->m_Configuration->ReadParameter( orthonormalityConditionWeight,
    "OrthonormalityConditionWeight", this->GetComponentLabel(), level, 0 );
  this->SetOrthonormalityConditionWeight( orthonormalityConditionWeight );

  /** Set the propernessConditionWeight of this level. */
  double propernessConditionWeight = 1.0;
  this->m_Configuration->ReadParameter( propernessConditionWeight,
    "PropernessConditionWeight", this->GetComponentLabel(), level, 0 );
  this->SetPropernessConditionWeight( propernessConditionWeight );

} // end BeforeEachResolution()


/**
 * ***************AfterEachIteration ****************************
 */

template <class TElastix>
void
TransformRigidityPenalty<TElastix>
::AfterEachIteration( void )
{
  /** Print some information. */
  xl::xout["iteration"]["Metric-LC"] <<
    this->GetLinearityConditionValue();
  xl::xout["iteration"]["Metric-OC"] <<
    this->GetOrthonormalityConditionValue();
  xl::xout["iteration"]["Metric-PC"] <<
    this->GetPropernessConditionValue();

} // end AfterEachIteration()


} // end namespace elastix


#endif // end #ifndef __elxTransformRigidityPenaltyTerm_HXX__

