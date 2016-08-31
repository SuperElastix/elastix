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
#ifndef __elxTransformRigidityPenaltyTerm_HXX__
#define __elxTransformRigidityPenaltyTerm_HXX__

#include "elxTransformRigidityPenaltyTerm.h"

#include "itkChangeInformationImageFilter.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
TransformRigidityPenalty< TElastix >
::BeforeRegistration( void )
{
  /** Read the fixed rigidity image if desired. */
  std::string fixedRigidityImageName = "";
  this->GetConfiguration()->ReadParameter( fixedRigidityImageName,
    "FixedRigidityImageName", this->GetComponentLabel(), 0, -1, false );

  typedef typename Superclass1::RigidityImageType   RigidityImageType;
  typedef itk::ImageFileReader< RigidityImageType > RigidityImageReaderType;
  typename RigidityImageReaderType::Pointer fixedRigidityReader;
  typedef itk::ChangeInformationImageFilter< RigidityImageType > ChangeInfoFilterType;
  typedef typename ChangeInfoFilterType::Pointer                 ChangeInfoFilterPointer;
  typedef typename RigidityImageType::DirectionType              DirectionType;

  if( fixedRigidityImageName != "" )
  {
    /** Use the FixedRigidityImage. */
    this->SetUseFixedRigidityImage( true );

    /** Create the reader and set the filename. */
    fixedRigidityReader = RigidityImageReaderType::New();
    fixedRigidityReader->SetFileName( fixedRigidityImageName.c_str() );

    /** Possibly overrule the direction cosines. */
    ChangeInfoFilterPointer infoChanger = ChangeInfoFilterType::New();
    DirectionType           direction;
    direction.SetIdentity();
    infoChanger->SetOutputDirection( direction );
    infoChanger->SetChangeDirection( !this->GetElastix()->GetUseDirectionCosines() );
    infoChanger->SetInput( fixedRigidityReader->GetOutput() );

    /** Do the reading. */
    try
    {
      infoChanger->Update();
    }
    catch( itk::ExceptionObject & excp )
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
    this->SetFixedRigidityImage( infoChanger->GetOutput() );
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
  if( movingRigidityImageName != "" )
  {
    /** Use the movingRigidityImage. */
    this->SetUseMovingRigidityImage( true );

    /** Create the reader and set the filename. */
    movingRigidityReader = RigidityImageReaderType::New();
    movingRigidityReader->SetFileName( movingRigidityImageName.c_str() );

    /** Possibly overrule the direction cosines. */
    ChangeInfoFilterPointer infoChanger = ChangeInfoFilterType::New();
    DirectionType           direction;
    direction.SetIdentity();
    infoChanger->SetOutputDirection( direction );
    infoChanger->SetChangeDirection( !this->GetElastix()->GetUseDirectionCosines() );
    infoChanger->SetInput( movingRigidityReader->GetOutput() );

    /** Do the reading. */
    try
    {
      infoChanger->Update();
    }
    catch( itk::ExceptionObject & excp )
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
    this->SetMovingRigidityImage( infoChanger->GetOutput() );
  }
  else
  {
    this->SetUseMovingRigidityImage( false );
  }

  /** Important check: at least one rigidity image must be given. */
  if( fixedRigidityImageName == "" && movingRigidityImageName == "" )
  {
    xl::xout[ "warning" ] << "WARNING: FixedRigidityImageName and "
                          << "MovingRigidityImage are both not supplied.\n"
                          << "  The rigidity penalty term is evaluated on entire input "
                          << "transform domain." << std::endl;
  }

  /** Add target cells to xout["iteration"]. */
  xout[ "iteration" ].AddTargetCell( "Metric-LC" );
  xout[ "iteration" ].AddTargetCell( "Metric-OC" );
  xout[ "iteration" ].AddTargetCell( "Metric-PC" );
  xout[ "iteration" ].AddTargetCell( "||Gradient-LC||" );
  xout[ "iteration" ].AddTargetCell( "||Gradient-OC||" );
  xout[ "iteration" ].AddTargetCell( "||Gradient-PC||" );

  /** Format the metric as floats. */
  xl::xout[ "iteration" ][ "Metric-LC" ] << std::showpoint << std::fixed
                                         << std::setprecision( 10 );
  xl::xout[ "iteration" ][ "Metric-OC" ] << std::showpoint << std::fixed
                                         << std::setprecision( 10 );
  xl::xout[ "iteration" ][ "Metric-PC" ] << std::showpoint << std::fixed
                                         << std::setprecision( 10 );
  xl::xout[ "iteration" ][ "||Gradient-LC||" ] << std::showpoint << std::fixed
                                               << std::setprecision( 10 );
  xl::xout[ "iteration" ][ "||Gradient-OC||" ] << std::showpoint << std::fixed
                                               << std::setprecision( 10 );
  xl::xout[ "iteration" ][ "||Gradient-PC||" ] << std::showpoint << std::fixed
                                               << std::setprecision( 10 );

} // end BeforeRegistration()


/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
TransformRigidityPenalty< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of TransformRigidityPenalty metric took: "
         << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

  /** Check stuff. */
  this->CheckUseAndCalculationBooleans();

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
TransformRigidityPenalty< TElastix >
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

template< class TElastix >
void
TransformRigidityPenalty< TElastix >
::AfterEachIteration( void )
{
  /** Print some information. */
  xl::xout[ "iteration" ][ "Metric-LC" ]
    << this->GetLinearityConditionValue();
  xl::xout[ "iteration" ][ "Metric-OC" ]
    << this->GetOrthonormalityConditionValue();
  xl::xout[ "iteration" ][ "Metric-PC" ]
    << this->GetPropernessConditionValue();

  xl::xout[ "iteration" ][ "||Gradient-LC||" ]
    << this->GetLinearityConditionGradientMagnitude();
  xl::xout[ "iteration" ][ "||Gradient-OC||" ]
    << this->GetOrthonormalityConditionGradientMagnitude();
  xl::xout[ "iteration" ][ "||Gradient-PC||" ]
    << this->GetPropernessConditionGradientMagnitude();

} // end AfterEachIteration()


} // end namespace elastix

#endif // end #ifndef __elxTransformRigidityPenaltyTerm_HXX__
