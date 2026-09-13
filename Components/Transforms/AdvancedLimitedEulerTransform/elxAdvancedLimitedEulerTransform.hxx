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

#ifndef __elxAdvancedLimitedEulerTransform_HXX_
#define __elxAdvancedLimitedEulerTransform_HXX_

#include "elxAdvancedLimitedEulerTransform.h"
#include "itkImageGridSampler.h"
#include "itkContinuousIndex.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
AdvancedLimitedEulerTransformElastix< TElastix >
::AdvancedLimitedEulerTransformElastix()
{
  this->m_LimitedEulerTransform = LimitedEulerTransformType::New();
  this->SetCurrentTransform( this->m_LimitedEulerTransform );

} // end Constructor


/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
AdvancedLimitedEulerTransformElastix< TElastix >
::BeforeRegistration( void )
{
  /** Set center of rotation and initial translation. */
  this->InitializeTransform();

  /** Set the scales. */
  this->m_LimitedEulerTransform->SetScalesEstimation(true);
  this->SetScales();
  this->m_LimitedEulerTransform->SetScalesEstimation(false);

} // end BeforeRegistration()


/**
 * ************************* ReadFromFile ************************
 */

template< class TElastix >
void
AdvancedLimitedEulerTransformElastix< TElastix >
::ReadFromFile( void )
{
  /** Variables. */
  InputPointType centerOfRotationPoint;
  centerOfRotationPoint.Fill( 0.0 );

  /** Try first to read the CenterOfRotationPoint from the
   * transform parameter file, this is the new, and preferred
   * way, since elastix 3.402.
   */
  bool pointRead = this->ReadCenterOfRotationPoint( centerOfRotationPoint );

  if( !pointRead )
  {
    xl::xout[ "error" ] << "ERROR: No center of rotation is specified in "
                        << "the transform parameter file" << std::endl;
    itkExceptionMacro( << "Transform parameter file is corrupt." )
  }

  /** Set the center in this Transform. */
  this->m_LimitedEulerTransform->SetCenter( centerOfRotationPoint );

  /** Read the ComputeZYX. */
  if( SpaceDimension == 3 )
  {
    std::string computeZYX = "false";
    this->m_Configuration->ReadParameter( computeZYX, "ComputeZYX", 0 );
    if( computeZYX == "true" )
    {
      this->m_LimitedEulerTransform->SetComputeZYX( true );
    }
  }

  /** Read SharpnessOfLimits parameter */
  ScalarType sharpnessOfLimits = 50.0f;
  this->m_Configuration->ReadParameter( sharpnessOfLimits, "SharpnessOfLimits", 0 );
  this->m_LimitedEulerTransform->SetSharpnessOfLimits( sharpnessOfLimits );

  /** Read and set UpperLimits parameter */
  ParametersType upperLimits;
  upperLimits.SetSize(6);
  for( unsigned int i = 0; i < 6; i++ )
  {
    upperLimits[ i ] = i < 3 ? 3.14f : 200.0f;

    /** Returns zero when parameter was in the parameter file */
    bool found = this->m_Configuration->ReadParameter(
      upperLimits[ i ], "UpperLimits", i, false );
    if( !found )
    {
      for( unsigned int i = 0; i < 6; i++ )
      {
        upperLimits[ i ] = i < 3 ? 3.14f : 200.0f;
      }
      xl::xout[ "warning" ] << "WARNING: Upper limits should contain six values! "
                            << "Using default values instead!" << std::endl;
      break;
    }
  }
  this->m_LimitedEulerTransform->SetUpperLimits( upperLimits );

  /** Read and set LowerLimits parameter */
  ParametersType lowerLimits;
  lowerLimits.SetSize(6);
  for( unsigned int i = 0; i < 6; i++ )
  {
    lowerLimits[ i ] = i < 3 ? -3.14f : -200.0f;

    /** Returns zero when parameter was in the parameter file */
    bool found = this->m_Configuration->ReadParameter(
      lowerLimits[ i ], "LowerLimits", i, false );
    if( !found )
    {
      for( unsigned int i = 0; i < 6; i++ )
      {
        lowerLimits[ i ] = i < 3 ? -3.14f : -200.0f;
      }
      xl::xout[ "warning" ] << "WARNING: Lower limits should contain six values! "
                            << "Using default values instead!" << std::endl;
      break;
    }
  }
  this->m_LimitedEulerTransform->SetLowerLimits( lowerLimits );

  /** Call the ReadFromFile from the TransformBase.
   * BE AWARE: Only call Superclass2::ReadFromFile() after CenterOfRotation
   * is set, because it is used in the SetParameters()-function of this transform.
   */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* CreateDerivedTransformParametersMap ************************
 */
template <class TElastix>
auto
AdvancedLimitedEulerTransformElastix<TElastix>::CreateDerivedTransformParametersMap(void) const -> ParameterMapType
{
  ParameterMapType parameterMap{
    { "CenterOfRotationPoint", Conversion::ToVectorOfStrings(m_LimitedEulerTransform->GetCenter()) },
    { "SharpnessOfLimits", { Conversion::ToString(m_LimitedEulerTransform->GetSharpnessOfLimits()) } },
    { "UpperLimits", Conversion::ToVectorOfStrings(m_LimitedEulerTransform->GetUpperLimits()) },
    { "LowerLimits", Conversion::ToVectorOfStrings(m_LimitedEulerTransform->GetLowerLimits()) },
    { "UpperLimitsReached", Conversion::ToVectorOfStrings(m_LimitedEulerTransform->GetUpperLimitsReached()) },
    { "LowerLimitsReached", Conversion::ToVectorOfStrings(m_LimitedEulerTransform->GetLowerLimitsReached()) }
  };

  if (SpaceDimension == 3)
  {
    parameterMap["ComputeZYX"] = { Conversion::ToString(m_LimitedEulerTransform->GetComputeZYX()) };
  }
  return parameterMap;
} // end CreateDerivedTransformParametersMap()

/**
 * ************************* CreateDerivedTransformParametersMap ************************
 */
template <class TElastix>
auto
AdvancedLimitedEulerTransformElastix<TElastix>::AfterRegistration(void) -> void
{ 
} // end AfterRegistration()


/**
 * ************************* InitializeTransform *********************
 */

template< class TElastix >
void
AdvancedLimitedEulerTransformElastix< TElastix >
::InitializeTransform( void )
{
  /** Set all parameters to zero (no rotations, no translation). */
  this->m_LimitedEulerTransform->SetIdentity();

  /** Read and set UpperLimits parameter */
  ParametersType upperLimits;
  upperLimits.SetSize(6);
  for (unsigned int i = 0; i < 6; i++)
  {
    /** Returns zero when parameter was in the parameter file */
    bool found = this->m_Configuration->ReadParameter(
      upperLimits[i], "UpperLimits", i, false);
    if (!found)
    {
      for (unsigned int j = 0; j < 6; j++)
      {
        upperLimits[j] = j < 3 ? 3.14f : 200.0f;
      }
      xl::xout["warning"] << "WARNING: Upper limits should contain six values! "
        << "Using default values instead!" << std::endl;
      break;
    }
  }
  this->m_LimitedEulerTransform->SetUpperLimits(upperLimits);

  /** Read and set LowerLimits parameter */
  ParametersType lowerLimits;
  lowerLimits.SetSize(6);
  for (unsigned int i = 0; i < 6; i++)
  {
    /** Returns zero when parameter was in the parameter file */
    bool found = this->m_Configuration->ReadParameter(
      lowerLimits[i], "LowerLimits", i, false);
    if (!found)
    {
      for (unsigned int j = 0; j < 6; j++)
      {
        lowerLimits[j] = j < 3 ? -3.14f : -200.0f;
      }
      xl::xout["warning"] << "WARNING: Lower limits should contain six values! "
        << "Using default values instead!" << std::endl;
      break;
    }
  }
  this->m_LimitedEulerTransform->SetLowerLimits(lowerLimits);


  /** Verift that upper limits are larger than lower limits, otherwise throw error */
  for (unsigned int j = 0; j < 6; j++)
  {
    if (upperLimits[j] < lowerLimits[j])
    {
      itkExceptionMacro(<< "ERROR: Entry in the UpperLimits is smaller than "
        << " its corresponding entry in the LowerLimits.");
    }
  }
  
  /** Read SharpnessOfLimits parameter */
  ScalarType sharpnessOfLimits = 50.0f;
  this->m_Configuration->ReadParameter(sharpnessOfLimits, "SharpnessOfLimits", 0);
  this->m_LimitedEulerTransform->SetSharpnessOfLimits(sharpnessOfLimits);

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */
  IndexType      centerOfRotationIndex;
  InputPointType centerOfRotationPoint;
  bool           centerGivenAsIndex = true;
  bool           centerGivenAsPoint = true;
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    /** Initialize. */
    centerOfRotationIndex[ i ] = 0;
    centerOfRotationPoint[ i ] = 0.0;

    /** Check COR index: Returns zero when parameter was in the parameter file. */
    bool foundI = this->m_Configuration->ReadParameter(
      centerOfRotationIndex[ i ], "CenterOfRotation", i, false );
    if( !foundI )
    {
      centerGivenAsIndex &= false;
    }

    /** Check COR point: Returns zero when parameter was in the parameter file. */
    bool foundP = this->m_Configuration->ReadParameter(
      centerOfRotationPoint[ i ], "CenterOfRotationPoint", i, false );
    if( !foundP )
    {
      centerGivenAsPoint &= false;
    }

  } // end loop over SpaceDimension

  /** Check if CenterOfRotation has index-values within image. */
  bool CORIndexInImage = true;
  bool CORPointInImage = true;
  if( centerGivenAsIndex )
  {
    CORIndexInImage =  this->m_Registration->GetAsITKBaseType()
      ->GetFixedImage()->GetLargestPossibleRegion().IsInside(
      centerOfRotationIndex );
  }

  if( centerGivenAsPoint )
  {
    typedef itk::ContinuousIndex< double, SpaceDimension > ContinuousIndexType;
    ContinuousIndexType cindex;
    CORPointInImage = this->m_Registration->GetAsITKBaseType()
      ->GetFixedImage()->TransformPhysicalPointToContinuousIndex(
      centerOfRotationPoint, cindex );
  }

  /** Give a warning if necessary. */
  if( !CORIndexInImage && centerGivenAsIndex )
  {
    xl::xout[ "warning" ] << "WARNING: Center of Rotation (index) is not "
                          << "within image boundaries!" << std::endl;
  }

  /** Give a warning if necessary. */
  if( !CORPointInImage && centerGivenAsPoint && !centerGivenAsIndex )
  {
    xl::xout[ "warning" ] << "WARNING: Center of Rotation (point) is not "
                          << "within image boundaries!" << std::endl;
  }

  /** Check if user wants automatic transform initialization; false by default.
   * If an initial transform is given, automatic transform initialization is
   * not possible.
   */
  bool automaticTransformInitialization = false;
  bool tmpBool                          = false;
  this->m_Configuration->ReadParameter( tmpBool,
    "AutomaticTransformInitialization", 0 );
  if( tmpBool && this->Superclass1::GetInitialTransform() == 0 )
  {
    automaticTransformInitialization = true;
  }

  /**
   * Run the itkTransformInitializer if:
   * - No center of rotation was given, or
   * - The user asked for AutomaticTransformInitialization
   */
  bool centerGiven = centerGivenAsIndex || centerGivenAsPoint;
  if( !centerGiven || automaticTransformInitialization )
  {

    /** Use the TransformInitializer to determine a center of
     * of rotation and an initial translation.
     */
    TransformInitializerPointer transformInitializer
      = TransformInitializerType::New();
    transformInitializer->SetFixedImage(
      this->m_Registration->GetAsITKBaseType()->GetFixedImage() );
    transformInitializer->SetMovingImage(
      this->m_Registration->GetAsITKBaseType()->GetMovingImage() );
    transformInitializer->SetTransform( this->m_LimitedEulerTransform );

    /** Select the method of initialization. Default: "GeometricalCenter". */
    transformInitializer->GeometryOn();
    std::string method = "GeometricalCenter";
    this->m_Configuration->ReadParameter( method,
      "AutomaticTransformInitializationMethod", 0 );
    if( method == "CenterOfGravity" )
    {
      transformInitializer->MomentsOn();
    }

    transformInitializer->InitializeTransform();
  }

  /** Set the translation to zero, if no AutomaticTransformInitialization
   * was desired.
   */
  if( !automaticTransformInitialization )
  {
    OutputVectorType noTranslation;
    noTranslation.Fill( 0.0 );
    this->m_LimitedEulerTransform->SetTranslation( noTranslation );
  }

  /** Set the center of rotation if it was entered by the user. */
  if( centerGiven )
  {
    if( centerGivenAsIndex )
    {
      /** Convert from index-value to physical-point-value. */
      this->m_Registration->GetAsITKBaseType()->GetFixedImage()
      ->TransformIndexToPhysicalPoint(
        centerOfRotationIndex, centerOfRotationPoint );
    }
    this->m_LimitedEulerTransform->SetCenter( centerOfRotationPoint );
  }

  /** Apply the initial transform to the center of rotation, if
   * composition is used to combine the initial transform with the
   * the current (euler) transform.
   */
  if( this->GetUseComposition()
    && this->Superclass1::GetInitialTransform() != 0 )
  {
    InputPointType transformedCenterOfRotationPoint
      = this->Superclass1::GetInitialTransform()->TransformPoint(
      this->m_LimitedEulerTransform->GetCenter() );
    this->m_LimitedEulerTransform->SetCenter( transformedCenterOfRotationPoint );
  }

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()
  ->SetInitialTransformParameters( this->GetParameters() );

  /** Give feedback. */
  // \todo: should perhaps also print fixed parameters
  elxout << "Transform parameters are initialized as: "
         << this->GetParameters() << std::endl;

} // end InitializeTransform()


/**
 * ************************* SetScales *********************
 */

template< class TElastix >
void
AdvancedLimitedEulerTransformElastix< TElastix >
::SetScales( void )
{
  /** Create the new scales. */
  const NumberOfParametersType N = this->GetNumberOfParameters();
  ScalesType                   newscales( N );
  newscales.Fill( 1.0 );

  /** Check if automatic scales estimation is desired. */
  bool automaticScalesEstimation = false;
  this->m_Configuration->ReadParameter( automaticScalesEstimation,
    "AutomaticScalesEstimation", 0 );

  if( automaticScalesEstimation )
  {
    elxout << "Scales are estimated automatically." << std::endl;
    this->AutomaticScalesEstimation( newscales );
  }
  else
  {

    /** Here is an heuristic rule for estimating good values for
     * the rotation/translation scales.
     *
     * 1) Estimate the bounding box of your points (in physical units).
     * 2) Take the 3D Diagonal of that bounding box
     * 3) Multiply that by 10.0.
     * 4) use 1.0 /[ value from (3) ] as the translation scaling value.
     * 5) use 1.0 as the rotation scaling value.
     *
     * With this operation you bring the translation units
     * to the range of rotations (e.g. around -1 to 1).
     * After that, all your registration parameters are
     * in the relaxed range of -1:1. At that point you
     * can start setting your optimizer with step lengths
     * in the ranges of 0.001 if you are conservative, or
     * in the range of 0.1 if you want to live dangerously.
     * (0.1 radians is about 5.7 degrees).
     *
     * This heuristic rule is based on the naive assumption
     * that your registration may require translations as
     * large as 1/10 of the diagonal of the bounding box.
     */

    const double defaultScalingvalue = 100000.0;

    /** If the Dimension is 3, the first 3 parameters represent rotations.
     * If the Dimension is 2, only the first parameter represent a rotation.
     */
    unsigned int RotationPart = 3;
    if( SpaceDimension == 2 ) { RotationPart = 1; }

    /** this->m_Configuration->ReadParameter() returns 0 if there is a value given
     * in the parameter-file, and returns 1 if there is no value given in the
     * parameter-file.
     * Check which option is used:
     * - Nothing given in the parameter-file: rotations are scaled by the default
     *   value 100000.0
     * - Only one scale given in the parameter-file: rotations are scaled by this
     *   value.
     * - All scales are given in the parameter-file: each parameter is assigned its
     *   own scale.
     */

    std::size_t count
      = this->m_Configuration->CountNumberOfParameterEntries( "Scales" );

    /** Check which of the above options is used. */
    if( count == 0 )
    {
      /** In this case the first option is used. */
      for( unsigned int i = 0; i < RotationPart; i++ )
      {
        newscales[ i ] = defaultScalingvalue;
      }
    }
    else if( count == 1 )
    {
      /** In this case the second option is used. */
      double scale = defaultScalingvalue;
      this->m_Configuration->ReadParameter( scale, "Scales", 0 );
      for( unsigned int i = 0; i < RotationPart; i++ )
      {
        newscales[ i ] = scale;
      }
    }
    else if( count == this->GetNumberOfParameters() )
    {
      /** In this case the third option is used. */
      for( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
      {
        this->m_Configuration->ReadParameter( newscales[ i ], "Scales", i );
      }
    }
    else
    {
      /** In this case an error is made in the parameter-file.
       * An error is thrown, because using erroneous scales in the optimizer
       * can give unpredictable results.
       */
      itkExceptionMacro( << "ERROR: The Scales-option in the parameter-file"
                         << " has not been set properly." );
    }

  } // end else: no automaticScalesEstimation

  elxout << "Scales for transform parameters are: " << newscales << std::endl;

  /** Set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales( newscales );

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template< class TElastix >
bool
AdvancedLimitedEulerTransformElastix< TElastix >
::ReadCenterOfRotationPoint( InputPointType & rotationPoint ) const
{
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  InputPointType centerOfRotationPoint;
  bool           centerGivenAsPoint = true;
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    centerOfRotationPoint[ i ] = 0;

    /** Returns zero when parameter was in the parameter file */
    bool found = this->m_Configuration->ReadParameter(
      centerOfRotationPoint[ i ], "CenterOfRotationPoint", i, false );
    if( !found )
    {
      centerGivenAsPoint &= false;
    }
  }

  if( !centerGivenAsPoint )
  {
    return false;
  }

  /** Copy the temporary variable into the output of this function,
   * if everything went ok.
   */
  rotationPoint = centerOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()

} // end namespace elastix

#endif // end #ifndef __elxAdvancedLimitedEulerTransform_HXX_
