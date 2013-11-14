/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxAdvancedAffineTransform_HXX_
#define __elxAdvancedAffineTransform_HXX_

#include "elxAdvancedAffineTransform.h"
#include "itkImageGridSampler.h"
#include "itkContinuousIndex.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
AdvancedAffineTransformElastix<TElastix>
::AdvancedAffineTransformElastix()
{
  this->m_AffineTransform = AffineTransformType::New();
  this->SetCurrentTransform( this->m_AffineTransform );

} // end Constructor


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
AdvancedAffineTransformElastix<TElastix>
::BeforeRegistration( void )
{
  /** Task 1 - Set initial parameters. */
  this->InitializeTransform();

  /** Task 2 - Set the scales. */
  this->SetScales();

} // end BeforeRegistration()


/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
AdvancedAffineTransformElastix<TElastix>
::ReadFromFile( void )
{
  InputPointType centerOfRotationPoint;
  centerOfRotationPoint.Fill( 0.0 );
  bool pointRead = false;
  bool indexRead = false;

  /** Try first to read the CenterOfRotationPoint from the
   * transform parameter file, this is the new, and preferred
   * way, since elastix 3.402.
   */
  pointRead = this->ReadCenterOfRotationPoint( centerOfRotationPoint );

  /** If this did not succeed, probably a transform parameter file
   * is trying to be read that was generated using an older elastix
   * version. Try to read it as an index, and convert to point.
   */
  if ( !pointRead )
  {
    indexRead = this->ReadCenterOfRotationIndex( centerOfRotationPoint );
  }

  if ( !pointRead && !indexRead )
  {
    xl::xout["error"] << "ERROR: No center of rotation is specified in the "
      << "transform parameter file" << std::endl;
    itkExceptionMacro( << "Transform parameter file is corrupt.")
  }

  /** Set the center in this Transform. */
  this->m_AffineTransform->SetCenter( centerOfRotationPoint );

  /** Call the ReadFromFile from the TransformBase.
   * BE AWARE: Only call Superclass2::ReadFromFile() after CenterOfRotation
   * is set, because it is used in the SetParameters()-function of this transform.
   */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* WriteToFile ************************
 */

template <class TElastix>
void
AdvancedAffineTransformElastix<TElastix>
::WriteToFile( const ParametersType & param ) const
{
  /** Call the WriteToFile from the TransformBase. */
  this->Superclass2::WriteToFile( param );

  /** Write AdvancedAffineTransform specific things. */
  xout["transpar"] << std::endl << "// AdvancedAffineTransform specific" << std::endl;

  /** Set the precision of cout to 10. */
  xout["transpar"] << std::setprecision( 10 );

  /** Get the center of rotation point and write it to file. */
  InputPointType rotationPoint = this->m_AffineTransform->GetCenter();
  xout["transpar"] << "(CenterOfRotationPoint ";
  for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
  {
    xout["transpar"] << rotationPoint[ i ] << " ";
  }
  xout["transpar"] << rotationPoint[ SpaceDimension - 1 ] << ")" << std::endl;

  /** Set the precision back to default value. */
  xout["transpar"] << std::setprecision( this->m_Elastix->GetDefaultOutputPrecision() );

} // end WriteToFile()


/**
 * ************************* CreateTransformParametersMap ************************
 */

template <class TElastix>
void
AdvancedAffineTransformElastix<TElastix>
::CreateTransformParametersMap(
  const ParametersType & param,
  ParameterMapType * paramsMap ) const
{
  std::string parameterName;
  std::vector< std::string > parameterValues;
  char tmpValue[ 256 ];

  /** Call the CreateTransformParametersMap from the TransformBase. */
  this->Superclass2::CreateTransformParametersMap( param, paramsMap );

  /** Get the center of rotation point and write it to file. */
  InputPointType rotationPoint = this->m_AffineTransform->GetCenter();
  parameterName = "CenterOfRotationPoint";
  for ( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    sprintf( tmpValue, "%.10lf", rotationPoint[ i ] );
    parameterValues.push_back( tmpValue );
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );

} // end CreateTransformParametersMap()


/**
 * ************************* InitializeTransform *********************
 */

template <class TElastix>
void
AdvancedAffineTransformElastix<TElastix>
::InitializeTransform( void )
{
  /** Set all parameters to zero (no rotations, no translation). */
  this->m_AffineTransform->SetIdentity();

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */
  IndexType centerOfRotationIndex;
  InputPointType centerOfRotationPoint;

  bool centerGivenAsIndex = true;
  bool centerGivenAsPoint = true;
  for ( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    /** Initialize. */
    centerOfRotationIndex[ i ] = 0;
    centerOfRotationPoint[ i ] = 0.0;

    /** Check COR index: Returns zero when parameter was in the parameter file. */
    bool foundI = this->m_Configuration->ReadParameter(
      centerOfRotationIndex[ i ], "CenterOfRotation", i, false );
    if ( !foundI )
    {
      centerGivenAsIndex &= false;
    }

    /** Check COR point: Returns zero when parameter was in the parameter file. */
    bool foundP = this->m_Configuration->ReadParameter(
      centerOfRotationPoint[ i ], "CenterOfRotationPoint", i, false );
    if ( !foundP )
    {
      centerGivenAsPoint &= false;
    }
  } // end loop over SpaceDimension

  /** Check if CenterOfRotation has index-values within image. */
  bool CORIndexInImage = true;
  bool CORPointInImage = true;
  if ( centerGivenAsIndex )
  {
    CORIndexInImage =  this->m_Registration->GetAsITKBaseType()
      ->GetFixedImage()->GetLargestPossibleRegion().IsInside(
      centerOfRotationIndex );
  }

  if ( centerGivenAsPoint )
  {
    typedef itk::ContinuousIndex< double, SpaceDimension > ContinuousIndexType;
    ContinuousIndexType cindex;
    CORPointInImage = this->m_Registration->GetAsITKBaseType()
      ->GetFixedImage()->TransformPhysicalPointToContinuousIndex(
      centerOfRotationPoint, cindex );
  }

  /** Give a warning if necessary. */
  if ( !CORIndexInImage && centerGivenAsIndex )
  {
    xl::xout["warning"] << "WARNING: Center of Rotation (index) is not "
      << "within image boundaries!" << std::endl;
  }

  /** Give a warning if necessary. */
  if ( !CORPointInImage && centerGivenAsPoint && !centerGivenAsIndex )
  {
    xl::xout["warning"] << "WARNING: Center of Rotation (point) is not "
      << "within image boundaries!" << std::endl;
  }

  /** Check if user wants automatic transform initialization; false by default.
   * If an initial transform is given, automatic transform initialization is
   * not possible.
   */
  bool automaticTransformInitialization = false;
  bool tmpBool = false;
  this->m_Configuration->ReadParameter( tmpBool,
    "AutomaticTransformInitialization", 0 );
  if ( tmpBool && this->Superclass1::GetInitialTransform() == 0 )
  {
    automaticTransformInitialization = true;
  }

  /** Run the itkTransformInitializer if:
   * - No center of rotation was given, or
   * - The user asked for AutomaticTransformInitialization
   */
  bool centerGiven = centerGivenAsIndex || centerGivenAsPoint;
  if ( !centerGiven || automaticTransformInitialization )
  {
    /** Use the TransformInitializer to determine a center of
     * of rotation and an initial translation.
     */
    TransformInitializerPointer transformInitializer =
      TransformInitializerType::New();
    transformInitializer->SetFixedImage(
      this->m_Registration->GetAsITKBaseType()->GetFixedImage() );
    transformInitializer->SetMovingImage(
      this->m_Registration->GetAsITKBaseType()->GetMovingImage() );
    transformInitializer->SetFixedImageMask(
      this->m_Elastix->GetFixedMask() );
    // Note that setting the mask like this:
    //  this->m_Registration->GetAsITKBaseType()->GetMetric()->GetFixedImageMask() );
    // does not work since it is not yet initialized at this point in the metric.
    transformInitializer->SetMovingImageMask(
      this->m_Elastix->GetMovingMask() );
    transformInitializer->SetTransform( this->m_AffineTransform );

    /** Select the method of initialization. Default: "GeometricalCenter". */
    transformInitializer->GeometryOn();
    std::string method = "GeometricalCenter";
    this->m_Configuration->ReadParameter( method,
      "AutomaticTransformInitializationMethod", 0 );
    if ( method == "CenterOfGravity" )
    {
      transformInitializer->MomentsOn();
    }
    else if ( method == "Origins" )
    {
      transformInitializer->OriginsOn();
    }
    transformInitializer->InitializeTransform();
  }

  /** Set the translation to zero, if no AutomaticTransformInitialization
   * was desired.
   */
  if ( !automaticTransformInitialization )
  {
    OutputVectorType noTranslation;
    noTranslation.Fill(0.0);
    this->m_AffineTransform->SetTranslation( noTranslation );
  }

  /** Set the center of rotation if it was entered by the user. */
  if ( centerGiven )
  {
    if ( centerGivenAsIndex )
    {
      /** Convert from index-value to physical-point-value. */
      this->m_Registration->GetAsITKBaseType()->GetFixedImage()
        ->TransformIndexToPhysicalPoint(
        centerOfRotationIndex, centerOfRotationPoint );
    }
    this->m_AffineTransform->SetCenter( centerOfRotationPoint );
  }

  /** Apply the initial transform to the center of rotation, if
   * composition is used to combine the initial transform with the
   * the current (affine) transform.
   */
  if ( this->GetUseComposition()
    && this->Superclass1::GetInitialTransform() != 0 )
  {
    InputPointType transformedCenterOfRotationPoint
      = this->Superclass1::GetInitialTransform()->TransformPoint(
      this->m_AffineTransform->GetCenter() );
    this->m_AffineTransform->SetCenter(
      transformedCenterOfRotationPoint );
  }

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()->
    SetInitialTransformParameters( this->GetParameters() );

  /** Give feedback. */
  // \todo: should perhaps also print fixed parameters
  elxout << "Transform parameters are initialized as: "
    << this->GetParameters() << std::endl;

} // end InitializeTransform()


/**
 * ************************* SetScales *********************
 */

template <class TElastix>
void
AdvancedAffineTransformElastix<TElastix>
::SetScales( void )
{
  /** Create the new scales. */
  const NumberOfParametersType N = this->GetNumberOfParameters();
  ScalesType newscales( N );
  newscales.Fill( 1.0 );

  /** Check if automatic scales estimation is desired. */
  bool automaticScalesEstimation = false;
  this->m_Configuration->ReadParameter( automaticScalesEstimation,
    "AutomaticScalesEstimation", 0 );

  if ( automaticScalesEstimation )
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

    /** The first SpaceDimension * SpaceDimension number of parameters
     * represent rotations (4 in 2D and 9 in 3D).
     */
    const unsigned int rotationPart = SpaceDimension * SpaceDimension;

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
    const double defaultScalingvalue = 100000.0;

    std::size_t count
      = this->m_Configuration->CountNumberOfParameterEntries( "Scales" );

    /** Check which of the above options is used. */
    if ( count == 0 )
    {
      /** In this case the first option is used. */
      for ( unsigned int i = 0; i < rotationPart; i++ )
      {
        newscales[ i ] = defaultScalingvalue;
      }
    }
    else if ( count == 1 )
    {
      /** In this case the second option is used. */
      double scale = defaultScalingvalue;
      this->m_Configuration->ReadParameter( scale, "Scales", 0 );
      for ( unsigned int i = 0; i < rotationPart; i++ )
      {
        newscales[ i ] = scale;
      }
    }
    else if ( count == this->GetNumberOfParameters() )
    {
      /** In this case the third option is used. */
      for ( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
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

  /** And set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newscales );

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationIndex *********************
 */

template <class TElastix>
bool
AdvancedAffineTransformElastix<TElastix>
::ReadCenterOfRotationIndex( InputPointType & rotationPoint ) const
{
  /** Try to read CenterOfRotationIndex from the transform parameter
   * file, which is the rotationPoint, expressed in index-values.
   */
  IndexType centerOfRotationIndex;
  bool centerGivenAsIndex = true;
  for ( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    centerOfRotationIndex[ i ] = 0;

    /** Returns zero when parameter was in the parameter file. */
    bool found = this->m_Configuration->ReadParameter(
      centerOfRotationIndex[ i ], "CenterOfRotation", i, false );
    if ( !found )
    {
      centerGivenAsIndex &= false;
    }
  }

  if ( !centerGivenAsIndex )
  {
    return false;
  }

  /** Get spacing, origin and size of the fixed image.
   * We put this in a dummy image, so that we can correctly
   * calculate the center of rotation in world coordinates.
   */
  SpacingType   spacing;
  IndexType     index;
  PointType     origin;
  SizeType      size;
  DirectionType direction;
  direction.SetIdentity();
  for ( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    /** Read size from the parameter file. Zero by default, which is illegal. */
    size[ i ] = 0;
    this->m_Configuration->ReadParameter( size[ i ], "Size", i );

    /** Default index. Read index from the parameter file. */
    index[ i ] = 0;
    this->m_Configuration->ReadParameter( index[ i ], "Index", i );

    /** Default spacing. Read spacing from the parameter file. */
    spacing[ i ] = 1.0;
    this->m_Configuration->ReadParameter( spacing[ i ], "Spacing", i );

    /** Default origin. Read origin from the parameter file. */
    origin[ i ] = 0.0;
    this->m_Configuration->ReadParameter( origin[ i ], "Origin", i );

    /** Read direction cosines. Default identity */
    for ( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      this->m_Configuration->ReadParameter( direction( j, i ),
        "Direction", i * SpaceDimension + j );
    }
  }

  /** Check for image size. */
  bool illegalSize = false;
  for ( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    if ( size[ i ] == 0 )
    {
      illegalSize = true;
    }
  }

  if ( illegalSize )
  {
    xl::xout["error"] << "ERROR: One or more image sizes are 0!" << std::endl;
    return false;
  }

  /** Make a temporary image with the right region info,
   * so that the TransformIndexToPhysicalPoint-functions will be right.
   */
  typedef FixedImageType DummyImageType;
  typename DummyImageType::Pointer dummyImage = DummyImageType::New();
  RegionType region;
  region.SetIndex( index );
  region.SetSize( size );
  dummyImage->SetRegions( region );
  dummyImage->SetOrigin( origin );
  dummyImage->SetSpacing( spacing );
  dummyImage->SetDirection( direction );

  /** Convert center of rotation from index-value to physical-point-value. */
  dummyImage->TransformIndexToPhysicalPoint(
    centerOfRotationIndex, rotationPoint );

  /** Successfully read centerOfRotation as Index. */
  return true;

} // end ReadCenterOfRotationIndex()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template <class TElastix>
bool
AdvancedAffineTransformElastix<TElastix>
::ReadCenterOfRotationPoint( InputPointType & rotationPoint ) const
{
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  InputPointType centerOfRotationPoint;
  bool centerGivenAsPoint = true;
  for ( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    centerOfRotationPoint[ i ] = 0.0;

    /** Returns zero when parameter was in the parameter file. */
    bool found = this->m_Configuration->ReadParameter(
      centerOfRotationPoint[ i ], "CenterOfRotationPoint", i, false );
    if ( !found )
    {
      centerGivenAsPoint &= false;
    }
  }

  if ( !centerGivenAsPoint )
  {
    return false;
  }

  /** copy the temporary variable into the output of this function,
   * if everything went ok.
   */
  rotationPoint = centerOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()


} // end namespace elastix


#endif // end #ifndef __elxAdvancedAffineTransform_HXX_

