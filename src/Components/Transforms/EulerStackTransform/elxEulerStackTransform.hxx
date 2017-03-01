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
#ifndef __elxEulerStackTransform_hxx
#define __elxEulerStackTransform_hxx

#include "elxEulerStackTransform.h"

namespace elastix
{

/**
* ********************* Constructor ****************************
*/
template< class TElastix >
EulerStackTransform< TElastix >
::EulerStackTransform()
{} // end Constructor


/**
* ********************* InitializeAffineTransform ****************************
*/
template< class TElastix >
unsigned int
EulerStackTransform< TElastix >
::InitializeEulerTransform()
{
  /** Initialize the m_AffineDummySubTransform */
  this->m_EulerDummySubTransform = ReducedDimensionEulerTransformType::New();

  /** Create stack transform. */
  this->m_EulerStackTransform = EulerStackTransformType::New();

  /** Set stack transform as current transform. */
  this->SetCurrentTransform( this->m_EulerStackTransform );

  return 0;
}


/**
 * ******************* BeforeAll ***********************
 */

template< class TElastix >
int
EulerStackTransform< TElastix >
::BeforeAll( void )
{
  /** Initialize affine transform. */
  return InitializeEulerTransform();
}


/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
EulerStackTransform< TElastix >
::BeforeRegistration( void )
{
  /** Task 1 - Set the stack transform parameters. */

  /** Determine stack transform settings. Here they are based on the fixed image. */
  const SizeType imageSize = this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  this->m_NumberOfSubTransforms = imageSize[ SpaceDimension - 1 ];
  this->m_StackSpacing          = this->GetElastix()->GetFixedImage()->GetSpacing()[ SpaceDimension - 1 ];
  this->m_StackOrigin           = this->GetElastix()->GetFixedImage()->GetOrigin()[ SpaceDimension - 1 ];

  /** Set stack transform parameters. */
  this->m_EulerStackTransform->SetNumberOfSubTransforms( this->m_NumberOfSubTransforms );
  this->m_EulerStackTransform->SetStackOrigin( this->m_StackOrigin );
  this->m_EulerStackTransform->SetStackSpacing( this->m_StackSpacing );

  /** Initialize stack sub transforms. */
  this->m_EulerStackTransform->SetAllSubTransforms( this->m_EulerDummySubTransform );

  /** Task 2 - Give the registration an initial parameter-array. */
  ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
  dummyInitialParameters.Fill( 0.0 );

  /** Put parameters in the registration. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters( dummyInitialParameters );

  /** Task 3 - Initialize the transform */
  this->InitializeTransform();

  /** Task 4 - Set the scales. */
  this->SetScales();

} // end BeforeRegistration()


/**
 * ************************* ReadFromFile ************************
 */

template< class TElastix >
void
EulerStackTransform< TElastix >
::ReadFromFile( void )
{

  /** Read stack-spacing, stack-origin and number of sub-transforms. */
  this->GetConfiguration()->ReadParameter( this->m_NumberOfSubTransforms,
    "NumberOfSubTransforms", this->GetComponentLabel(), 0, 0 );
  this->GetConfiguration()->ReadParameter( this->m_StackOrigin,
    "StackOrigin", this->GetComponentLabel(), 0, 0 );
  this->GetConfiguration()->ReadParameter( this->m_StackSpacing,
    "StackSpacing", this->GetComponentLabel(), 0, 0 );

  ReducedDimensionInputPointType RDcenterOfRotationPoint;
  RDcenterOfRotationPoint.Fill( 0.0 );
  bool pointRead = false;
  bool indexRead = false;

  /** Try first to read the CenterOfRotationPoint from the
   * transform parameter file, this is the new, and preferred
   * way, since elastix 3.402.
   */
  pointRead = this->ReadCenterOfRotationPoint( RDcenterOfRotationPoint );

  /** If this did not succeed, probably a transform parameter file
   * is trying to be read that was generated using an older elastix
   * version. Try to read it as an index, and convert to point.
   */
  if( !pointRead )
  {
    indexRead = this->ReadCenterOfRotationIndex( RDcenterOfRotationPoint );
  }

  if( !pointRead && !indexRead )
  {
    xl::xout[ "error" ] << "ERROR: No center of rotation is specified in the "
                        << "transform parameter file" << std::endl;
    itkExceptionMacro( << "Transform parameter file is corrupt." )
  }

  this->InitializeEulerTransform();

  this->m_EulerDummySubTransform->SetCenter( RDcenterOfRotationPoint );

  /** Set stack transform parameters. */
  this->m_EulerStackTransform->SetNumberOfSubTransforms( this->m_NumberOfSubTransforms );
  this->m_EulerStackTransform->SetStackOrigin( this->m_StackOrigin );
  this->m_EulerStackTransform->SetStackSpacing( this->m_StackSpacing );

  /** Set stack subtransforms. */
  this->m_EulerStackTransform->SetAllSubTransforms( this->m_EulerDummySubTransform );

  /** Call the ReadFromFile from the TransformBase. */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* WriteToFile ************************
 *
 * Saves the TransformParameters as a vector and if wanted
 * also as a deformation field.
 */

template< class TElastix >
void
EulerStackTransform< TElastix >
::WriteToFile( const ParametersType & param ) const
{

  /** Call the WriteToFile from the TransformBase. */
  this->Superclass2::WriteToFile( param );

  /** Add some AffineTransform specific lines. */
  xout[ "transpar" ] << std::endl << "// EulerStackTransform specific" << std::endl;

  /** Set the precision of cout to 10. */
  xout[ "transpar" ] << std::setprecision( 10 );

  /** Get the center of rotation point and write it to file. */
  ReducedDimensionInputPointType rotationPoint = this->m_EulerDummySubTransform->GetCenter();
  xout[ "transpar" ] << "(CenterOfRotationPoint ";
  for( unsigned int i = 0; i < ReducedSpaceDimension - 1; i++ )
  {
    xout[ "transpar" ] << rotationPoint[ i ] << " ";
  }
  xout[ "transpar" ] << rotationPoint[ ReducedSpaceDimension - 1 ] << ")" << std::endl;

  /** Write the stack spacing, stack origin and number of sub transforms. */
  xout[ "transpar" ] << "(StackSpacing " << this->m_EulerStackTransform->GetStackSpacing() << ")" << std::endl;
  xout[ "transpar" ] << "(StackOrigin " << this->m_EulerStackTransform->GetStackOrigin() << ")" << std::endl;
  xout[ "transpar" ] << "(NumberOfSubTransforms " << this->m_EulerStackTransform->GetNumberOfSubTransforms() << ")" << std::endl;

  /** Set the precision back to default value. */
  xout[ "transpar" ] << std::setprecision( this->m_Elastix->GetDefaultOutputPrecision() );

} // end WriteToFile()


/**
 * ********************* InitializeTransform ****************************
 */

template< class TElastix >
void
EulerStackTransform< TElastix >
::InitializeTransform()
{

  /** Set all parameters to zero (no rotations, no translation). */
  this->m_EulerDummySubTransform->SetIdentity();

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */

  ContinuousIndexType                 centerOfRotationIndex;
  InputPointType                      centerOfRotationPoint;
  ReducedDimensionContinuousIndexType redDimCenterOfRotationIndex;
  ReducedDimensionInputPointType      redDimCenterOfRotationPoint;

  bool     centerGivenAsIndex = true;
  bool     centerGivenAsPoint = true;
  SizeType fixedImageSize     = this->m_Registration->GetAsITKBaseType()->
    GetFixedImage()->GetLargestPossibleRegion().GetSize();

  /** Try to read center of rotation point (COP) from parameter file. */
  for( unsigned int i = 0; i < ReducedSpaceDimension; i++ )
  {
    /** Initialize. */
    centerOfRotationIndex[ i ]       = 0;
    centerOfRotationPoint[ i ]       = 0.0;
    redDimCenterOfRotationIndex[ i ] = 0;
    redDimCenterOfRotationPoint[ i ] = 0.0;

    /** Check COR index: Returns zero when parameter was in the parameter file. */
    const bool foundI = this->m_Configuration->ReadParameter(
      centerOfRotationIndex[ i ], "CenterOfRotation", i, false );
    if( !foundI )
    {
      centerGivenAsIndex &= false;
    }

    /** Check COR point: Returns zero when parameter was in the parameter file. */
    const bool foundP = this->m_Configuration->ReadParameter(
      redDimCenterOfRotationPoint[ i ], "CenterOfRotationPoint", i, false );
    if( !foundP )
    {
      centerGivenAsPoint &= false;
    }
  } // end loop over SpaceDimension

  /** Determine the center of rotation as the center of the image if no center was given */
  const bool centerGiven = centerGivenAsIndex || centerGivenAsPoint;
  if( !centerGiven  )
  {
    /** Use center of image as default center of rotation */
    for( unsigned int k = 0; k < SpaceDimension; k++ )
    {
      centerOfRotationIndex[ k ] = ( fixedImageSize[ k ] - 1.0f ) / 2.0f;
    }

    /** Convert from continuous index to physical point */
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()->
      TransformContinuousIndexToPhysicalPoint( centerOfRotationIndex, centerOfRotationPoint );

    for( unsigned int k = 0; k < ReducedSpaceDimension; k++ )
    {
      redDimCenterOfRotationPoint[ k ] = redDimCenterOfRotationPoint[ k ];
    }

    /** FIX: why may the cop not work when using direction cosines? */
    bool UseDirectionCosines = true;
    this->m_Configuration->ReadParameter( UseDirectionCosines, "UseDirectionCosines", 0 );
    if( !UseDirectionCosines )
    {
      elxout << "warning: a wrong center of rotation could have been set, "
             << " please check the transform matrix in the header file" << std::endl;
    }
  }

  /** Transform center of rotation point to physical point if given as index in parameter file. */
  if( centerGivenAsIndex )
  {
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()
      ->TransformContinuousIndexToPhysicalPoint( centerOfRotationIndex, centerOfRotationPoint );

    for( unsigned int k = 0; k < ReducedSpaceDimension; k++ )
    {
      redDimCenterOfRotationPoint[ k ] = centerOfRotationPoint[ k ];
    }
  }

  /** Transform center of rotation point using initial transform if present. */
  InitialTransformCenter( redDimCenterOfRotationPoint );

  /** Set the center of rotation point. */
  this->m_EulerDummySubTransform->SetCenter( redDimCenterOfRotationPoint );

  /** Set the translation to zero */
  ReducedDimensionOutputVectorType noTranslation;
  noTranslation.Fill( 0.0 );
  this->m_EulerDummySubTransform->SetTranslation( noTranslation );

  /** Set all subtransforms to a copy of the dummy Translation sub transform. */
  this->m_EulerStackTransform->SetAllSubTransforms( this->m_EulerDummySubTransform );

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()->
    SetInitialTransformParameters( this->GetParameters() );

} // end InitializeTransform()


template< class TElastix >
void
EulerStackTransform< TElastix >
::InitialTransformCenter( ReducedDimensionInputPointType & point )
{
  /** Apply the initial transform to the center of rotation, if
   * composition is used to combine the initial transform with the
   * the current (euler) transform.
   */
  if( this->GetUseComposition()
    && this->Superclass1::GetInitialTransform() != 0 )
  {
    /** Transform point to voxel coordinates. */
    InputPointType      fullDimensionCenterPoint;
    ContinuousIndexType fullDimensionCenterIndex;
    for( unsigned int i = 0; i < ReducedSpaceDimension; ++i )
    {
      fullDimensionCenterPoint[ i ] = point[ i ];
    }
    fullDimensionCenterPoint[ SpaceDimension - 1 ] = 0;
    this->m_Registration->GetAsITKBaseType()->
      GetFixedImage()->TransformPhysicalPointToContinuousIndex( fullDimensionCenterPoint, fullDimensionCenterIndex );

    /** Get size of image and number of time points. */
    const SizeType fixedImageSize = this->m_Registration->GetAsITKBaseType()->
      GetFixedImage()->GetLargestPossibleRegion().GetSize();
    const unsigned int numTimePoints = fixedImageSize[ SpaceDimension - 1 ];

    /** Transform center of rotation point for each time point and
     * compute average. */
    ReducedDimensionInputPointType averagePoint;
    averagePoint.Fill( 0.0 );
    for( unsigned int t = 0; t < numTimePoints; ++t )
    {
      /** Set time point and transform back to point. */
      fullDimensionCenterIndex[ SpaceDimension - 1 ] = t;
      this->m_Registration->GetAsITKBaseType()->
        GetFixedImage()->TransformContinuousIndexToPhysicalPoint( fullDimensionCenterIndex, fullDimensionCenterPoint );

      /** Transform point using initial transform. */
      InputPointType transformedCenterOfRotationPoint
        = this->Superclass1::GetInitialTransform()->TransformPoint(
        fullDimensionCenterPoint );

      /** Add to averagePoint. */
      for( unsigned int d = 0; d < ReducedSpaceDimension; ++d )
      {
        averagePoint[ d ] += transformedCenterOfRotationPoint[ d ];
      }
    }
    for( unsigned int d = 0; d < ReducedSpaceDimension; ++d )
    {
      averagePoint[ d ] /= numTimePoints;
    }

    point = averagePoint;
  }
}


/**
 * ************************* SetScales *********************
 */

template< class TElastix >
void
EulerStackTransform< TElastix >
::SetScales( void )
{

  /** Create the new scales. */
  const NumberOfParametersType N = this->GetNumberOfParameters();
  ScalesType                   newscales( N );

  /** Check if automatic scales estimation is desired. */
  bool automaticScalesEstimationStackTransform = false;
  this->m_Configuration->ReadParameter( automaticScalesEstimationStackTransform,
    "AutomaticScalesEstimationStackTransform", 0 );

  if( automaticScalesEstimationStackTransform )
  {
    this->AutomaticScalesEstimationStackTransform( this->m_EulerStackTransform->GetNumberOfSubTransforms(), newscales );
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

    /** In 2D, the first parameter is an angle, the other two translations;
     * in 3D, the first three parameters are angles, the last three translations.
     */
    const unsigned int numRotationParsPerDimension = ReducedSpaceDimension == 2 ? 1 : 3;
    const unsigned int numTotalParsPerDimension    = ReducedSpaceDimension == 2 ? 3 : 6;

    /** this->m_Configuration->ReadParameter() returns 0 if there is a value given
     * in the parameter-file, and returns 1 if there is no value given in the
     * parameter-file.
     *
     * Check which option is used:
     * - Nothing given in the parameter-file: rotations are scaled by the default
     *   value 100000.0
     * - Only one scale given in the parameter-file: rotations are scaled by this
     *   value.
     * - All scales are given in the parameter-file: each parameter is assigned its
     *   own scale.
     */
    const double defaultScalingvalue = 10000.0;

    const int sizeLastDimension
      = this->GetElastix()->GetFixedImage()
        ->GetLargestPossibleRegion().GetSize()[ SpaceDimension - 1 ];

    std::size_t count
      = this->m_Configuration->CountNumberOfParameterEntries( "Scales" );

    /** Check which of the above options is used. */
    if( count == 0 )
    {
      /** In this case the first option is used. */
      newscales.Fill( defaultScalingvalue );

      /** The non-rotation scales are set to 1.0 for all dimensions */
      for( unsigned int i = numRotationParsPerDimension; i < ( numTotalParsPerDimension * sizeLastDimension ); i += numTotalParsPerDimension )
      {
        for( unsigned int j = numRotationParsPerDimension; j < numTotalParsPerDimension; ++j )
        {
          newscales[ i + j - numRotationParsPerDimension ] = 1.0;
        }
      }
    }
    else if( count == 1 )
    {
      /** In this case the second option is used. */
      double scale = defaultScalingvalue;
      this->m_Configuration->ReadParameter( scale, "Scales", 0 );
      newscales.Fill( scale );

      /** The non-rotation scales are set to 1.0 for all dimensions */
      for( unsigned int i = numRotationParsPerDimension; i < ( numTotalParsPerDimension * sizeLastDimension ); i += numTotalParsPerDimension )
      {
        for( unsigned int j = numRotationParsPerDimension; j < numTotalParsPerDimension; ++j )
        {
          newscales[ i + j - numRotationParsPerDimension ] = 1.0;
        }
      }
    }
    else if( count == this->GetNumberOfParameters() )
    {
      newscales.Fill( 1.0 );
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

  } // end else: no automaticScalesEstimationStackTransform

  elxout << "Scales for transform parameters are: " << newscales << std::endl;

  /** And set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newscales );

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationIndex *********************
 */

template< class TElastix >
bool
EulerStackTransform< TElastix >
::ReadCenterOfRotationIndex( ReducedDimensionInputPointType & rotationPoint ) const
{
  /** Try to read CenterOfRotationIndex from the transform parameter
   * file, which is the rotationPoint, expressed in index-values.
   */
  ReducedDimensionContinuousIndexType redDimCenterOfRotationIndex;
  bool                                centerGivenAsIndex = true;
  for( unsigned int i = 0; i < ReducedSpaceDimension; i++ )
  {
    redDimCenterOfRotationIndex[ i ] = 0;

    /** Returns zero when parameter was in the parameter file. */
    bool found = this->m_Configuration->ReadParameter(
      redDimCenterOfRotationIndex[ i ], "CenterOfRotation", i, false );
    if( !found )
    {
      centerGivenAsIndex &= false;
    }
  }

  if( !centerGivenAsIndex )
  {
    return false;
  }

  /** Get spacing, origin and size of the fixed image.
   * We put this in a dummy image, so that we can correctly
   * calculate the center of rotation in world coordinates.
   */
  ReducedDimensionSpacingType   spacing;
  ReducedDimensionIndexType     index;
  ReducedDimensionPointType     origin;
  ReducedDimensionSizeType      size;
  ReducedDimensionDirectionType direction;
  direction.SetIdentity();

  for( unsigned int i = 0; i < ReducedSpaceDimension; i++ )
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
    for( unsigned int j = 0; j < ReducedSpaceDimension; j++ )
    {
      this->m_Configuration->ReadParameter( direction( j, i ),
        "Direction", i * SpaceDimension + j );
    }
  }

  /** Check for image size. */
  bool illegalSize = false;
  for( unsigned int i = 0; i < ReducedSpaceDimension; i++ )
  {
    if( size[ i ] == 0 )
    {
      illegalSize = true;
    }
  }

  if( illegalSize )
  {
    xl::xout[ "error" ] << "ERROR: One or more image sizes are 0!" << std::endl;
    return false;
  }

  /** Make a temporary image with the right region info,
   * so that the TransformIndexToPhysicalPoint-functions will be right.
   */
  typedef  ReducedDimensionImageType DummyImageType;
  typename DummyImageType::Pointer dummyImage = DummyImageType::New();
  ReducedDimensionRegionType redDimRegion;

  redDimRegion.SetIndex( index );
  redDimRegion.SetSize( size );
  dummyImage->SetRegions( redDimRegion );
  dummyImage->SetOrigin( origin );
  dummyImage->SetSpacing( spacing );
  dummyImage->SetDirection( direction );

  /** Convert center of rotation from index-value to physical-point-value. */
  dummyImage->TransformContinuousIndexToPhysicalPoint(
    redDimCenterOfRotationIndex, rotationPoint );

  /** Successfully read centerOfRotation as Index. */
  return true;

} // end ReadCenterOfRotationIndex()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template< class TElastix >
bool
EulerStackTransform< TElastix >
::ReadCenterOfRotationPoint( ReducedDimensionInputPointType & rotationPoint ) const
{
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  ReducedDimensionInputPointType redDimCenterOfRotationPoint;
  bool                           centerGivenAsPoint = true;
  for( unsigned int i = 0; i < ReducedSpaceDimension; i++ )
  {
    redDimCenterOfRotationPoint[ i ] = 0.0;

    /** Returns zero when parameter was in the parameter file. */
    bool found = this->m_Configuration->ReadParameter(
      redDimCenterOfRotationPoint[ i ], "CenterOfRotationPoint", i, false );
    if( !found )
    {
      centerGivenAsPoint &= false;
    }
  }

  if( !centerGivenAsPoint )
  {
    return false;
  }

  /** copy the temporary variable into the output of this function,
   * if everything went ok.
   */
  rotationPoint = redDimCenterOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()


} // end namespace elastix

#endif // end #ifndef __elxEulerStackTransform_hxx
