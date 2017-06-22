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
#ifndef __elxAdvancedBSplineTransform_hxx
#define __elxAdvancedBSplineTransform_hxx

#include "elxAdvancedBSplineTransform.h"

#include "itkImageRegionExclusionConstIteratorWithIndex.h"
#include "vnl/vnl_math.h"


namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
AdvancedBSplineTransform< TElastix >
::AdvancedBSplineTransform()
{
} // end Constructor()


/**
 * ************ InitializeBSplineTransform ***************
 */

template< class TElastix >
unsigned int
AdvancedBSplineTransform< TElastix >
::InitializeBSplineTransform( void )
{
  /** Initialize the right BSplineTransform and GridScheduleComputer. */
  if( this->m_Cyclic )
  {
    this->m_GridScheduleComputer = CyclicGridScheduleComputerType::New();
    this->m_GridScheduleComputer->SetBSplineOrder( this->m_SplineOrder );

    if( this->m_SplineOrder == 1 )
    {
      this->m_BSplineTransform = CyclicBSplineTransformLinearType::New();
    }
    else if( this->m_SplineOrder == 2 )
    {
      this->m_BSplineTransform = CyclicBSplineTransformQuadraticType::New();
    }
    else if( this->m_SplineOrder == 3 )
    {
      this->m_BSplineTransform = CyclicBSplineTransformCubicType::New();
    }
    else
    {
      itkExceptionMacro( << "ERROR: The provided spline order is not supported." );
      return 1;
    }
  }
  else
  {
    this->m_GridScheduleComputer = GridScheduleComputerType::New();
    this->m_GridScheduleComputer->SetBSplineOrder( this->m_SplineOrder );

    if( this->m_SplineOrder == 1 )
    {
      this->m_BSplineTransform = BSplineTransformLinearType::New();
    }
    else if( this->m_SplineOrder == 2 )
    {
      this->m_BSplineTransform = BSplineTransformQuadraticType::New();
    }
    else if( this->m_SplineOrder == 3 )
    {
      this->m_BSplineTransform = BSplineTransformCubicType::New();
    }
    else
    {
      itkExceptionMacro( << "ERROR: The provided spline order is not supported." );
      return 1;
    }
  }

  this->SetCurrentTransform( this->m_BSplineTransform );
  this->m_GridUpsampler = GridUpsamplerType::New();
  this->m_GridUpsampler->SetBSplineOrder( this->m_SplineOrder );

  return 0;
} // end InitializeBSplineTransform()


/**
 * ******************* BeforeAll ***********************
 */

template< class TElastix >
int
AdvancedBSplineTransform< TElastix >
::BeforeAll( void )
{
  /** Read spline order and periodicity setting from configuration file. */
  this->m_SplineOrder = 3;
  this->GetConfiguration()->ReadParameter( this->m_SplineOrder,
    "BSplineTransformSplineOrder", this->GetComponentLabel(), 0, 0, true );
  this->m_Cyclic = false;
  this->GetConfiguration()->ReadParameter( this->m_Cyclic,
    "UseCyclicTransform", this->GetComponentLabel(), 0, 0, true );

  return this->InitializeBSplineTransform();
} // end BeforeAll()


/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
AdvancedBSplineTransform< TElastix >
::BeforeRegistration( void )
{
  /** Set initial transform parameters to a 1x1x1 grid, with deformation (0,0,0).
   * In the method BeforeEachResolution() this will be replaced by the right grid size.
   * This seems not logical, but it is required, since the registration
   * class checks if the number of parameters in the transform is equal to
   * the number of parameters in the registration class. This check is done
   * before calling the BeforeEachResolution() methods.
   */

  /** Task 1 - Set the Grid. */

  /** Declarations. */
  RegionType  gridregion;
  SizeType    gridsize;
  IndexType   gridindex;
  SpacingType gridspacing;
  OriginType  gridorigin;

  /** Fill everything with default values. */
  gridsize.Fill( 1 );
  gridindex.Fill( 0 );
  gridspacing.Fill( 1.0 );
  gridorigin.Fill( 0.0 );

  /** Set gridsize for large dimension to 4 to prevent errors when checking
     * on support region size.
     */
  gridsize.SetElement( gridsize.GetSizeDimension() - 1, 4 );

  /** Set it all. */
  gridregion.SetIndex( gridindex );
  gridregion.SetSize( gridsize );
  this->m_BSplineTransform->SetGridRegion( gridregion );
  this->m_BSplineTransform->SetGridSpacing( gridspacing );
  this->m_BSplineTransform->SetGridOrigin( gridorigin );

  /** Task 2 - Give the registration an initial parameter-array. */
  ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
  dummyInitialParameters.Fill( 0.0 );

  /** Put parameters in the registration. */
  this->m_Registration->GetAsITKBaseType()
    ->SetInitialTransformParameters( dummyInitialParameters );

  /** Precompute the B-spline grid regions. */
  this->PreComputeGridInformation();

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
AdvancedBSplineTransform< TElastix >
::BeforeEachResolution( void )
{
  /** What is the current resolution level? */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Define the grid. */
  if( level == 0 )
  {
    this->InitializeTransform();
  }
  else
  {
    /** Upsample the B-spline grid, if required. */
    this->IncreaseScale();
  }

  /** Get the PassiveEdgeWidth and use it to set the OptimizerScales. */
  unsigned int passiveEdgeWidth = 0;
  this->GetConfiguration()->ReadParameter( passiveEdgeWidth,
    "PassiveEdgeWidth", this->GetComponentLabel(), level, 0, false );
  this->SetOptimizerScales( passiveEdgeWidth );

} // end BeforeEachResolution()


/**
 * ******************** PreComputeGridInformation ***********************
 */

template< class TElastix >
void
AdvancedBSplineTransform< TElastix >
::PreComputeGridInformation( void )
{
  /** Get the total number of resolution levels. */
  unsigned int nrOfResolutions
    = this->m_Registration->GetAsITKBaseType()->GetNumberOfLevels();

  /** Set up grid schedule computer with image info. */
  this->m_GridScheduleComputer->SetImageOrigin(
    this->GetElastix()->GetFixedImage()->GetOrigin() );
  this->m_GridScheduleComputer->SetImageSpacing(
    this->GetElastix()->GetFixedImage()->GetSpacing() );
  this->m_GridScheduleComputer->SetImageDirection(
    this->GetElastix()->GetFixedImage()->GetDirection() );
  this->m_GridScheduleComputer->SetImageRegion(
    this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion() );

  /** Take the initial transform only into account, if composition is used. */
  if( this->GetUseComposition() )
  {
    this->m_GridScheduleComputer->SetInitialTransform( this->Superclass1::GetInitialTransform() );
  }

  /** Get the grid spacing schedule from the parameter file.
   *
   * Method 1: The user specifies "FinalGridSpacingInVoxels"
   * Method 2: The user specifies "FinalGridSpacingInPhysicalUnits"
   *
   * Method 1 and 2 additionally take the "GridSpacingSchedule".
   * The GridSpacingSchedule is defined by downsampling factors
   * for each resolution, for each dimension (just like the image
   * pyramid schedules). So, for 2D images, and 3 resulutions,
   * we can specify:
   * (GridSpacingSchedule 4.0 4.0 2.0 2.0 1.0 1.0)
   * Which is the default schedule, if no GridSpacingSchedule is supplied.
   */

  /** Determine which method is used. */
  bool        method1 = false;
  std::size_t count1  = this->m_Configuration
    ->CountNumberOfParameterEntries( "FinalGridSpacingInVoxels" );
  if( count1 > 0 )
  {
    method1 = true;
  }

  bool        method2 = false;
  std::size_t count2  = this->m_Configuration
    ->CountNumberOfParameterEntries( "FinalGridSpacingInPhysicalUnits" );
  if( count2 > 0 )
  {
    method2 = true;
  }

  /** Throw an exception if both methods are used. */
  if( count1 > 0 && count2 > 0 )
  {
    itkExceptionMacro( << "ERROR: You can not specify both \"FinalGridSpacingInVoxels\""
        " and \"FinalGridSpacingInPhysicalUnits\" in the parameter file." );
  }

  /** Declare variables and set defaults. */
  SpacingType finalGridSpacingInVoxels;
  SpacingType finalGridSpacingInPhysicalUnits;
  finalGridSpacingInVoxels.Fill( 16.0 );
  finalGridSpacingInPhysicalUnits.Fill( 8.0 );

  /** Method 1: Read the FinalGridSpacingInVoxels. */
  if( method1 )
  {
    for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      this->m_Configuration->ReadParameter(
        finalGridSpacingInVoxels[ dim ], "FinalGridSpacingInVoxels",
        this->GetComponentLabel(), dim, 0 );
    }

    /** Compute the grid spacing in physical units. */
    for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      finalGridSpacingInPhysicalUnits[ dim ]
        = finalGridSpacingInVoxels[ dim ]
        * this->GetElastix()->GetFixedImage()->GetSpacing()[ dim ];
    }
  }

  /** Method 2: Read the FinalGridSpacingInPhysicalUnits. */
  if( method2 )
  {
    for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      this->m_Configuration->ReadParameter(
        finalGridSpacingInPhysicalUnits[ dim ], "FinalGridSpacingInPhysicalUnits",
        this->GetComponentLabel(), dim, 0 );
    }
  }

  /** Set up a default grid spacing schedule. */
  this->m_GridScheduleComputer->SetDefaultSchedule( nrOfResolutions, 2.0 );
  GridScheduleType gridSchedule;
  this->m_GridScheduleComputer->GetSchedule( gridSchedule );

  /** Read what the user has specified. This overrules everything. */
  count2 = this->m_Configuration
    ->CountNumberOfParameterEntries( "GridSpacingSchedule" );
  unsigned int entry_nr = 0;
  if( count2 == 0 )
  {
    // keep the default schedule
  }
  else if( count2 == nrOfResolutions )
  {
    for( unsigned int res = 0; res < nrOfResolutions; ++res )
    {
      for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        this->m_Configuration->ReadParameter( gridSchedule[ res ][ dim ],
          "GridSpacingSchedule", entry_nr, false );
      }
      ++entry_nr;
    }
  }
  else if( count2 == nrOfResolutions * SpaceDimension )
  {
    for( unsigned int res = 0; res < nrOfResolutions; ++res )
    {
      for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        this->m_Configuration->ReadParameter( gridSchedule[ res ][ dim ],
          "GridSpacingSchedule", entry_nr, false );
        ++entry_nr;
      }
    }
  }
  else
  {
    xl::xout[ "error" ]
      << "ERROR: Invalid GridSpacingSchedule! The number of entries"
      << " behind the GridSpacingSchedule option should equal the"
      << " numberOfResolutions, or the numberOfResolutions * ImageDimension."
      << std::endl;
    itkExceptionMacro( << "ERROR: Invalid GridSpacingSchedule!" );
  }

  /** Output a warning that the gridspacing may be adapted to fit the Cyclic
     * behavior of the transform.
     */
  if( this->m_Cyclic )
  {
    xl::xout[ "warning" ]
      << "WARNING: The provided grid spacing may be adapted to fit the cyclic "
      << "behavior of the CyclicBSplineTransform." << std::endl;
  }

  /** Set the grid schedule and final grid spacing in the schedule computer. */
  this->m_GridScheduleComputer->SetFinalGridSpacing( finalGridSpacingInPhysicalUnits );
  this->m_GridScheduleComputer->SetSchedule( gridSchedule );

  /** Compute the necessary information. */
  this->m_GridScheduleComputer->ComputeBSplineGrid();

} // end PreComputeGridInformation()


/**
 * ******************** InitializeTransform ***********************
 */

template< class TElastix >
void
AdvancedBSplineTransform< TElastix >
::InitializeTransform( void )
{
  /** Compute the B-spline grid region, origin, and spacing. */
  RegionType    gridRegion;
  OriginType    gridOrigin;
  SpacingType   gridSpacing;
  DirectionType gridDirection;
  this->m_GridScheduleComputer->GetBSplineGrid( 0,
    gridRegion, gridSpacing, gridOrigin, gridDirection );

  /** Set it in the BSplineTransform. */
  this->m_BSplineTransform->SetGridRegion( gridRegion );
  this->m_BSplineTransform->SetGridSpacing( gridSpacing );
  this->m_BSplineTransform->SetGridOrigin( gridOrigin );
  this->m_BSplineTransform->SetGridDirection( gridDirection );

  /** Set initial parameters for the first resolution to 0.0. */
  ParametersType initialParameters( this->GetNumberOfParameters() );
  initialParameters.Fill( 0.0 );
  this->m_Registration->GetAsITKBaseType()
    ->SetInitialTransformParametersOfNextLevel( initialParameters );

} // end InitializeTransform()


/**
 * *********************** IncreaseScale ************************
 */

template< class TElastix >
void
AdvancedBSplineTransform< TElastix >
::IncreaseScale( void )
{
  /** What is the current resolution level? */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** The current grid. */
  OriginType    currentGridOrigin    = this->m_BSplineTransform->GetGridOrigin();
  SpacingType   currentGridSpacing   = this->m_BSplineTransform->GetGridSpacing();
  RegionType    currentGridRegion    = this->m_BSplineTransform->GetGridRegion();
  DirectionType currentGridDirection = this->m_BSplineTransform->GetGridDirection();

  /** The new required grid. */
  OriginType    requiredGridOrigin; requiredGridOrigin.Fill( 0.0 );
  SpacingType   requiredGridSpacing; requiredGridSpacing.Fill( 1.0 );
  RegionType    requiredGridRegion;
  DirectionType requiredGridDirection;
  this->m_GridScheduleComputer->GetBSplineGrid( level,
    requiredGridRegion, requiredGridSpacing, requiredGridOrigin, requiredGridDirection );

  /** Get the latest transform parameters. */
  ParametersType latestParameters
    = this->m_Registration->GetAsITKBaseType()->GetLastTransformParameters();

  /** Setup the GridUpsampler. */
  this->m_GridUpsampler->SetCurrentGridOrigin( currentGridOrigin );
  this->m_GridUpsampler->SetCurrentGridSpacing( currentGridSpacing );
  this->m_GridUpsampler->SetCurrentGridRegion( currentGridRegion );
  this->m_GridUpsampler->SetCurrentGridDirection( currentGridDirection );
  this->m_GridUpsampler->SetRequiredGridOrigin( requiredGridOrigin );
  this->m_GridUpsampler->SetRequiredGridSpacing( requiredGridSpacing );
  this->m_GridUpsampler->SetRequiredGridRegion( requiredGridRegion );
  this->m_GridUpsampler->SetRequiredGridDirection( requiredGridDirection );

  /** Compute the upsampled B-spline parameters. */
  ParametersType upsampledParameters;
  this->m_GridUpsampler->UpsampleParameters( latestParameters, upsampledParameters );

  /** Set the new grid definition in the BSplineTransform. */
  this->m_BSplineTransform->SetGridOrigin( requiredGridOrigin );
  this->m_BSplineTransform->SetGridSpacing( requiredGridSpacing );
  this->m_BSplineTransform->SetGridRegion( requiredGridRegion );
  this->m_BSplineTransform->SetGridDirection( requiredGridDirection );

  /** Set the initial parameters for the next level. */
  this->m_Registration->GetAsITKBaseType()
    ->SetInitialTransformParametersOfNextLevel( upsampledParameters );

  /** Set the parameters in the BsplineTransform. */
  this->m_BSplineTransform->SetParameters(
    this->m_Registration->GetAsITKBaseType()
    ->GetInitialTransformParametersOfNextLevel() );

}  // end IncreaseScale()


/**
 * ************************* ReadFromFile ************************
 */

template< class TElastix >
void
AdvancedBSplineTransform< TElastix >
::ReadFromFile( void )
{
  /** Read spline order and periodicity settings and initialize BSplineTransform. */
  this->m_SplineOrder = 3;
  this->GetConfiguration()->ReadParameter( this->m_SplineOrder,
    "BSplineTransformSplineOrder", this->GetComponentLabel(), 0, 0 );
  this->m_Cyclic = false;
  this->GetConfiguration()->ReadParameter( this->m_Cyclic,
    "UseCyclicTransform", this->GetComponentLabel(), 0, 0 );
  this->InitializeBSplineTransform();

  /** Read and Set the Grid: this is a BSplineTransform specific task. */

  /** Declarations. */
  RegionType    gridregion;
  SizeType      gridsize;
  IndexType     gridindex;
  SpacingType   gridspacing;
  OriginType    gridorigin;
  DirectionType griddirection;

  /** Fill everything with default values. */
  gridsize.Fill( 1 );
  gridindex.Fill( 0 );
  gridspacing.Fill( 1.0 );
  gridorigin.Fill( 0.0 );
  griddirection.SetIdentity();

  /** Get GridSize, GridIndex, GridSpacing and GridOrigin. */
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    this->m_Configuration->ReadParameter( gridsize[ i ], "GridSize", i );
    this->m_Configuration->ReadParameter( gridindex[ i ], "GridIndex", i );
    this->m_Configuration->ReadParameter( gridspacing[ i ], "GridSpacing", i );
    this->m_Configuration->ReadParameter( gridorigin[ i ], "GridOrigin", i );
    for( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      this->m_Configuration->ReadParameter( griddirection( j, i ),
        "GridDirection", i * SpaceDimension + j );
    }
  }

  /** Set it all. */
  gridregion.SetIndex( gridindex );
  gridregion.SetSize( gridsize );
  this->m_BSplineTransform->SetGridRegion( gridregion );
  this->m_BSplineTransform->SetGridSpacing( gridspacing );
  this->m_BSplineTransform->SetGridOrigin( gridorigin );
  this->m_BSplineTransform->SetGridDirection( griddirection );

  /** Call the ReadFromFile from the TransformBase.
   * This must be done after setting the Grid, because later the
   * ReadFromFile from TransformBase calls SetParameters, which
   * checks the parameter-size, which is based on the GridSize.
   */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* WriteToFile ************************
 */

template< class TElastix >
void
AdvancedBSplineTransform< TElastix >
::WriteToFile( const ParametersType & param ) const
{
  /** Call the WriteToFile from the TransformBase. */
  this->Superclass2::WriteToFile( param );

  /** Add some BSplineTransform specific lines. */
  xout[ "transpar" ] << std::endl << "// BSplineTransform specific" << std::endl;

  /** Get the GridSize, GridIndex, GridSpacing,
   * GridOrigin, and GridDirection of this transform. */
  SizeType      size      = this->m_BSplineTransform->GetGridRegion().GetSize();
  IndexType     index     = this->m_BSplineTransform->GetGridRegion().GetIndex();
  SpacingType   spacing   = this->m_BSplineTransform->GetGridSpacing();
  OriginType    origin    = this->m_BSplineTransform->GetGridOrigin();
  DirectionType direction = this->m_BSplineTransform->GetGridDirection();

  /** Write the GridSize of this transform. */
  xout[ "transpar" ] << "(GridSize ";
  for( unsigned int i = 0; i < SpaceDimension - 1; i++ )
  {
    xout[ "transpar" ] << size[ i ] << " ";
  }
  xout[ "transpar" ] << size[ SpaceDimension - 1 ] << ")" << std::endl;

  /** Write the GridIndex of this transform. */
  xout[ "transpar" ] << "(GridIndex ";
  for( unsigned int i = 0; i < SpaceDimension - 1; i++ )
  {
    xout[ "transpar" ] << index[ i ] << " ";
  }
  xout[ "transpar" ] << index[ SpaceDimension - 1 ] << ")" << std::endl;

  /** Set the precision of cout to 2, because GridSpacing and
   * GridOrigin must have at least one digit precision.
   */
  xout[ "transpar" ] << std::setprecision( 10 );

  /** Write the GridSpacing of this transform. */
  xout[ "transpar" ] << "(GridSpacing ";
  for( unsigned int i = 0; i < SpaceDimension - 1; i++ )
  {
    xout[ "transpar" ] << spacing[ i ] << " ";
  }
  xout[ "transpar" ] << spacing[ SpaceDimension - 1 ] << ")" << std::endl;

  /** Write the GridOrigin of this transform. */
  xout[ "transpar" ] << "(GridOrigin ";
  for( unsigned int i = 0; i < SpaceDimension - 1; i++ )
  {
    xout[ "transpar" ] << origin[ i ] << " ";
  }
  xout[ "transpar" ] << origin[ SpaceDimension - 1 ] << ")" << std::endl;

  /** Write the GridDirection of this transform. */
  xout[ "transpar" ] << "(GridDirection";
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    for( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      xout[ "transpar" ] << " " << direction( j, i );
    }
  }
  xout[ "transpar" ] << ")" << std::endl;

  /** Write the spline order and periodicity of this transform. */
  xout[ "transpar" ] << "(BSplineTransformSplineOrder " << m_SplineOrder << ")" << std::endl;
  std::string m_CyclicString = "false";
  if( m_Cyclic )
  {
    m_CyclicString = "true";
  }
  xout[ "transpar" ] << "(UseCyclicTransform \"" << m_CyclicString << "\")" << std::endl;

  /** Set the precision back to default value. */
  xout[ "transpar" ] << std::setprecision(
    this->m_Elastix->GetDefaultOutputPrecision() );

} // end WriteToFile()


/**
 * ************************* CreateTransformParametersMap ************************
 */

template< class TElastix >
void
AdvancedBSplineTransform< TElastix >
::CreateTransformParametersMap(
  const ParametersType & param,
  ParameterMapType * paramsMap ) const
{
  std::ostringstream         tmpStream;
  std::string                parameterName;
  std::vector< std::string > parameterValues;

  /** Call the CreateTransformParametersMap from the TransformBase. */
  this->Superclass2::CreateTransformParametersMap( param, paramsMap );

  /** Add some BSplineTransform specific lines. */

  /** Get the GridSize, GridIndex, GridSpacing,
   * GridOrigin, and GridDirection of this transform.
   */
  SizeType      size      = this->m_BSplineTransform->GetGridRegion().GetSize();
  IndexType     index     = this->m_BSplineTransform->GetGridRegion().GetIndex();
  SpacingType   spacing   = this->m_BSplineTransform->GetGridSpacing();
  OriginType    origin    = this->m_BSplineTransform->GetGridOrigin();
  DirectionType direction = this->m_BSplineTransform->GetGridDirection();

  /** Write the GridSize of this transform. */
  parameterName = "GridSize";
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    tmpStream.str( "" ); tmpStream << size[ i ];
    parameterValues.push_back( tmpStream.str() );
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write the GridIndex of this transform. */
  parameterName = "GridIndex";
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    tmpStream.str( "" ); tmpStream << index[ i ];
    parameterValues.push_back( tmpStream.str() );
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Set the precision of cout to 2, because GridSpacing and
   * GridOrigin must have at least one digit precision.
   */
//  xout["transpar"] << std::setprecision( 10 );

  /** Write the GridSpacing of this transform. */
  parameterName = "GridSpacing";
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    tmpStream.str( "" ); tmpStream << spacing[ i ];
    parameterValues.push_back( tmpStream.str() );
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write the GridOrigin of this transform. */
  parameterName = "GridOrigin";
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    tmpStream.str( "" ); tmpStream << origin[ i ];
    parameterValues.push_back( tmpStream.str() );
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write the GridDirection of this transform. */
  parameterName = "GridDirection";
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    for( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      tmpStream.str( "" ); tmpStream << direction( j, i );
      parameterValues.push_back( tmpStream.str() );
    }
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write the spline order and periodicity of this transform. */
  parameterName = "BSplineTransformSplineOrder";
  tmpStream.str( "" ); tmpStream << this->m_SplineOrder;
  parameterValues.push_back( tmpStream.str() );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  parameterName = "UseCyclicTransform";
  std::string cyclicString = "false";
  if( this->m_Cyclic )
  {
    cyclicString = "true";
  }
  parameterValues.push_back( cyclicString );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Set the precision back to default value. */
//  xout["transpar"] << std::setprecision(
//  this->m_Elastix->GetDefaultOutputPrecision() );

} // end CreateTransformParametersMap()


/**
 * *********************** SetOptimizerScales ***********************
 */

template< class TElastix >
void
AdvancedBSplineTransform< TElastix >
::SetOptimizerScales( const unsigned int edgeWidth )
{
  /** Some typedefs. */
  typedef itk::ImageRegionExclusionConstIteratorWithIndex< ImageType > IteratorType;
  typedef typename RegistrationType::ITKBaseType                       ITKRegistrationType;
  typedef typename ITKRegistrationType::OptimizerType                  OptimizerType;
  typedef typename OptimizerType::ScalesType                           ScalesType;
  typedef typename ScalesType::ValueType                               ScalesValueType;

  /** Define new scales. */
  const NumberOfParametersType numberOfParameters
    = this->m_BSplineTransform->GetNumberOfParameters();
  const unsigned long offset = numberOfParameters / SpaceDimension;
  ScalesType          newScales( numberOfParameters );
  newScales.Fill( itk::NumericTraits< ScalesValueType >::OneValue() );
  const ScalesValueType infScale = 10000.0;

  if( edgeWidth == 0 )
  {
    /** Just set the unit scales into the optimizer. */
    this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newScales );
    return;
  }

  /** Get the grid region information and create a fake coefficient image. */
  RegionType   gridregion = this->m_BSplineTransform->GetGridRegion();
  SizeType     gridsize   = gridregion.GetSize();
  IndexType    gridindex  = gridregion.GetIndex();
  ImagePointer coeff      = ImageType::New();
  coeff->SetRegions( gridregion );
  coeff->Allocate();

  /** Determine inset region. (so, the region with active parameters). */
  RegionType insetgridregion;
  SizeType   insetgridsize;
  IndexType  insetgridindex;
  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    insetgridsize[ i ] = static_cast< unsigned int >( vnl_math_max(
      0, static_cast< int >( gridsize[ i ] - 2 * edgeWidth ) ) );
    if( insetgridsize[ i ] == 0 )
    {
      xl::xout[ "error" ]
        << "ERROR: you specified a PassiveEdgeWidth of "
        << edgeWidth
        << ", while the total grid size in dimension "
        << i
        << " is only "
        << gridsize[ i ] << "." << std::endl;
      itkExceptionMacro( << "ERROR: the PassiveEdgeWidth is too large!" );
    }
    insetgridindex[ i ] = gridindex[ i ] + edgeWidth;
  }
  insetgridregion.SetSize( insetgridsize );
  insetgridregion.SetIndex( insetgridindex );

  /** Set up iterator over the coefficient image. */
  IteratorType cIt( coeff, coeff->GetLargestPossibleRegion() );
  cIt.SetExclusionRegion( insetgridregion );
  cIt.GoToBegin();

  /** Set the scales to infinity that correspond to edge coefficients
   * This (hopefully) makes sure they are not optimized during registration.
   */
  while( !cIt.IsAtEnd() )
  {
    const IndexType &   index      = cIt.GetIndex();
    const unsigned long baseOffset = coeff->ComputeOffset( index );
    for( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      const unsigned int scalesIndex = static_cast< unsigned int >(
        baseOffset + i * offset );
      newScales[ scalesIndex ] = infScale;
    }
    ++cIt;
  }

  /** Set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newScales );

} // end SetOptimizerScales()


} // end namespace elastix

#endif // end #ifndef __elxAdvancedBSplineTransform_hxx
