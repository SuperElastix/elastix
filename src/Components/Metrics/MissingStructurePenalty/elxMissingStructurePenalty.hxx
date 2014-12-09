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
#ifndef __elxMissingStructurePenalty_HXX__
#define __elxMissingStructurePenalty_HXX__

#include "elxMissingStructurePenalty.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
MissingStructurePenalty< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of MissingStructurePenalty metric took: "
         << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;
} // end Initialize()


/**
* ***************** BeforeAllBase ***********************
*/

template< class TElastix >
int
MissingStructurePenalty< TElastix >
::BeforeAllBase( void )
{
  this->Superclass2::BeforeAllBase();

  /** Check if the current configuration uses this metric. */
  unsigned int count = 0;
  for( unsigned int i = 0; i < this->m_Configuration
    ->CountNumberOfParameterEntries( "Metric" ); ++i )
  {
    std::string metricName = "";
    this->m_Configuration->ReadParameter( metricName, "Metric", i );
    if( metricName == "MissingStructurePenalty" ) { count++; }
  }
  if( count == 0 ) { return 0; }

  //if ( count == 1 )
  //{
  //  /** Check for appearance of "-fmesh". */

  //  check = this->m_Configuration->GetCommandLineArgument( fmeshArgument );
  //  if ( check.empty() )
  //  {
  //  elxout << "-fmesh       " << check << std::endl;
  //  m_NumberOfMeshes = 1;
  //  return 0;
  //  }
  //}

  std::string componentLabel( this->GetComponentLabel() );
  std::string metricNumber = componentLabel.substr( 6, 2 ); // strip "Metric" keep number

  /** Check Command line options and print them to the log file. */
  elxout << "Command line options from MissingStructurePenalty (" << componentLabel << "):" << std::endl;
  std::string check( "" );

  this->m_NumberOfMeshes = 0;

  for( char ch = 'A'; ch <= 'Z'; ch++ )
  {
    std::ostringstream fmeshArgument( "-fmesh", std::ios_base::ate );
    fmeshArgument << ch << metricNumber;
    check = this->m_Configuration->GetCommandLineArgument( fmeshArgument.str() );
    if( check.empty() )
    {
      break;
    }
    else
    {
      elxout << fmeshArgument.str() << "\t" << check << std::endl;
      this->m_NumberOfMeshes++;

    }

  }
  /** Return a value. */
  return 0;

} // end BeforeAllBase()


/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
MissingStructurePenalty< TElastix >
::BeforeRegistration( void )
{
  std::string componentLabel( this->GetComponentLabel() );
  std::string metricNumber = componentLabel.substr( 6, 2 ); // strip "Metric" keep number

  elxout << "MissingStructurePenalty" << metricNumber << " BeforeRegistration " << std::endl;

  FixedMeshContainerPointer meshPointerContainer = FixedMeshContainerType::New();
  meshPointerContainer->Reserve( this->m_NumberOfMeshes );
  //meshPointerContainer->CreateIndex(this->m_NumberOfMeshes-1);
  unsigned int meshNumber;
  char         ch;
  for( meshNumber = 0, ch = 'A'; meshNumber < this->m_NumberOfMeshes; ++meshNumber, ++ch )
  {
    std::ostringstream fmeshArgument( "-fmesh", std::ios_base::ate );
    fmeshArgument << ch << metricNumber;
    std::string fixedMeshName = this->GetConfiguration()->GetCommandLineArgument( fmeshArgument.str() );
    typename MeshType::Pointer fixedMesh = 0;
	if (itksys::SystemTools::GetFilenameLastExtension(fixedMeshName) == ".txt")
	{
		this->ReadTransformixPoints(fixedMeshName, fixedMesh);
	}
	else
	{
		this->ReadMesh(fixedMeshName, fixedMesh);
	}

    meshPointerContainer->SetElement( meshNumber, dynamic_cast<  MeshType * >( fixedMesh.GetPointer() ) );

  }

  this->SetFixedMeshContainer( meshPointerContainer );

  typename PointSetType::Pointer dummyPointSet = PointSetType::New();
  this->SetFixedPointSet( dummyPointSet );  // TODO solve dirty hack
  this->SetMovingPointSet( dummyPointSet ); // TODO solve dirty hack
  // itkCombinationImageToImageMetric.hxx checks if metric base class is ImageMetricType or PointSetMetricType.
  // This class is derived from SingleValuedPointSetToPointSetMetric which needs a moving pointset.
  // Without interfering with some general elastix files, this hack gives me the functionality that I needed.
  // TODO: make itkCombinationImageToImageMetric check for a base class metric that doesn't use an image or moving pointset.

} // end BeforeRegistration()


/**
 * ***************** AfterEachIteration ***********************
 */

template< class TElastix >
void
MissingStructurePenalty< TElastix >
::AfterEachIteration( void )
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** What is the current iteration number? */
  const unsigned int iter = this->m_Elastix->GetIterationCounter();

  /** Decide whether or not to write the result image this iteration. */
  bool writeResultMeshThisIteration = false;
  this->m_Configuration->ReadParameter( writeResultMeshThisIteration,
    "WriteResultMeshAfterEachIteration", "", level, 0, false );

  /** Writing result mesh. */
  if( writeResultMeshThisIteration )
  {
    std::string componentLabel( this->GetComponentLabel() );
    std::string metricNumber = componentLabel.substr( 6, 2 ); // strip "Metric" keep number

    /** Create a name for the final result. */
    std::string resultMeshFormat = "vtk";
    this->m_Configuration->ReadParameter( resultMeshFormat,
      "ResultMeshFormat", 0, false );
    char ch = 'A';
    for( MeshIdType meshId = 0; meshId < this->m_NumberOfMeshes; ++meshId, ++ch )
    {

      std::ostringstream makeFileName( "" );
      makeFileName
        << this->m_Configuration->GetCommandLineArgument( "-out" )
        << "resultmesh" << ch << metricNumber
        << "." << this->m_Configuration->GetElastixLevel()
        << ".R" << level
        << ".It" << std::setfill( '0' ) << std::setw( 7 ) << iter
        << "." << resultMeshFormat;

      try
      {
        this->WriteResultMesh( makeFileName.str().c_str(), meshId );
      }
      catch( itk::ExceptionObject & excp )
      {
        xl::xout[ "error" ] << "Exception caught: " << std::endl;
        xl::xout[ "error" ] << excp
                            << "Resuming elastix." << std::endl;
      }
    } // end for
  }   // end if

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution ***********************
 */

template< class TElastix >
void
MissingStructurePenalty< TElastix >
::AfterEachResolution( void )
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Decide whether or not to write the result mesh this resolution. */
  bool writeResultMeshThisResolution = false;
  this->m_Configuration->ReadParameter( writeResultMeshThisResolution,
    "WriteResultMeshAfterEachResolution", "", level, 0, false );

  /** Writing result mesh. */
  if( writeResultMeshThisResolution )
  {
    std::string componentLabel( this->GetComponentLabel() );
    std::string metricNumber = componentLabel.substr( 6, 2 ); // strip "Metric" keep number

    /** Create a name for the final result. */
    std::string resultMeshFormat = "vtk";
    this->m_Configuration->ReadParameter( resultMeshFormat,
      "ResultMeshFormat", 0, false );
    char ch = 'A';
    for( MeshIdType meshId = 0; meshId < this->m_NumberOfMeshes; ++meshId, ++ch )
    {

      std::ostringstream makeFileName( "" );
      makeFileName
        << this->m_Configuration->GetCommandLineArgument( "-out" )
        << "resultmesh" << ch << metricNumber
        << "." << this->m_Configuration->GetElastixLevel()
        << ".R" << level
        << "." << resultMeshFormat;

      try
      {
        this->WriteResultMesh( makeFileName.str().c_str(), meshId );
      }
      catch( itk::ExceptionObject & excp )
      {
        xl::xout[ "error" ] << "Exception caught: " << std::endl;
        xl::xout[ "error" ] << excp
                            << "Resuming elastix." << std::endl;
      }
    } // end for
  } // end if

} // end AfterEachResolution()


/**
 * ************** ReadMesh *********************
 */

template< class TElastix >
unsigned int
MissingStructurePenalty< TElastix >
::ReadMesh(
  const std::string & meshFileName,
  typename MeshType::Pointer & mesh )
{
  typedef itk::MeshFileReader< MeshType > MeshReaderType;

  /** Read the input mesh. */
  typename MeshReaderType::Pointer meshReader = MeshReaderType::New();
  meshReader->SetFileName( meshFileName.c_str() );
  elxout << "  Reading input mesh file: " << meshFileName << std::endl;
  try
  {
    meshReader->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & err )
  {
    xl::xout[ "error" ] << "  Error while opening input mesh file." << std::endl;
    xl::xout[ "error" ] << err << std::endl;
  }

  /** Some user-feedback. */
  mesh = meshReader->GetOutput();
  unsigned long nrofpoints = mesh->GetNumberOfPoints();
  elxout << "  Number of specified input points: " << nrofpoints << std::endl;

  return nrofpoints;
} // end ReadMesh()


/**
 * ******************* WriteResultMesh ********************
 */

template< class TElastix >
void
MissingStructurePenalty< TElastix >
::WriteResultMesh( const char * filename, MeshIdType meshId )
{
  /** Typedef's for writing the output mesh. */
  typedef itk::MeshFileWriter< MeshType > MeshWriterType;

  /** Create writer. */
  typename MeshWriterType::Pointer meshWriter = MeshWriterType::New();

  /** Setup the pipeline. */

  /** Set the points of the latest transformation. */
  const MappedMeshContainerPointer mappedMeshContainer = this->GetMappedMeshContainer();
  FixedMeshPointer                 mappedMesh          = mappedMeshContainer->ElementAt( meshId );

  /** Use pointer to the mesh data of fixedMesh; const_cast are assumed since outputMesh will only be used for writing the output*/
  FixedMeshConstPointer fixedMesh        = this->GetFixedMeshContainer()->ElementAt( meshId );
  bool                  tempSetPointData = ( mappedMesh->GetPointData() == NULL );
  bool                  tempSetCells     = ( mappedMesh->GetCells() == NULL );
  bool                  tempSetCellData  = ( mappedMesh->GetCellData() == NULL );

  if( tempSetPointData )
  {
    // temporarily set pointdata
    mappedMesh->SetPointData( const_cast< typename MeshType::PointDataContainer * >( fixedMesh->GetPointData() ) );
  }

  if( tempSetCells )
  {
    // temporarily set cells
    mappedMesh->SetCells( const_cast< typename MeshType::CellsContainer * >( fixedMesh->GetCells() ) );
  }
  if( tempSetCellData )
  {
    // temporarily set celldata
    mappedMesh->SetCellData( const_cast< typename MeshType::CellDataContainer * >( fixedMesh->GetCellData() ) );
  }

  mappedMesh->Modified();
  mappedMesh->Update();

  meshWriter->SetInput( mappedMesh );
  meshWriter->SetFileName( filename );

  try
  {
    meshWriter->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "MissingStructurePenalty - WriteResultMesh()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing mapped mesh.\n";
    excp.SetDescription( err_str );

    /** Pass the exception to an higher level. */
    throw excp;
  }

  if( tempSetPointData )
  {
    // restore pointdata as undefined
    mappedMesh->SetPointData( NULL );
  }

  if( tempSetCells )
  {
    // restore cells as undefined
    mappedMesh->SetCells( NULL );
  }
  if( tempSetCellData )
  {
    // restore celldata as undefined
    mappedMesh->SetCellData( NULL );
  }

} // end WriteResultMesh()


/**
 * ******************* ReadTransformixPoints ********************
 */

template< class TElastix >
unsigned int
MissingStructurePenalty< TElastix >
::ReadTransformixPoints(
  const std::string & filename,
  typename MeshType::Pointer & mesh )   //const
{
  /*
  FB: Majority of the code is copied from elxTransformBase.hxx: TransformPointsSomePoints()
Function to read 2d structures by reading elastix point files (transformix format) and connecting
the sequence of points to form a 2d connected polydata contour.
  */
  /** Typedef's. */
  typedef typename FixedImageType::RegionType           FixedImageRegionType;
  typedef typename FixedImageType::PointType            FixedImageOriginType;
  typedef typename FixedImageType::SpacingType          FixedImageSpacingType;
  typedef typename FixedImageType::IndexType            FixedImageIndexType;
  typedef typename FixedImageIndexType::IndexValueType  FixedImageIndexValueType;
  typedef typename MovingImageType::IndexType           MovingImageIndexType;
  typedef
    itk::ContinuousIndex< double, FixedImageDimension >   FixedImageContinuousIndexType;
  typedef
    itk::ContinuousIndex< double, MovingImageDimension >  MovingImageContinuousIndexType;
  typedef typename FixedImageType::DirectionType FixedImageDirectionType;

  typedef bool DummyIPPPixelType;
  typedef itk::DefaultStaticMeshTraits<
    DummyIPPPixelType, FixedImageDimension,
    FixedImageDimension, CoordRepType >                  MeshTraitsType;
  typedef itk::PointSet< DummyIPPPixelType,
    FixedImageDimension, MeshTraitsType >                PointSetType;
  typedef itk::TransformixInputPointFileReader<
    PointSetType >                                      IPPReaderType;
  typedef itk::Vector< float, FixedImageDimension > DeformationVectorType;

  /** Construct an ipp-file reader. */
  typename IPPReaderType::Pointer ippReader = IPPReaderType::New();
  ippReader->SetFileName( filename.c_str() );

  /** Read the input points. */
  elxout << "  Reading input point file: " << filename << std::endl;
  try
  {
    ippReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    xl::xout[ "error" ] << "  Error while opening input point file." << std::endl;
    xl::xout[ "error" ] << err << std::endl;
  }

  /** Some user-feedback. */
  if( ippReader->GetPointsAreIndices() )
  {
    elxout << "  Input points are specified as image indices." << std::endl;
  }
  else
  {
    elxout << "  Input points are specified in world coordinates." << std::endl;
  }
  unsigned int nrofpoints = ippReader->GetNumberOfPoints();
  elxout << "  Number of specified input points: " << nrofpoints << std::endl;

  /** Get the set of input points. */
  typename PointSetType::Pointer inputPointSet = ippReader->GetOutput();

  /** Create the storage classes. */
  std::vector< FixedImageIndexType > inputindexvec(  nrofpoints );
  //MeshType::PointsContainerPointer inputpointvec = MeshType::PointsContainer
  //inputpointvec->Reserve(nrofpoints);
  std::vector< InputPointType > inputpointvec(  nrofpoints );
  //std::vector< OutputPointType >        outputpointvec( nrofpoints );
  std::vector< FixedImageIndexType >   outputindexfixedvec( nrofpoints );
  std::vector< MovingImageIndexType >  outputindexmovingvec( nrofpoints );
  std::vector< DeformationVectorType > deformationvec( nrofpoints );

  /** Make a temporary image with the right region info,
  * which we can use to convert between points and indices.
  * By taking the image from the resampler output, the UseDirectionCosines
  * parameter is automatically taken into account. */
  FixedImageRegionType region;
  FixedImageOriginType origin
    = this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputOrigin();
  FixedImageSpacingType spacing
    = this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputSpacing();
  FixedImageDirectionType direction
    = this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputDirection();
  region.SetIndex(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputStartIndex() );
  region.SetSize(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetSize() );

  typename FixedImageType::Pointer dummyImage = FixedImageType::New();
  dummyImage->SetRegions( region );
  dummyImage->SetOrigin( origin );
  dummyImage->SetSpacing( spacing );
  dummyImage->SetDirection( direction );

  /** Temp vars */
  FixedImageContinuousIndexType  fixedcindex;
  MovingImageContinuousIndexType movingcindex;

  /** Also output moving image indices if a moving image was supplied. */
  bool alsoMovingIndices = false;
  typename MovingImageType::Pointer movingImage = this->GetElastix()->GetMovingImage();
  if( movingImage.IsNotNull() )
  {
    alsoMovingIndices = true;
  }

  /** Read the input points, as index or as point. */
  if( !( ippReader->GetPointsAreIndices() ) )
  {
    for( unsigned int j = 0; j < nrofpoints; j++ )
    {
      /** Compute index of nearest voxel in fixed image. */
      InputPointType point; point.Fill( 0.0f );
      inputPointSet->GetPoint( j, &point );
      inputpointvec[ j ] = point;
      dummyImage->TransformPhysicalPointToContinuousIndex(
        point, fixedcindex );
      for( unsigned int i = 0; i < FixedImageDimension; i++ )
      {
        inputindexvec[ j ][ i ] = static_cast< FixedImageIndexValueType >(
          vnl_math_rnd( fixedcindex[ i ] ) );
      }
    }
  }
  else   //so: inputasindex
  {
    for( unsigned int j = 0; j < nrofpoints; j++ )
    {
      /** The read point from the inutPointSet is actually an index
      * Cast to the proper type.
      */
      InputPointType point; point.Fill( 0.0f );
      inputPointSet->GetPoint( j, &point );
      for( unsigned int i = 0; i < FixedImageDimension; i++ )
      {
        inputindexvec[ j ][ i ] = static_cast< FixedImageIndexValueType >(
          vnl_math_rnd( point[ i ] ) );
      }
      /** Compute the input point in physical coordinates. */
      dummyImage->TransformIndexToPhysicalPoint(
        inputindexvec[ j ], inputpointvec[ j ] );
    }
  }
  /** FB: create a mesh containing the points**/

  //MeshType::PointsContainer * meshpointset = dynamic_cast<MeshType::PointsContainer *>(inputpointvec);
  mesh = MeshType::New();
  mesh->SetPoints( inputPointSet->GetPoints() );

  /** Floris: make connected mesh (polygon) if data is 2d by assuming the sequence of points being connected**/
  if( FixedImageDimension == 2 )
  {
    typedef typename MeshType::CellType::CellAutoPointer CellAutoPointer;
    typedef itk::LineCell< typename MeshType::CellType > LineType;

    for( int i = 0; i < nrofpoints; ++i )
    {

      // Create a link to the previous point in the column (below the current point)
      CellAutoPointer line;
      line.TakeOwnership(  new LineType  );

      line->SetPointId( 0, i ); // line between points 0 and 1
      line->SetPointId( 1, ( i + 1 ) % nrofpoints );
      //std::cout << "Linked point: " << MeshIndex << " and " << MeshIndex - 1 << std::endl;
      mesh->SetCell( i, line );
    }
  }

  std::cout << "mesh->GetNumberOfCells()" << mesh->GetNumberOfCells() << std::endl;
  std::cout << "mesh->GetNumberOfPoints()" << mesh->GetNumberOfPoints() << std::endl;
  typename MeshType::PointsContainer::ConstPointer points = mesh->GetPoints();

  typename MeshType::PointsContainerConstIterator pointsBegin = points->Begin();
  typename MeshType::PointsContainerConstIterator pointsEnd   = points->End();
  for( pointsBegin; pointsBegin != pointsEnd; ++pointsBegin )
  {
    std::cout << "point " << pointsBegin->Index() << ": " << pointsBegin->Value().GetVnlVector() << std::endl;
  }

  typedef typename MeshType::CellsContainer::Iterator CellIterator;
  CellIterator cellIterator = mesh->GetCells()->Begin();
  CellIterator CellsEnd     = mesh->GetCells()->End();

  typename CellInterfaceType::PointIdIterator beginpointer;
  typename CellInterfaceType::PointIdIterator endpointer;

  for( cellIterator; cellIterator != CellsEnd; ++cellIterator )
  {
    std::cout << "Cell Index: " << cellIterator->Index() << std::endl;
    beginpointer = cellIterator->Value()->PointIdsBegin();
    endpointer   = cellIterator->Value()->PointIdsEnd();

    for( beginpointer; beginpointer != endpointer; ++beginpointer )
    {
      std::cout << "Id: " << *beginpointer << std::endl;
    }
  }
  return nrofpoints;
} // end ReadTransformixPoints()

} // end namespace elastix

#endif // end #ifndef __elxMissingStructurePenalty_HXX__

