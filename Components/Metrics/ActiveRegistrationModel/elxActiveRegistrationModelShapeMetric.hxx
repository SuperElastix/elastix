/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxActiveRegistrationModelShapeMetric_hxx__
#define __elxActiveRegistrationModelShapeMetric_hxx__

#include <typeinfo>
#include <iterator>

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
ActiveRegistrationModelShapeMetric< TElastix >
::Initialize( void )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of ActiveRegistrationModel metric took: "
         << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeAllBase ***********************
 */

template< class TElastix >
int
ActiveRegistrationModelShapeMetric< TElastix >
::BeforeAllBase( void )
{

  this->Superclass2::BeforeAllBase();

  std::string componentLabel( this->GetComponentLabel() );
  std::string metricNumber = componentLabel.substr( 6, 2 ); // strip "Metric" keep number
  this->SetMetricNumber( std::stoul( metricNumber ) );
  
  // Paths to shape models for loading
  this->m_LoadShapeModelFileNames = ReadPath( std::string("LoadShapeModel") );

  // Paths to shape models for loading
  this->m_SaveShapeModelFileNames = ReadPath( std::string("SaveShapeModel") );
  
  // Paths to directories with shapes for model building
  this->m_ShapeDirectories = ReadPath( std::string("BuildShapeModel") );

  if( this->m_SaveShapeModelFileNames.size() > 0 )
  {
    if( this->m_SaveShapeModelFileNames.size() != this->m_ShapeDirectories.size() )
    {
      itkExceptionMacro( "The number of destinations for saving shape models must match the number of directories." )
    }
  }
  
  if( this->m_ShapeDirectories.size() > 0 )
  {
    // Reference shapes for model building
    this->m_ReferenceFilenames = ReadPath("ReferenceShape");

    if (this->m_ReferenceFilenames.size() != this->m_ShapeDirectories.size())
    {
      itkExceptionMacro(<< "The number of reference shapes does not match the number of directories given.");
    }
  }
  
  // At least one model must be specified
  if( 0 == ( this->m_LoadShapeModelFileNames.size() + this->m_ShapeDirectories.size() ) )
  {
    itkExceptionMacro( << "No statistical shape model specified for " << this->GetComponentLabel() << "." << std::endl
                       << "  Specify previously built models with (LoadShapeModel" << this->GetMetricNumber()
                       << " \"path/to/hdf5/file1\" \"path/to/hdf5/file2\" ) or " << std::endl 
                       << "  specify directories with shapes using (BuildShapeModel" << this->GetMetricNumber()
                       << " \"path/to/directory1\" \"path/to/directory2\") and " << std::endl
                       << "  corresponding reference shapes using \"(ReferenceShape" << this->GetMetricNumber()
                       << " \"path/to/reference1\" \"path/to/reference2\")." << std::endl
    );
  }
  
  return 0;

} // end BeforeAllBase()



/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
ActiveRegistrationModelShapeMetric< TElastix >
::BeforeRegistration( void )
{
  StatisticalModelContainerPointer statisticalModelContainer = StatisticalModelContainerType::New();
  statisticalModelContainer->Reserve( this->m_LoadShapeModelFileNames.size() + this->m_ShapeDirectories.size() );

  // Load models
  if( this->m_LoadShapeModelFileNames.size() > 0 )
  {
    elxout << std::endl << "Loading models for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << " ... " << std::endl;
    
    for( StatisticalModelIdType statisticalModelId = 0; statisticalModelId < this->m_LoadShapeModelFileNames.size(); ++statisticalModelId )
    {
      // Load model
      StatisticalModelPointer statisticalModel;
      try
      {
        StatisticalModelRepresenterPointer representer = StatisticalModelRepresenterType::New();
        statisticalModel = itk::StatismoIO< StatisticalModelMeshType >::LoadStatisticalModel( representer, this->m_LoadShapeModelFileNames[ statisticalModelId ] );
        statisticalModelContainer->SetElement( statisticalModelId, statisticalModel );
      }
      catch( statismo::StatisticalModelException &e )
      {
        itkExceptionMacro( "Error loading statistical shape model: " << e.what() );
      }
      
      elxout << "  Loaded model " << this->m_LoadShapeModelFileNames[ statisticalModelId ].c_str() << "." << std::endl;
      elxout << "  Number of principal components: " << statisticalModel->GetNumberOfPrincipalComponents() << "." << std::endl;
      elxout << "  Eigenvalues: " << statisticalModel->GetPCAVarianceVector().apply(std::sqrt) << "." << std::endl;
      elxout << "  Noise variance: " << statisticalModel->GetNoiseVariance() << "." << std::endl;
    }
  }
  
  // Build models
  if( this->m_ShapeDirectories.size() > 0 )
  {
    elxout << std::endl << "Building models for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << " ... " << std::endl;

    // Noise parameter for probabilistic pca model
    StatisticalModelVectorType noiseVariance = this->ReadNoiseVariance();

    // Number of principal components to keep by variance
    StatisticalModelVectorType totalVariance = this->ReadTotalVariance();
    
    // Loop over all data directories
    for( StatisticalModelIdType statisticalModelId = 0; statisticalModelId < this->m_ShapeDirectories.size(); ++statisticalModelId )
    { 
      // Load data
      StatisticalModelDataManagerPointer dataManager;
      try 
      {
        dataManager = this->ReadMeshesFromDirectory(this->m_ShapeDirectories[ statisticalModelId ],
                                                    this->m_ReferenceFilenames[ statisticalModelId ]);
      }
      catch( statismo::StatisticalModelException &e )
      {
        itkExceptionMacro( "Error loading samples in " << this->m_ShapeDirectories[ statisticalModelId ] <<": " << e.what() );
      }
      
      // Build model
      elxout << "  Building statistical shape model for metric " << this->GetMetricNumber() << "... ";
      StatisticalModelPointer statisticalModel;
      try
      {
        StatisticalModelBuilderPointer pcaModelBuilder = StatisticalModelBuilderType::New();
        statisticalModel = pcaModelBuilder->BuildNewModel( dataManager->GetData(), noiseVariance[ statisticalModelId ] );
        elxout << "  Done." << std::endl
               << "  Number of modes: " << statisticalModel->GetNumberOfPrincipalComponents() << "." << std::endl
               << "  Eigenvalues: " << statisticalModel->GetPCAVarianceVector().apply(std::sqrt) << "." << std::endl
               << "  Noise variance: " << statisticalModel->GetNoiseVariance()
               << "." << std::endl;
        
        // Pick out first principal components
        if( totalVariance[ statisticalModelId ] < 1.0 )
        {
          elxout << "  Reducing model to " << totalVariance[ statisticalModelId ] * 100.0 << "% variance ... ";
          StatisticalModelReducedVarianceBuilderPointer reducedVarianceModelBuilder = StatisticalModelReducedVarianceBuilderType::New();
          statisticalModel = reducedVarianceModelBuilder->BuildNewModelWithVariance( statisticalModel, totalVariance[ statisticalModelId ] );
          elxout << " Done." << std::endl
                 << "  Number of modes retained: " << statisticalModel->GetNumberOfPrincipalComponents() << "." << std::endl;
        }
      }
      catch( statismo::StatisticalModelException& e )
      {
        itkExceptionMacro( << "Error building statistical shape model: " << e.what() );
      }

      if( this->m_SaveShapeModelFileNames.size() > 0 )
      {
        elxout << "  Saving shape model " << statisticalModelId << " to " << this->m_SaveShapeModelFileNames[ statisticalModelId ] << ". " << std::endl;
        try
        {
          itk::StatismoIO< StatisticalModelMeshType >::SaveStatisticalModel(statisticalModel, this->m_SaveShapeModelFileNames[ statisticalModelId ]);
        }
        catch( statismo::StatisticalModelException& e )
        {
          itkExceptionMacro( "Could not save shape model to " << this->m_SaveShapeModelFileNames[ statisticalModelId ] << ".");
        }
      }

      statisticalModelContainer->SetElement( statisticalModelId, statisticalModel );
    }
  }

  this->SetStatisticalModelContainer( statisticalModelContainer );

  // SingleValuedPointSetToPointSetMetric (from which this class is derived) needs a fixed and moving point set
  typename FixedPointSetType::Pointer fixedDummyPointSet = FixedPointSetType::New();
  typename MovingPointSetType::Pointer movingDummyPointSet = MovingPointSetType::New();
  this->SetFixedPointSet( fixedDummyPointSet );  // FB: TODO solve hack
  this->SetMovingPointSet( movingDummyPointSet ); // FB: TODO solve hack

  std::cout << std::endl;
} // end BeforeRegistration()



/**
 * ***************** loadShapesFromDirectory ***********************
 */

template< class TElastix >
typename ActiveRegistrationModelShapeMetric< TElastix >::StatisticalModelDataManagerPointer
ActiveRegistrationModelShapeMetric< TElastix >
::ReadMeshesFromDirectory(
        std::string shapeDataDirectory,
        std::string referenceFilename)
{
  
  itk::Directory::Pointer directory = itk::Directory::New();
  if( !directory->Load( shapeDataDirectory.c_str() ) )
  {
    itkExceptionMacro( "No files found in " << shapeDataDirectory << ".");
  }
  
  // Read reference shape
  StatisticalModelMeshPointer reference = StatisticalModelMeshType::New();
  if( this->ReadMesh( referenceFilename, reference ) == 0 )
  {
    itkExceptionMacro( "Failed to read reference file " << referenceFilename << ".");
  }

  StatisticalModelRepresenterPointer representer = StatisticalModelRepresenterType::New();
  representer->SetReference( reference );

  StatisticalModelDataManagerPointer dataManager = StatisticalModelDataManagerType::New();
  dataManager->SetRepresenter( representer.GetPointer() );
  
  for( int i = 0; i < directory->GetNumberOfFiles(); ++i )
  {
    const char * filename = directory->GetFile( i );
    if( std::strcmp( filename, referenceFilename.c_str() ) == 0 || std::strcmp( filename, "." ) == 0 || std::strcmp( filename, ".." ) == 0 )
    {
      continue;
    }

    std::string fullpath = shapeDataDirectory + "/" + filename;
    StatisticalModelMeshPointer mesh = StatisticalModelMeshType::New();

    unsigned long numberOfMeshPoints = this->ReadMesh( fullpath.c_str(), mesh );
    if( numberOfMeshPoints > 0 )
    {
      dataManager->AddDataset( mesh, fullpath.c_str() );
    }
  }
  
  return dataManager;
}



/**
 * ************** ReadShape *********************
 */

template< class TElastix >
unsigned long
ActiveRegistrationModelShapeMetric< TElastix >
::ReadMesh(
  const std::string& meshFilename,
  StatisticalModelMeshPointer& mesh )
{
  // Read the input mesh. */
  MeshReaderPointer meshReader = MeshReaderType::New();
  meshReader->SetFileName( meshFilename.c_str() );

  elxout << "  Reading input mesh file: " << meshFilename << " ... ";
  try
  {
    meshReader->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & err )
  {
    elxout << "skipping " << meshFilename << " (not a valid mesh file or file does not exist)." << std::endl;
    return 0;
  }

  // Some user-feedback. 
  mesh = meshReader->GetOutput();
  unsigned long numberOfPoints = mesh->GetNumberOfPoints();
  if( numberOfPoints > 0 )
  {
    elxout << "read " << numberOfPoints << " points." << std::endl;
  }
  else
  {
    elxout << "skipping " << meshFilename << " (no points in mesh file)." << std::endl;
  }

  return numberOfPoints;
} // end ReadMesh()



/**
 * ******************* WriteMesh ********************
 */

template< class TElastix >
void
ActiveRegistrationModelShapeMetric< TElastix >
::WriteMesh( const char * filename, StatisticalModelMeshType mesh )
{
  // Create writer.
  MeshFileWriterPointer meshWriter = MeshFileWriterType::New();

  meshWriter->SetInput( mesh );
  meshWriter->SetFileName( filename );

  try
  {
    meshWriter->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    // Add information to the exception.
    excp.SetLocation( "ActiveRegistrationModel - WriteMesh()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing mesh.\n";
    excp.SetDescription( err_str );

    // Pass the exception to an higher level.
    throw excp;
  }
} // end WriteMesh()



/**
 * ***************** ReadPath ***********************
 */

template< class TElastix >
typename ActiveRegistrationModelShapeMetric< TElastix>::StatisticalModelPathVectorType
ActiveRegistrationModelShapeMetric< TElastix >
::ReadPath( std::string path )
{
  std::ostringstream key;
  key << path << this->GetMetricNumber();

  StatisticalModelPathVectorType pathVector;
  for( unsigned int i = 0; i < this->GetConfiguration()->CountNumberOfParameterEntries( key.str() ); ++i )
  {
    std::string value = "";
    this->m_Configuration->ReadParameter( value, key.str(), i );
    pathVector.push_back( value );
  }
  
  return pathVector;
}



/**
 * ***************** ReadNoiseVariance ***********************
 */

template< class TElastix >
typename ActiveRegistrationModelShapeMetric< TElastix >::StatisticalModelVectorType
ActiveRegistrationModelShapeMetric< TElastix >
::ReadNoiseVariance()
{
  std::ostringstream key( "NoiseVariance", std::ios_base::ate );
  key << this->GetMetricNumber();

  StatisticalModelVectorType noiseVarianceVector = StatisticalModelVectorType( this->m_ShapeDirectories.size(), 0.0 );
  unsigned int n = this->GetConfiguration()->CountNumberOfParameterEntries( key.str() );
  
  if( n == 0 )
  {
    elxout << "WARNING: NoiseVariance not specified for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << "." << std::endl
           << "  A default value of " << noiseVarianceVector[ 0 ] << " will be used (non-probabilistic PCA) for metric " << this->GetMetricNumber() << "." << std::endl;
    
    return noiseVarianceVector;
  }
  
  for(unsigned int i = 0; i < this->GetConfiguration()->CountNumberOfParameterEntries( key.str() ); ++i)
  {
    std::string value = "";
    this->m_Configuration->ReadParameter( value, key.str(), i );
    
    char *e;
    errno = 0;
    double noiseVariance = std::strtod( value.c_str(), &e );
    
    if ( *e != '\0' || // error, we didn't consume the entire string
         errno != 0 )  // error, overflow or underflow
    {
      itkExceptionMacro( << "Invalid number format for NoiseVariance entry " << i << "." );
    }
    
    if( noiseVariance < 0 )
    {
      itkExceptionMacro( << "NoiseVariance entry number " << i << " is negative (" << noiseVariance << "). Variance must be positive by definition. Please correct your parameter file." );
    }
    
    noiseVarianceVector[ i ] = noiseVariance;
  }
  
  if( n == 1 && noiseVarianceVector.size() > 1 )
  {
    // Fill the rest of the elements
    noiseVarianceVector.fill( noiseVarianceVector[ 0 ] );
  }
  
  return noiseVarianceVector;
}



/**
 * ***************** ReadTotalVariance ***********************
 */

template< class TElastix >
typename ActiveRegistrationModelShapeMetric< TElastix >::StatisticalModelVectorType
ActiveRegistrationModelShapeMetric< TElastix >
::ReadTotalVariance()
{
  std::ostringstream key( "TotalVariance", std::ios_base::ate );
  key << this->GetMetricNumber();

  StatisticalModelVectorType totalVarianceVector = StatisticalModelVectorType( this->m_ShapeDirectories.size(), 1.0 );
  unsigned int n = this->GetConfiguration()->CountNumberOfParameterEntries( key.str() );
  
  if( n == 0 )
  {
    elxout << "WARNING: TotalVariance not specified for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << "." << std::endl
           << "  A default value of 1.0 will be used (all principal componontents) for metric " << this->GetMetricNumber() << "." << std::endl;
    
    return totalVarianceVector;
  }
  
  for(unsigned int i = 0; i < this->GetConfiguration()->CountNumberOfParameterEntries( key.str() ); ++i)
  {
    std::string value = "";
    this->m_Configuration->ReadParameter( value, key.str(), i );
    
    char *e;
    errno = 0;
    double totalVariance = std::strtod( value.c_str(), &e );
    
    if ( *e != '\0' || // error, we didn't consume the entire string
         errno != 0 )  // error, overflow or underflow
    {
      itkExceptionMacro( << "Invalid number format for NoiseVariance entry " << i << "." );
    }
    
    if( totalVariance < 0.0 || totalVariance > 1.0 )
    {
      itkExceptionMacro( << "TotalVariance entries must lie in [0.0; 1.0] but entry number " << i << " is " << totalVariance << ". Please correct your parameter file." );
    }
    
    totalVarianceVector[ i ] = totalVariance;
  }
  
  if( n == 1 && totalVarianceVector.size() > 1 )
  {
    // Need to fill the rest of the elements
    totalVarianceVector.fill( totalVarianceVector[ 0 ] );
  }
  
  return totalVarianceVector;
}




/**
 * ***************** AfterEachIteration ***********************
 */

template< class TElastix >
void
ActiveRegistrationModelShapeMetric< TElastix >
::AfterEachIteration( void )
{
  const unsigned int iter = this->m_Elastix->GetIterationCounter();

  /** Decide whether or not to write final model image */
  bool writeShapeModelReconstructionAfterEachIteration = false;
  this->m_Configuration->ReadParameter( writeShapeModelReconstructionAfterEachIteration,
                                        "WriteShapeModelReconstructionAfterEachIteration", 0, false );

  if( writeShapeModelReconstructionAfterEachIteration ) {

    for( unsigned int statisticalModelId = 0; statisticalModelId < this->GetStatisticalModelContainer()->Size(); statisticalModelId++ ) {
      StatisticalModelVectorType coeffs = this->GetStatisticalModelContainer()->ElementAt( statisticalModelId )->ComputeCoefficients(
              this->TransformMesh( this->GetStatisticalModelContainer()->ElementAt( statisticalModelId)->DrawMean() ) );
        std::string shapeFormat = "vtk";
        this->m_Configuration->ReadParameter( shapeFormat, "ResultShapeFormat", 0, false );

      std::ostringstream makeFileName("");
      makeFileName
              << this->m_Configuration->GetCommandLineArgument("-out")
              << "Metric" << this->GetMetricNumber()
              << "ShapeModel" << statisticalModelId
              << "Iteration" << iter
              << "Shape." << shapeFormat;

      elxout << "  Writing shape model " << statisticalModelId << " shape for "
             << this->GetComponentLabel() << " after iteration " << iter << " to " << makeFileName.str() << ". ";
      elxout << " Coefficents are [" << coeffs << "]." << std::endl;

      MeshFileWriterPointer meshWriter = MeshFileWriterType::New();
      meshWriter->SetInput( this->GetStatisticalModelContainer()->ElementAt(statisticalModelId)->DrawSample( coeffs ) );
      meshWriter->SetFileName(makeFileName.str());
      meshWriter->Update();
    }
  }
}



/**
 * ***************** AfterEachResolution ***********************
 */

template< class TElastix >
void
ActiveRegistrationModelShapeMetric< TElastix >
::AfterEachResolution( void )
{
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Decide whether or not to write model image after each resolution */
  bool writeShapeModelReconstructionAfterEachResolution = false;
  this->m_Configuration->ReadParameter( writeShapeModelReconstructionAfterEachResolution,
                                        "WriteShapeModelReconstructionAfterEachResolution", 0, false );

  if( writeShapeModelReconstructionAfterEachResolution ) {
    for( unsigned int statisticalModelId = 0; statisticalModelId < this->GetStatisticalModelContainer()->Size(); statisticalModelId++ ) {
      StatisticalModelVectorType coeffs = this->GetStatisticalModelContainer()->ElementAt( statisticalModelId )->ComputeCoefficients(
              this->TransformMesh( this->GetStatisticalModelContainer()->ElementAt( statisticalModelId)->DrawMean() ) );


      std::string shapeFormat = "vtk";
      this->m_Configuration->ReadParameter( shapeFormat, "ResultShapeFormat", 0, false );

      std::ostringstream makeFileName("");
      makeFileName
              << this->m_Configuration->GetCommandLineArgument("-out")
              << "Metric" << this->GetMetricNumber()
              << "IntensityModel" << statisticalModelId
              << "Resolution" << level
              << "Image." << shapeFormat;

      elxout << "  Writing intensity model " << statisticalModelId << " image " << " for "
             << this->GetComponentLabel() << " after resolution " << level << " to " << makeFileName.str() << ". ";
      elxout << " Coefficents are [" << coeffs << "]." << std::endl;

      MeshFileWriterPointer meshWriter = MeshFileWriterType::New();
      meshWriter->SetInput( this->GetStatisticalModelContainer()->ElementAt(statisticalModelId)->DrawSample( coeffs ) );
      meshWriter->SetFileName(makeFileName.str());
      meshWriter->Update();
    }
  }
} // end AfterEachResolution()


/**
 * ***************** AfterRegistration ***********************
 */

template< class TElastix >
void
ActiveRegistrationModelShapeMetric< TElastix >
::AfterRegistration( void )
{
  /** Decide whether or not to write the mean images */
  bool writeShapeModelMeanShape = false;
  this->m_Configuration->ReadParameter( writeShapeModelMeanShape,
                                        "WriteShapeModelMeanShapeAfterRegistration", 0, false );

  std::string shapeFormat = "vtk";
  this->m_Configuration->ReadParameter( shapeFormat, "ResultShapeFormat", 0, false );

  if( writeShapeModelMeanShape )
  {
    for( unsigned int statisticalModelId = 0; statisticalModelId < this->GetStatisticalModelContainer()->Size(); statisticalModelId++ )
    {
      std::ostringstream makeFileName( "" );
      makeFileName
      << this->m_Configuration->GetCommandLineArgument( "-out" )
      << "Metric" << this->GetMetricNumber()
      << "ShapeModel" << statisticalModelId
      << "MeanShape." << shapeFormat;

      elxout << "  Writing statistical model " << statisticalModelId << " mean shape for " << this->GetComponentLabel() << " to " << makeFileName.str() << std::endl;

      MeshFileWriterPointer meshWriter = MeshFileWriterType::New();
      meshWriter->SetInput( this->GetStatisticalModelContainer()->ElementAt( statisticalModelId )->DrawMean() );
      meshWriter->SetFileName( makeFileName.str() );
      meshWriter->Update();
    }
  }

  /** Decide whether or not to write final model image */
  bool writeShapeModelFinalReconstruction = false;
  this->m_Configuration->ReadParameter( writeShapeModelFinalReconstruction,
                                        "WriteShapeModelFinalReconstructionAfterRegistration", 0, false );

  /** Decide whether or not to write sample probability */
  bool writeShapeModelFinalReconstructionProbability = false;
  this->m_Configuration->ReadParameter( writeShapeModelFinalReconstructionProbability,
                                        "WriteShapeModelFinalShapeProbabilityAfterRegistration", 0, false );

  if( writeShapeModelFinalReconstruction || writeShapeModelFinalReconstructionProbability )
  {
    for( unsigned int statisticalModelId = 0; statisticalModelId < this->GetStatisticalModelContainer()->Size(); statisticalModelId++ )
    {
      StatisticalModelVectorType coeffs = this->GetStatisticalModelContainer()->ElementAt( statisticalModelId )->ComputeCoefficients(
              this->TransformMesh( this->GetStatisticalModelContainer()->ElementAt( statisticalModelId)->DrawMean() ) );

      if( writeShapeModelFinalReconstruction )
      {
        std::ostringstream makeFileName( "" );
        makeFileName
                << this->m_Configuration->GetCommandLineArgument( "-out" )
                << "Metric" << this->GetMetricNumber()
                << "ShapeModel" << statisticalModelId
                << "FinalShape." << shapeFormat;

        elxout << "  Writing statistical model final image " << statisticalModelId << " for " << this->GetComponentLabel() << " to " << makeFileName.str() << std::endl;

        MeshFileWriterPointer meshWriter = MeshFileWriterType::New();
        meshWriter->SetInput( this->GetStatisticalModelContainer()->ElementAt( statisticalModelId )->DrawSample( coeffs ) );
        meshWriter->SetFileName( makeFileName.str() );
        meshWriter->Update();
      }

      if( writeShapeModelFinalReconstructionProbability ) {
        std::ostringstream makeProbFileName;
        makeProbFileName
                << this->m_Configuration->GetCommandLineArgument("-out")
                << "Metric" << this->GetMetricNumber()
                << "ShapeModel" << statisticalModelId
                << "Probability.txt";

        elxout << "  Writing shape model " << statisticalModelId << " final shape probablity for " << this->GetComponentLabel()
               << " to " << makeProbFileName.str() << ". ";
        elxout << "  Coefficents are [" << coeffs << "]." << std::endl;
        ofstream probabilityFile;
        probabilityFile.open(makeProbFileName.str());
        probabilityFile << this->GetStatisticalModelContainer()->GetElement( statisticalModelId )->ComputeLogProbabilityOfCoefficients( coeffs );
        probabilityFile.close();
      }
    }
  }

  bool writeShapeModelPrincipalComponents = false;
  this->m_Configuration->ReadParameter( writeShapeModelPrincipalComponents,
                                        "WriteShapeModelPrincipalComponentsAfterRegistration", 0, false );

  if( writeShapeModelPrincipalComponents )
  {
    for( unsigned int i = 0; i < this->GetStatisticalModelContainer()->Size(); i++ )
    {
      std::string shapeFormat = "vtk";
      this->m_Configuration->ReadParameter( shapeFormat, "ResultShapeFormat", 0, false );

      MeshFileWriterPointer meshWriter = MeshFileWriterType::New();

      for( unsigned int j = 0; j < this->GetStatisticalModelContainer()->GetElement( i )->GetNumberOfPrincipalComponents(); j++ ) {
        StatisticalModelVectorType plus3std = StatisticalModelVectorType(
                this->GetStatisticalModelContainer()->GetElement( i )->GetNumberOfPrincipalComponents(), 0.0 );
        plus3std[ j ] = 3.0;

        std::ostringstream makeFileNamePC("");
        makeFileNamePC
            << this->m_Configuration->GetCommandLineArgument("-out")
            << "Metric" << this->GetMetricNumber()
            << "ShapeModel" << i
            << "PC" << j << "." << shapeFormat;

        elxout << "  Writing shape model " << i << " principal component " << j
               << " for " << this->GetComponentLabel() << " to " << makeFileNamePC.str() << std::endl;
        meshWriter->SetInput(this->GetStatisticalModelContainer()->GetElement( i )->DrawPCABasisSample( j ));
        meshWriter->SetFileName( makeFileNamePC.str() );
        meshWriter->Update();

        std::ostringstream makeFileNameP3STD( "" );
        makeFileNameP3STD
                << this->m_Configuration->GetCommandLineArgument("-out")
                << "Metric" << this->GetMetricNumber()
                << "ShapeModel" << i
                << "PC" << j << "plus3std." << shapeFormat;

        elxout << "  Writing shape model " << i << " principal component " << j << " plus 3 standard deviations"
               << " for " << this->GetComponentLabel() << " to " << makeFileNameP3STD.str() << std::endl;
        meshWriter->SetInput(this->GetStatisticalModelContainer()->GetElement( i )->DrawSample( plus3std )) ;
        meshWriter->SetFileName( makeFileNameP3STD.str() );
        meshWriter->Update();

        StatisticalModelVectorType minus3std = StatisticalModelVectorType(
                this->GetStatisticalModelContainer()->GetElement( i )->GetNumberOfPrincipalComponents(), 0.0 );
        minus3std[ j ] = -3.0;

        std::ostringstream makeFileNamePCM3STD("");
        makeFileNamePCM3STD
                << this->m_Configuration->GetCommandLineArgument("-out")
                << "Metric" << this->GetMetricNumber()
                << "ShapeModel" << i
                << "PC" << j << "minus3std." << shapeFormat;

        elxout << "  Writing shape model " << i << " principal component " << j << " minus 3 standard deviations"
               << " for " << this->GetComponentLabel() << " to " << makeFileNamePCM3STD.str() << std::endl;
        meshWriter->SetInput(this->GetStatisticalModelContainer()->GetElement(i)->DrawSample( minus3std ));
        meshWriter->SetFileName( makeFileNamePCM3STD.str() );
        meshWriter->Update();
      }
    }
  }

  bool writeShapeModelEigenValues = false;
  this->m_Configuration->ReadParameter( writeShapeModelEigenValues,
                                        "WriteShapeModelEigenValuesAfterRegistration", 0, false );
  if( writeShapeModelEigenValues ) {
    for( unsigned int i = 0; i < this->GetStatisticalModelContainer()->Size(); i++ ) {
      std::ostringstream makeFileNameEigVal( "" );
      makeFileNameEigVal
          << this->m_Configuration->GetCommandLineArgument("-out")
          << "Metric" << this->GetMetricNumber()
          << "ShapeModel" << i
          << "EigenValues.txt";

      std::ofstream f;
      f.open(makeFileNameEigVal.str());
      f << this->GetStatisticalModelContainer()->GetElement(i)->GetPCAVarianceVector().apply(std::sqrt);
      f.close();
    }
  }

} // end AfterRegistration()

} // end namespace elastix

#endif // end #ifndef __elxActiveRegistrationModelShapeMetric_hxx__

