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
#ifndef __elxActiveRegistrationModelIntensityMetric_hxx__
#define __elxActiveRegistrationModelIntensityMetric_hxx__

#include "elxActiveRegistrationModelIntensityMetric.h"
#include "itkTimeProbe.h"
#include "itkCastImageFilter.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

  template< class TElastix >
  void
  ActiveRegistrationModelIntensityMetric< TElastix >
  ::Initialize( void )
  {
    itk::TimeProbe timer;
    timer.Start();
    this->Superclass1::Initialize();
    timer.Stop();
    elxout << "Initialization of ActiveRegistrationModelIntensityMetric metric took: "
    << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

  } // end Initialize()



/**
 * ***************** BeforeAllBase ***********************
 */

template< class TElastix >
int
ActiveRegistrationModelIntensityMetric< TElastix >
::BeforeAllBase( void )
{

  this->Superclass2::BeforeAllBase();

  std::string componentLabel( this->GetComponentLabel() );
  std::string metricNumber = componentLabel.substr( 6, 2 ); // strip "Metric" keep number
  this->SetMetricNumber( metricNumber );

  // Paths to shape models for loading
  this->m_LoadIntensityModelFileNames = ReadPath( std::string("LoadIntensityModel") );

  // Paths to shape models for saving
  this->m_SaveIntensityModelFileNames = ReadPath( std::string( "SaveIntensityModel" ) );

  // Paths to directories with images for model building
  this->m_ImageDirectories = ReadPath( std::string("BuildIntensityModel") );

  if( this->m_SaveIntensityModelFileNames.size() > 0 )
  {
    if( this->m_SaveIntensityModelFileNames.size() != this->m_ImageDirectories.size() )
    {
      itkExceptionMacro( "The number of destinations for saving intensity models must match the number of directories." );
    }
  }

  if( this->m_ImageDirectories.size() > 0 )
  {
    // Reference images for model building
    this->m_ReferenceFilenames = ReadPath( "ReferenceImage" );

    if( this->m_ReferenceFilenames.size() != this->m_ImageDirectories.size() )
    {
      itkExceptionMacro( << "The number of reference images does not match the number of directories given." );
    }
  }

  // Write reconstructed image each iteration
  std::string value = "";
  this->m_Configuration->ReadParameter( value, "WriteReconstructedImageEachIteration", 0 );
  if( value == "true" )
  {
    // this->WriteReconstructedImageEachIterationOn();
  }

  // At least one model must be specified
  if( 0 == ( this->m_LoadIntensityModelFileNames.size() + this->m_ImageDirectories.size() ) )
  {
    itkExceptionMacro( << "No statistical image model specified for " << this->GetComponentLabel() << "." << std::endl
                         << "  Specify previously built models with (LoadIntensityModel" << this->GetMetricNumber()
                         << " \"path/to/hdf5/file1\" \"path/to/hdf5/file2\" ) or " << std::endl
                         << "  specify directories with shapes using (BuildIntensityModel" << this->GetMetricNumber()
                         << " \"path/to/directory1\" \"path/to/directory2\") and " << std::endl
                         << "  corresponding reference shapes using \"(ReferenceImage" << this->GetMetricNumber()
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
ActiveRegistrationModelIntensityMetric< TElastix >
::BeforeRegistration( void )
{
  StatisticalModelContainerPointer statisticalModelContainer = StatisticalModelContainerType::New();
  statisticalModelContainer->Reserve( this->m_LoadIntensityModelFileNames.size() + this->m_ImageDirectories.size() );

  // Load models
  if( this->m_LoadIntensityModelFileNames.size() > 0 )
  {
    elxout << std::endl << "Loading models for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << " ... " << std::endl;

    for( StatisticalModelIdType statisticalModelId = 0; statisticalModelId < this->m_LoadIntensityModelFileNames.size(); ++statisticalModelId )
    {
      // Load model
      StatisticalModelPointer statisticalModel;
      try
      {
        StatisticalModelRepresenterPointer representer = StatisticalModelRepresenterType::New();
        statisticalModel = itk::StatismoIO< StatisticalModelImageType > ::LoadStatisticalModel( representer.GetPointer(), this->m_LoadIntensityModelFileNames[ statisticalModelId ] );
        statisticalModelContainer->SetElement( statisticalModelId, statisticalModel );
      }
      catch( statismo::StatisticalModelException &e )
      {
        itkExceptionMacro( "Error loading statistical shape model: " << e.what() );
      }

      elxout << "  Loaded model " << this->m_LoadIntensityModelFileNames[ statisticalModelId ].c_str() << "." << std::endl
             << "  Number modes: " << statisticalModel->GetNumberOfPrincipalComponents() << "." << std::endl
             << "  Variance: " << statisticalModel->GetPCAVarianceVector() << "." << std::endl
             << "  Noise variance: " << statisticalModel->GetNoiseVariance() << "." << std::endl;
    }
  }

  // Build models
  if( this->m_ImageDirectories.size() )
  {
    elxout << std::endl << "Building models for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << " ... " << std::endl;

    // Noise parameter for probabilistic pca model
    StatisticalModelVectorType noiseVariance = this->ReadNoiseVariance();

    // Number of principal components to keep by variance
    StatisticalModelVectorType totalVariance = this->ReadTotalVariance();

    // Loop over all data directories
    for( StatisticalModelIdType statisticalModelId = 0; statisticalModelId < this->m_ImageDirectories.size(); ++statisticalModelId )
    {
      // Load data
      StatisticalModelDataManagerPointer dataManager;
      try
      {
        dataManager = this->ReadImagesFromDirectory( this->m_ImageDirectories[ statisticalModelId ], this->m_ReferenceFilenames[ statisticalModelId ] );
      }
      catch( statismo::StatisticalModelException &e )
      {
        itkExceptionMacro( "Error loading samples in " << this->m_ImageDirectories[ statisticalModelId ] << ": " << e.what() );
      }

      // Build model
      elxout << "  Building statistical intensity model for metric " << this->GetMetricNumber() << " ... ";
      StatisticalModelPointer statisticalModel;
      try
      {
        StatisticalModelBuilderPointer pcaModelBuilder = StatisticalModelBuilderType::New();
        statisticalModel = pcaModelBuilder->BuildNewModel( dataManager->GetData(), noiseVariance[ statisticalModelId ] );
        elxout << " Done." << std::endl
               << "  Number of modes: " << statisticalModel->GetNumberOfPrincipalComponents() << "." << std::endl
               << "  Variance: " << statisticalModel->GetPCAVarianceVector()
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
      catch( statismo::StatisticalModelException &e )
      {
        itkExceptionMacro( << "Error building statistical shape model: " << e.what() );
      }

      if( this->m_SaveIntensityModelFileNames.size() > 0 )
      {
        elxout << "  Saving intensity model " << statisticalModelId << " to " << this->m_SaveIntensityModelFileNames[ statisticalModelId ] << ". " << std::endl;
        try
        {
          itk::StatismoIO< StatisticalModelImageType >::SaveStatisticalModel(statisticalModel, this->m_SaveIntensityModelFileNames[ statisticalModelId ]);
        }
        catch( statismo::StatisticalModelException& e )
        {
          itkExceptionMacro( "Could not save shape model to " << this->m_SaveIntensityModelFileNames[ statisticalModelId ] << ".");
        }
      }

      statisticalModelContainer->SetElement( statisticalModelId, statisticalModel );
    }
  }

  this->SetStatisticalModelContainer( statisticalModelContainer );

  std::cout << std::endl;
} // end BeforeRegistration()



/**
 * ***************** ReadPath ***********************
 */

template< class TElastix >
typename ActiveRegistrationModelIntensityMetric< TElastix >::StatisticalModelPathVectorType
ActiveRegistrationModelIntensityMetric< TElastix >
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
typename ActiveRegistrationModelIntensityMetric< TElastix >::StatisticalModelVectorType
ActiveRegistrationModelIntensityMetric< TElastix >
::ReadNoiseVariance()
{
  std::ostringstream key( "NoiseVariance", std::ios_base::ate );
  key << this->GetMetricNumber();

  StatisticalModelVectorType noiseVarianceVector = StatisticalModelVectorType( this->m_ImageDirectories.size(), 0.0 );
  unsigned int n = this->GetConfiguration()->CountNumberOfParameterEntries( key.str() );

  if( n == 0 )
  {
    elxout << "WARNING: NoiseVariance not specified for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << "." << std::endl
    << "  A default value of " << noiseVarianceVector[ 0 ] << " will be used (non-probabilistic PCA)." << std::endl;

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

    elxout << "  " << key.str() << ": " << noiseVariance << std::endl;

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
typename ActiveRegistrationModelIntensityMetric< TElastix >::StatisticalModelVectorType
ActiveRegistrationModelIntensityMetric< TElastix >
::ReadTotalVariance()
{
  std::ostringstream key( "TotalVariance", std::ios_base::ate );
  key << this->GetMetricNumber();

  StatisticalModelVectorType totalVarianceVector = StatisticalModelVectorType( this->m_ImageDirectories.size(), 1.0 );
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
 * ***************** LoadImagesFromDirectory ***********************
 */

template< class TElastix >
typename ActiveRegistrationModelIntensityMetric< TElastix >::StatisticalModelDataManagerPointer
ActiveRegistrationModelIntensityMetric< TElastix >
::ReadImagesFromDirectory(
  std::string imageDataDirectory,
  std::string referenceFilename )
{

  itk::Directory::Pointer directory = itk::Directory::New();
  if( !directory->Load( imageDataDirectory.c_str() ) )
  {
    itkExceptionMacro( "No files found in " << imageDataDirectory << ".");
  }

  // Read reference image
  StatisticalModelImagePointer reference = StatisticalModelImageType::New();
  if( !ReadImage( referenceFilename, reference ) )
  {
    itkExceptionMacro( "Failed to read reference file " << referenceFilename << ".");
  }

  StatisticalModelRepresenterPointer representer = StatisticalModelRepresenterType::New();
  representer->SetReference( reference );

  StatisticalModelDataManagerPointer dataManager = StatisticalModelDataManagerType::New();
  dataManager->SetRepresenter( representer.GetPointer() );

  for( unsigned int i = 0; i < directory->GetNumberOfFiles(); ++i )
  {
    const char* filename = directory->GetFile( i );
    if( std::strcmp( filename, referenceFilename.c_str() ) == 0 || std::strcmp( filename, "." ) == 0 || std::strcmp( filename, ".." ) == 0 )
    {
      continue;
    }

    std::string fullpath = imageDataDirectory + "/" + filename;
    StatisticalModelImagePointer image = StatisticalModelImageType::New();

    if( this->ReadImage( fullpath.c_str(), image ) )
    {
      dataManager->AddDataset( image, fullpath.c_str() );
    }
  }

  return dataManager;
}



/**
 * ************** ReadImage *********************
 */

template< class TElastix >
bool
ActiveRegistrationModelIntensityMetric< TElastix >
::ReadImage(
  const std::string& imageFilename,
  StatisticalModelImagePointer& image )
{
  // Read the input mesh. */
  ImageReaderPointer imageReader = ImageReaderType::New();
  imageReader->SetFileName( imageFilename.c_str() );

  elxout << "  Reading input image: " << imageFilename << " ... ";
  try
  {
    imageReader->UpdateLargestPossibleRegion();
    elxout << "done." << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    elxout << " skipping " << imageFilename << "(not a valid image file or file does not exist)." << std::endl;
    return false;
  }

  image = imageReader->GetOutput();
  return true;

} // end ReadImage()



/**
 * ***************** AfterEachIteration ***********************
 */

template< class TElastix >
void
ActiveRegistrationModelIntensityMetric< TElastix >
::AfterEachIteration( void )
{
  const unsigned int iter = this->m_Elastix->GetIterationCounter();

  /** Decide whether or not to write final model image */
  bool writeIntensityModelReconstructionAfterEachIteration = false;
  this->m_Configuration->ReadParameter( writeIntensityModelReconstructionAfterEachIteration,
                                        "WriteIntensityModelReconstructionAfterEachIteration", 0, false );


  if( writeIntensityModelReconstructionAfterEachIteration ) {
    this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->Update();
    for( unsigned int statisticalModelId = 0; statisticalModelId < this->GetStatisticalModelContainer()->Size(); statisticalModelId++ ) {
      StatisticalModelVectorType coeffs = this->GetStatisticalModelContainer()->GetElement( statisticalModelId )
              ->ComputeCoefficients(this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->GetOutput());


      std::string imageFormat = "nii.gz";
      this->m_Configuration->ReadParameter(imageFormat, "ResultImageFormat", 0, false);

      std::ostringstream makeFileName("");
      makeFileName
              << this->m_Configuration->GetCommandLineArgument("-out")
              << "Metric" << this->GetMetricNumber()
              << "IntensityModel" << statisticalModelId
              << "Iteration" << iter
              << "Image." << imageFormat;

      elxout << "  Writing intensity model " << statisticalModelId << " image for "
             << this->GetComponentLabel() << " after iteration " << iter << " to " << makeFileName.str() << ". ";
      elxout << " Coefficents are [" << coeffs << "]." << std::endl;

      MovingImageFileWriterPointer imageWriter = MovingImageFileWriterType::New();
      imageWriter->SetInput(this->GetStatisticalModelContainer()->GetElement(statisticalModelId)->DrawSample(coeffs));
      imageWriter->SetFileName(makeFileName.str());
      imageWriter->Update();
    }
  }
}



/**
 * ***************** AfterEachResolution ***********************
 */

template< class TElastix >
void
ActiveRegistrationModelIntensityMetric< TElastix >
::AfterEachResolution( void )
{
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Decide whether or not to write model image after each resolution */
  bool writeIntensityModelReconstructionAfterEachResolution = false;
  this->m_Configuration->ReadParameter( writeIntensityModelReconstructionAfterEachResolution,
                                        "WriteIntensityModelReconstructionAfterEachResolution", 0, false );

  if( writeIntensityModelReconstructionAfterEachResolution ) {
    this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->Update();

    for( unsigned int statisticalModelId = 0; statisticalModelId < this->GetStatisticalModelContainer()->Size(); statisticalModelId++ ) {
      StatisticalModelVectorType coeffs = this->GetStatisticalModelContainer()->GetElement( statisticalModelId )
              ->ComputeCoefficients(this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->GetOutput());


      std::string imageFormat = "nii.gz";
      this->m_Configuration->ReadParameter(imageFormat, "ResultImageFormat", 0, false);

      std::ostringstream makeFileName("");
      makeFileName
              << this->m_Configuration->GetCommandLineArgument("-out")
              << "Metric" << this->GetMetricNumber()
              << "IntensityModel" << statisticalModelId
              << "Resolution" << level
              << "Image." << imageFormat;

      elxout << "  Writing intensity model " << statisticalModelId << " image " << " for "
             << this->GetComponentLabel() << " after resolution " << level << " to " << makeFileName.str() << ". ";
      elxout << " Coefficents are [" << coeffs << "]." << std::endl;

      MovingImageFileWriterPointer imageWriter = MovingImageFileWriterType::New();
      imageWriter->SetInput(this->GetStatisticalModelContainer()->ElementAt( statisticalModelId )->DrawSample(coeffs));
      imageWriter->SetFileName(makeFileName.str());
      imageWriter->Update();
    }
  }
} // end AfterEachResolution()



/**
 * ***************** AfterRegistration ***********************
 */

template< class TElastix >
void
ActiveRegistrationModelIntensityMetric< TElastix >
::AfterRegistration( void )
{
  /** Decide whether or not to write the mean images */
  bool writeIntensityModelMeanImage = false;
  this->m_Configuration->ReadParameter( writeIntensityModelMeanImage,
    "WriteIntensityModelMeanImageAfterRegistration", 0, false );

  if( writeIntensityModelMeanImage )
  {
    for( unsigned int statisticalModelId = 0; statisticalModelId < this->GetStatisticalModelContainer()->Size(); statisticalModelId++ )
    {
      std::string meanImageFormat = "nii.gz";
      this->m_Configuration->ReadParameter( meanImageFormat, "ResultImageFormat", 0, false );

      std::ostringstream makeFileName( "" );
      makeFileName
        << this->m_Configuration->GetCommandLineArgument( "-out" )
        << "Metric" << this->GetMetricNumber()
        << "IntensityModel" << statisticalModelId
        << "MeanImage." << meanImageFormat;

      elxout << "  Writing intensity model " << statisticalModelId << " mean image for " << this->GetComponentLabel() << " to "
             << makeFileName.str() << std::endl;

      FixedImageFileWriterPointer imageWriter = FixedImageFileWriterType::New();
      imageWriter->SetInput( this->GetStatisticalModelContainer()->ElementAt( statisticalModelId )->DrawMean() );
      imageWriter->SetFileName( makeFileName.str() );
      imageWriter->Update();
    }
  }

  /** Decide whether or not to write final model image */
  bool writeIntensityModelFinalReconstruction = false;
  this->m_Configuration->ReadParameter( writeIntensityModelFinalReconstruction,
                                        "WriteIntensityModelFinalReconstructionAfterRegistration", 0, false );

  /** Decide whether or not to write sample probability */
  bool writeIntensityModelFinalReconstructionProbability = false;
  this->m_Configuration->ReadParameter( writeIntensityModelFinalReconstructionProbability,
                                        "WriteIntensityModelFinalReconstructionProbabilityAfterRegistration", 0, false );

  if( writeIntensityModelFinalReconstruction || writeIntensityModelFinalReconstructionProbability )
  {
    this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->Update();
    for( unsigned int statisticalModelId = 0; statisticalModelId < this->GetStatisticalModelContainer()->Size(); statisticalModelId++ ) {
      StatisticalModelVectorType coeffs = this->GetStatisticalModelContainer()->GetElement(
              statisticalModelId )->ComputeCoefficients( this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->GetOutput());

      if( writeIntensityModelFinalReconstruction ) {
        std::string imageFormat = "nii.gz";
        this->m_Configuration->ReadParameter(imageFormat, "ResultImageFormat", 0, false);

        std::ostringstream makeFileName("");
        makeFileName
                << this->m_Configuration->GetCommandLineArgument("-out")
                << "Metric" << this->GetMetricNumber()
                << "IntensityModel" << statisticalModelId
                << "FinalImage." << imageFormat;

        elxout << "  Writing intensity model " << statisticalModelId << " final image for " << this->GetComponentLabel() << " to " <<
               makeFileName.str() << ".";
        elxout << " Coefficents are [" << coeffs << "]." << std::endl;

        MovingImageFileWriterPointer imageWriter = MovingImageFileWriterType::New();
        imageWriter->SetInput(this->GetStatisticalModelContainer()->ElementAt(statisticalModelId)->DrawSample(coeffs));
        imageWriter->SetFileName(makeFileName.str());
        imageWriter->Update();
      }

      if( writeIntensityModelFinalReconstructionProbability ) {
        std::ostringstream makeProbFileName;
        makeProbFileName
        << this->m_Configuration->GetCommandLineArgument("-out")
        << "Metric" << this->GetMetricNumber()
        << "IntensityModel" << statisticalModelId
        << "Probability.txt";

        elxout << "  Writing intensity model " << statisticalModelId << " final image probablity for " << this->GetComponentLabel()
               << " to " << makeProbFileName.str() << ". ";
        elxout << "  Coefficents are [" << coeffs << "]." << std::endl;
        ofstream probabilityFile;
        probabilityFile.open(makeProbFileName.str());
        probabilityFile <<
        this->GetStatisticalModelContainer()->ElementAt( statisticalModelId )->ComputeLogProbabilityOfCoefficients(coeffs);
        probabilityFile.close();
      }
    }
  }

  bool writeIntensityModelPrincipalComponents = false;
  this->m_Configuration->ReadParameter( writeIntensityModelPrincipalComponents,
                                        "WriteIntensityModelPrincipalComponentsAfterRegistration", 0, false );

  if( writeIntensityModelPrincipalComponents )
  {
    for( unsigned int statisticalModelId = 0; statisticalModelId < this->GetStatisticalModelContainer()->Size(); statisticalModelId++ )
    {
      std::string imageFormat = "nii.gz";
      this->m_Configuration->ReadParameter( imageFormat, "ResultImageFormat", 0, false );

      MovingImageFileWriterPointer imageWriter = MovingImageFileWriterType::New();

      for( unsigned int j = 0; j < this->GetStatisticalModelContainer()->GetElement( statisticalModelId )->GetNumberOfPrincipalComponents(); j++ ) {
        StatisticalModelVectorType plus3std = StatisticalModelVectorType(
                this->GetStatisticalModelContainer()->GetElement( statisticalModelId )->GetNumberOfPrincipalComponents(), 0.0 );
        plus3std[ j ] = 3.0;

        std::ostringstream makeFileNameP3STD( "" );
        makeFileNameP3STD
                << this->m_Configuration->GetCommandLineArgument("-out")
                << "Metric" << this->GetMetricNumber()
                << "IntensityModel" << statisticalModelId
                << "PC" << j << "plus3std." << imageFormat;

        elxout << "  Writing intensity model " << statisticalModelId << " principal component " << j << " plus 3 standard deviations"
               << " for " << this->GetComponentLabel() << " to " << makeFileNameP3STD.str() << std::endl;
        imageWriter->SetInput(this->GetStatisticalModelContainer()->GetElement( statisticalModelId )->DrawSample( plus3std ) ) ;
        imageWriter->SetFileName( makeFileNameP3STD.str() );
        imageWriter->Update();

        StatisticalModelVectorType minus3std = StatisticalModelVectorType(
                this->GetStatisticalModelContainer()->GetElement( statisticalModelId )->GetNumberOfPrincipalComponents(), 0.0 );
        minus3std[ j ] = -3.0;

        std::ostringstream makeFileNamePCM3STD("");
        makeFileNamePCM3STD
                << this->m_Configuration->GetCommandLineArgument("-out")
                << "Metric" << this->GetMetricNumber()
                << "IntensityModel" << statisticalModelId
                << "PC" << j << "minus3std." << imageFormat;

        elxout << "  Writing intensity model " << statisticalModelId << " principal component " << j << " minus 3 standard deviations"
               << " for " << this->GetComponentLabel() << " to " << makeFileNamePCM3STD.str() << std::endl;
        imageWriter->SetInput(this->GetStatisticalModelContainer()->GetElement(statisticalModelId)->DrawSample( minus3std ) );
        imageWriter->SetFileName( makeFileNamePCM3STD.str() );
        imageWriter->Update();
      }
    }
  }
} // end AfterRegistration()


} // end namespace elastix

#endif // end #ifndef __elxActiveRegistrationModelIntensityMetric_hxx__
