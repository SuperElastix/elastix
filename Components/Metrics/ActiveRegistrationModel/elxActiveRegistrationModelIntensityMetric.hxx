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

  StatisticalModelVectorContainerPointer statisticalModelMeanVectorContainer = StatisticalModelVectorContainerType::New();
  statisticalModelMeanVectorContainer->Reserve( this->m_LoadIntensityModelFileNames.size() + this->m_ImageDirectories.size() );

  StatisticalModelMatrixContainerPointer statisticalModelOrthonormalPCABasisMatrixContainer = StatisticalModelMatrixContainerType::New();
  statisticalModelOrthonormalPCABasisMatrixContainer->Reserve( this->m_LoadIntensityModelFileNames.size() + this->m_ImageDirectories.size() );

  StatisticalModelVectorContainerPointer statisticalModelVarianceVectorContainer = StatisticalModelVectorContainerType::New();
  statisticalModelVarianceVectorContainer->Reserve( this->m_LoadIntensityModelFileNames.size() + this->m_ImageDirectories.size() );

  StatisticalModelScalarContainerPointer statisticalModelNoiseVarianceContainer = StatisticalModelScalarContainerType::New();
  statisticalModelNoiseVarianceContainer->Reserve( this->m_LoadIntensityModelFileNames.size() + this->m_ImageDirectories.size() );

  StatisticalModelRepresenterContainerPointer statisticalModelRepresenterContainer = StatisticalModelRepresenterContainerType::New();
  statisticalModelRepresenterContainer->Reserve( this->m_LoadIntensityModelFileNames.size() + this->m_ImageDirectories.size() );

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
        statisticalModelMeanVectorContainer->SetElement( statisticalModelId, statisticalModel->GetMeanVector() );
        statisticalModelOrthonormalPCABasisMatrixContainer->SetElement( statisticalModelId, statisticalModel->GetOrthonormalPCABasisMatrix() );
        statisticalModelVarianceVectorContainer->SetElement( statisticalModelId, statisticalModel->GetPCAVarianceVector() );
        statisticalModelNoiseVarianceContainer->SetElement( statisticalModelId, statisticalModel->GetNoiseVariance() );
        statisticalModelRepresenterContainer->SetElement( statisticalModelId, representer );
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
        itkExceptionMacro( "Error loading samples in " << this->m_ImageDirectories[ statisticalModelId ] <<": " << e.what() );
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
          ReducedVarianceModelBuilderPointer reducedVarianceModelBuilder = ReducedVarianceModelBuilderType::New();
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

      statisticalModelMeanVectorContainer->SetElement( statisticalModelId, statisticalModel->GetMeanVector());
      statisticalModelOrthonormalPCABasisMatrixContainer->SetElement( statisticalModelId, statisticalModel->GetOrthonormalPCABasisMatrix());
      statisticalModelVarianceVectorContainer->SetElement( statisticalModelId, statisticalModel->GetPCAVarianceVector() );
      statisticalModelNoiseVarianceContainer->SetElement( statisticalModelId, noiseVariance[ statisticalModelId ]);

      StatisticalModelRepresenterPointer representer = StatisticalModelRepresenterType::New();
      representer->SetReference( statisticalModel->GetRepresenter()->GetReference() );
      statisticalModelRepresenterContainer->SetElement( statisticalModelId, representer );
    }
  }

  this->SetMeanVectorContainer( statisticalModelMeanVectorContainer );
  this->SetBasisMatrixContainer( statisticalModelOrthonormalPCABasisMatrixContainer) ;
  this->SetVarianceContainer( statisticalModelVarianceVectorContainer );
  this->SetNoiseVarianceContainer( statisticalModelNoiseVarianceContainer) ;
  this->SetRepresenterContainer( statisticalModelRepresenterContainer );

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


template< class TElastix >
void
ActiveRegistrationModelIntensityMetric< TElastix >
::AfterEachIteration( void )
{
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();
  const unsigned int iter = this->m_Elastix->GetIterationCounter();

  /** Decide whether or not to write final model image */
  bool writeIntensityModelImageAfterEachIteration = false;
  this->m_Configuration->ReadParameter( writeIntensityModelImageAfterEachIteration,
                                        "WriteIntensityModelImageAfterEachIteration", "", level, 0, false );

//  if( writeIntensityModelImageAfterEachIteration ) {
//    for (unsigned int i = 0; i < this->GetStatisticalModelContainer()->Size(); i++) {
//      // Compute coefficients for non-zero model intensities. This assumes that the model was built on masked images
//      typename RepresenterType::DatasetConstPointerType reference = this->GetStatisticalModelContainer()->GetElement(
//              i)->GetRepresenter()->GetReference();
//
//      this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->GetOutput()->Update();
//      FixedImageConstPointer resultImage = this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->GetOutput();
//
//      typename StatisticalModelType::DomainType::DomainPointsListType domain = this->GetStatisticalModelContainer()->GetElement(
//              i)->GetRepresenter()->GetDomain().GetDomainPoints();
//      StatisticalModelPointValueListType pointsInsideDomain = StatisticalModelPointValueListType();
//      for (unsigned int j = 0; j < domain.size(); j++) {
//        MovingImagePointType point = domain[j];
//        typename MovingImageType::IndexType index;
//        reference->TransformPhysicalPointToIndex(point, index);
//        MovingImagePixelType intensity = reference->GetPixel(index);
//        if (intensity > 0.0) {
//          intensity = resultImage->GetPixel(index);
//          pointsInsideDomain.push_back(StatisticalModelPointValuePairType(point, intensity));
//        }
//      }
//
//      StatisticalModelVectorType modelCoefficients = this->GetStatisticalModelContainer()->GetElement(
//              i)->ComputeCoefficientsForPointValues(pointsInsideDomain, false);
//      elxout << pointsInsideDomain.size() << "/" << domain.size() << " points used for final model coefficients. ";
//      elxout << " Coefficents are [" << modelCoefficients << "]." << std::endl;
//
//      std::string imageFormat = "png";
//
//      std::ostringstream makeFileName("");
//      makeFileName
//      << this->m_Configuration->GetCommandLineArgument("-out")
//      << "IntensityModelImage" << i
//      << "Metric" << this->GetMetricNumber()
//      << ".R" << level
//      << ".It" << std::setfill( '0' ) << std::setw( 7 ) << iter
//      << "." << imageFormat;
//
//      elxout << "  Writing statistical model final image " << i << " for " << this->GetComponentLabel() << " to " <<
//      makeFileName.str() << ". " << std::endl;
//
//      typedef itk::Image< unsigned char, 2 > PNGImageType;
//      typedef itk::CastImageFilter< FixedImageType, PNGImageType > CastImageFilterType;
//
//      typename CastImageFilterType::Pointer castImageFilter = CastImageFilterType::New();
//      castImageFilter->SetInput(this->GetStatisticalModelContainer()->GetElement(i)->DrawSample(modelCoefficients));
//
//      typedef itk::ImageFileWriter< PNGImageType > PNGImageWriterType;
//      typename PNGImageWriterType::Pointer imageWriter = PNGImageWriterType::New();
//      imageWriter->SetInput(castImageFilter->GetOutput());
//      imageWriter->SetFileName(makeFileName.str());
//      imageWriter->Update();
//
//    }
//  }
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
  bool writeIntensityModelImageAfterEachResolution = false;
  this->m_Configuration->ReadParameter( writeIntensityModelImageAfterEachResolution,
                                        "WriteIntensityModelImageAfterEachResolution", "", level, 0, false );

//  if( writeIntensityModelImageAfterEachResolution )
//  {
//    for( unsigned int i = 0; i < this->GetStatisticalModelContainer()->Size(); i++ )
//    {
//      std::string meanImageFormat = "nii.gz";
//      this->m_Configuration->ReadParameter( meanImageFormat, "ResultImageFormat", 0, false );
//
//      std::ostringstream makeFileName( "" );
//      makeFileName
//      << this->m_Configuration->GetCommandLineArgument( "-out" )
//      << "IntensityModel" << i
//      << "Metric" << this->GetMetricNumber()
//      << "Resolution " << level
//      << "Image." << meanImageFormat;
//
//      elxout << "  Writing statistical model final image " << i << " for " << this->GetComponentLabel() << " to " << makeFileName.str() << ". ";
//
//      this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->Update();
//      StatisticalModelVectorType modelCoefficients = this->GetStatisticalModelContainer()->GetElement( i )->ComputeCoefficients( this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->GetOutput() );
//
//      elxout << " Coefficents are [" << modelCoefficients << "]." << std::endl;
//
//      MovingImageFileWriterPointer imageWriter = MovingImageFileWriterType::New();
//      imageWriter->SetInput( this->GetStatisticalModelContainer()->GetElement( i )->DrawSample( modelCoefficients ) );
//      imageWriter->SetFileName( makeFileName.str() );
//      imageWriter->Update();
//    }
//  }
} // end AfterEachResolution()



/**
 * ***************** AfterRegistration ***********************
 */

template< class TElastix >
void
ActiveRegistrationModelIntensityMetric< TElastix >
::AfterRegistration( void )
{
//  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();
//
//  /** Decide whether or not to write the mean images */
//  bool writeIntensityModelMeanImage = false;
//  this->m_Configuration->ReadParameter( writeIntensityModelMeanImage,
//    "WriteIntensityModelMeanImageAfterRegistration", "", level, 0, false );
//
//  if( writeIntensityModelMeanImage )
//  {
//    for( unsigned int i = 0; i < this->GetStatisticalModelContainer()->Size(); i++ )
//    {
//      std::string meanImageFormat = "nii.gz";
//      this->m_Configuration->ReadParameter( meanImageFormat, "ResultImageFormat", 0, false );
//
//      std::ostringstream makeFileName( "" );
//      makeFileName
//        << this->m_Configuration->GetCommandLineArgument( "-out" )
//        << "IntensityModel" << i
//        << "Metric" << this->GetMetricNumber()
//        << "MeanImage." << meanImageFormat;
//
//      elxout << "  Writing statistical model mean image " << i << " for " << this->GetComponentLabel() << " to " << makeFileName.str() << std::endl;
//
//      FixedImageFileWriterPointer imageWriter = FixedImageFileWriterType::New();
//      imageWriter->SetInput( this->GetStatisticalModelContainer()->GetElement( i )->DrawMean() );
//      imageWriter->SetFileName( makeFileName.str() );
//      imageWriter->Update();
//    }
//  }
//
//  /** Decide whether or not to write sample probability */
//  bool writeIntensityModelProbability = false;
//  this->m_Configuration->ReadParameter( writeIntensityModelProbability,
//                                        "WriteIntensityModelProbabilityAfterRegistration", "", level, 0, false );
//
//  /** Decide whether or not to write sample cosine */
//  bool writeIntensityModelCosineSimilarity = false;
//  this->m_Configuration->ReadParameter( writeIntensityModelCosineSimilarity,
//                                        "WriteIntensityModelCosineSimilarityAfterRegistration", "", level, 0, false );
//
//  /** Decide whether or not to write sample probability */
//  bool writeIntensityModelProjectionMagnitude = false;
//  this->m_Configuration->ReadParameter( writeIntensityModelProjectionMagnitude,
//                                        "WriteIntensityModelProjectionMagnitudeAfterRegistration", "", level, 0, false );
//
//
//  /** Decide whether or not to write final model image */
//  bool writeIntensityModelFinalImage = false;
//  this->m_Configuration->ReadParameter( writeIntensityModelFinalImage,
//                                        "WriteIntensityModelFinalImageAfterRegistration", "", level, 0, false );
//
//  if( writeIntensityModelFinalImage || writeIntensityModelProbability || writeIntensityModelCosineSimilarity || writeIntensityModelProjectionMagnitude )
//  {
//    for( unsigned int i = 0; i < this->GetStatisticalModelContainer()->Size(); i++ ) {
//      // Compute coefficients for non-zero model intensities. This assumes that the model was built on masked images
//      typename RepresenterType::DatasetConstPointerType reference = this->GetStatisticalModelContainer()->GetElement(
//              i)->GetRepresenter()->GetReference();
//      FixedImageConstPointer resultImage = this->GetElastix()->GetElxResamplerBase()->GetAsITKBaseType()->GetOutput();
//
//      typename StatisticalModelType::DomainType::DomainPointsListType domain = this->GetStatisticalModelContainer()->GetElement(
//              i)->GetRepresenter()->GetDomain().GetDomainPoints();
//      StatisticalModelPointValueListType pointsInsideDomain = StatisticalModelPointValueListType();
//      for (unsigned int j = 0; j < domain.size(); j++) {
//        MovingImagePointType point = domain[j];
//        typename MovingImageType::IndexType index;
//        reference->TransformPhysicalPointToIndex(point, index);
//        MovingImagePixelType intensity = reference->GetPixel(index);
//        if (intensity > 0.0) {
//          intensity = resultImage->GetPixel(index);
//          pointsInsideDomain.push_back(StatisticalModelPointValuePairType(point, intensity));
//        }
//      }
//
//      StatisticalModelVectorType modelCoefficients = this->GetStatisticalModelContainer()->GetElement(
//              i)->ComputeCoefficientsForPointValues(pointsInsideDomain, false);
//      elxout << pointsInsideDomain.size() << "/" << domain.size() << " points used for final model coefficients. ";
//      elxout << " Coefficents are [" << modelCoefficients << "]." << std::endl;
//
//      if (writeIntensityModelFinalImage) {
//        std::string meanImageFormat = "nii.gz";
//        this->m_Configuration->ReadParameter(meanImageFormat, "ResultImageFormat", 0, false);
//
//        std::ostringstream makeFileName("");
//        makeFileName
//        << this->m_Configuration->GetCommandLineArgument("-out")
//        << "IntensityModel" << i
//        << "Metric" << this->GetMetricNumber()
//        << "FinalImage." << meanImageFormat;
//
//        elxout << "  Writing statistical model final image " << i << " for " << this->GetComponentLabel() << " to " <<
//        makeFileName.str() << ". ";
//
//        MovingImageFileWriterPointer imageWriter = MovingImageFileWriterType::New();
//        imageWriter->SetInput(this->GetStatisticalModelContainer()->GetElement(i)->DrawSample(modelCoefficients));
//        imageWriter->SetFileName(makeFileName.str());
//        imageWriter->Update();
//      }
//
//      if (writeIntensityModelProbability) {
//        std::ostringstream makeProbFileName;
//        makeProbFileName
//        << this->m_Configuration->GetCommandLineArgument("-out")
//        << "IntensityModel" << i
//        << "Metric" << this->GetMetricNumber()
//        << "Probability.txt";
//        ofstream probabilityFile;
//        probabilityFile.open(makeProbFileName.str());
//        probabilityFile <<
//        this->GetStatisticalModelContainer()->GetElement(i)->ComputeLogProbabilityOfCoefficients(modelCoefficients);
//        probabilityFile.close();
//      }
//
//      if (writeIntensityModelCosineSimilarity) {
//        std::ostringstream makeFractionFileName;
//        makeFractionFileName
//        << this->m_Configuration->GetCommandLineArgument("-out")
//        << "IntensityModel" << i
//        << "Metric" << this->GetMetricNumber()
//        << "CosineSimilarity.txt";
//
//        StatisticalModelImagePointer reconstructedImage = this->GetStatisticalModelContainer()->GetElement(
//                i)->DrawSample(modelCoefficients);
//        double reconstructedImageSumOfSquares = 0.0;
//        double originalImageSumOfSquares = 0.0;
//
//        double dotProduct = 0.0;
//        for (unsigned int j = 0; j < domain.size(); j++) {
//          MovingImagePointType point = domain[j];
//          typename MovingImageType::IndexType index;
//          reference->TransformPhysicalPointToIndex(point, index);
//          MovingImagePixelType intensity = reference->GetPixel(index);
//          if (intensity > 0.0) {
//            typename MovingImageType::IndexType reconstructedImageIndex;
//            reconstructedImage->TransformPhysicalPointToIndex(point, reconstructedImageIndex);
//            MovingImagePixelType reconstructedImageIntensity = reconstructedImage->GetPixel(index);
//            reconstructedImageSumOfSquares += reconstructedImageIntensity * reconstructedImageIntensity;
//            typename MovingImageType::IndexType resultImageIndex;
//            resultImage->TransformPhysicalPointToIndex(point, resultImageIndex);
//            MovingImagePixelType resultImageIntensity = resultImage->GetPixel(resultImageIndex);
//            originalImageSumOfSquares += resultImageIntensity * resultImageIntensity;
//
//            dotProduct += reconstructedImageIntensity * resultImageIntensity;
//          }
//        }
//
//        double cosine = dotProduct / (sqrt(reconstructedImageSumOfSquares) * sqrt(originalImageSumOfSquares));
//        ofstream cosineFile;
//        cosineFile.open(makeFractionFileName.str());
//        cosineFile << cosine;
//        cosineFile.close();
//      }
//
//      if (writeIntensityModelProjectionMagnitude) {
//        std::ostringstream makeFractionFileName;
//        makeFractionFileName
//        << this->m_Configuration->GetCommandLineArgument("-out")
//        << "IntensityModel" << i
//        << "Metric" << this->GetMetricNumber()
//        << "ProjectionMagnitude.txt";
//
//        StatisticalModelImagePointer reconstructedImage = this->GetStatisticalModelContainer()->GetElement(i)->DrawSample(modelCoefficients);
//        double reconstructedImageSumOfSquares = 0.0;
//        double originalImageSumOfSquares = 0.0;
//
//        int n = 0;
//        for (unsigned int j = 0; j < domain.size(); j++) {
//          MovingImagePointType point = domain[j];
//          typename MovingImageType::IndexType index;
//          reference->TransformPhysicalPointToIndex(point, index);
//          MovingImagePixelType intensity = reference->GetPixel(index);
//          if (intensity > 0.0) {
//            n++;
//          }
//        }
//
//        StatisticalModelVectorType resultImageVector = StatisticalModelVectorType(n, 0.0);
//        StatisticalModelVectorType reconstructedImageVector = StatisticalModelVectorType(n, 0.0);
//        for (unsigned int j = 0; j < domain.size(); j++) {
//          MovingImagePointType point = domain[j];
//          typename MovingImageType::IndexType index;
//          reference->TransformPhysicalPointToIndex(point, index);
//          MovingImagePixelType intensity = reference->GetPixel(index);
//          if (intensity > 0.0) {
//            typename MovingImageType::IndexType reconstructedImageIndex;
//            reconstructedImage->TransformPhysicalPointToIndex(point, reconstructedImageIndex);
//            reconstructedImageVector[ j ] = reconstructedImage->GetPixel(index);
//
//            typename MovingImageType::IndexType resultImageIndex;
//            resultImage->TransformPhysicalPointToIndex(point, resultImageIndex);
//            resultImageVector[ j ] = resultImage->GetPixel(resultImageIndex);
//          }
//        }
//
//        double scalar = dot_product( reconstructedImageVector, resultImageVector ) / resultImageVector.magnitude();
//        StatisticalModelVectorType projection = scalar * resultImageVector;
//        double projectionMagnitude = (projection-reconstructedImageVector).magnitude();
//
//        ofstream projectionMagnitudeFile;
//        projectionMagnitudeFile.open(makeFractionFileName.str());
//        projectionMagnitudeFile << projectionMagnitude;
//        projectionMagnitudeFile.close();
//      }
//    }
//  }
//
//  bool writeIntensityModelPrincipalComponents = false;
//  this->m_Configuration->ReadParameter( writeIntensityModelPrincipalComponents,
//                                        "WriteIntensityModelPrincipalComponentsAfterRegistration", "", level, 0, false );
//
//  if( writeIntensityModelPrincipalComponents )
//  {
//    for( unsigned int i = 0; i < this->GetStatisticalModelContainer()->Size(); i++ )
//    {
//      std::string meanImageFormat = "nii.gz";
//      this->m_Configuration->ReadParameter( meanImageFormat, "ResultImageFormat", 0, false );
//
//      StatisticalModelVectorType variance = this->GetStatisticalModelContainer()->GetElement( i )->GetPCAVarianceVector();
//      MovingImageFileWriterPointer imageWriter = MovingImageFileWriterType::New();
//
//      // 1st principal component
//      StatisticalModelVectorType pc0plus3std = StatisticalModelVectorType( this->GetStatisticalModelContainer()->GetElement( i )->GetNumberOfPrincipalComponents(), 0.0 );
//      pc0plus3std[ 0 ] = 3.0;
//
//      std::ostringstream makeFileNamePC0P3STD( "" );
//      makeFileNamePC0P3STD
//      << this->m_Configuration->GetCommandLineArgument( "-out" )
//      << "IntensityModel" << i
//      << "Metric" << this->GetMetricNumber()
//      << "PC0plus3std." << meanImageFormat;
//
//      elxout << "  Writing statistical model principal component 0 plus 3 standard deviations for model " << i << " for " << this->GetComponentLabel() << " to " << makeFileNamePC0P3STD.str() << std::endl;
//      imageWriter->SetInput( this->GetStatisticalModelContainer()->GetElement( i )->DrawSample( pc0plus3std ) );
//      imageWriter->SetFileName( makeFileNamePC0P3STD.str() );
//      imageWriter->Update();
//
//      StatisticalModelVectorType pc0minus3std = StatisticalModelVectorType( this->GetStatisticalModelContainer()->GetElement( i )->GetNumberOfPrincipalComponents(), 0.0 );
//      pc0minus3std[ 0 ] = -3.0;
//
//      std::ostringstream makeFileNamePC0M3STD( "" );
//      makeFileNamePC0M3STD
//      << this->m_Configuration->GetCommandLineArgument( "-out" )
//      << "IntensityModel" << i
//      << "Metric" << this->GetMetricNumber()
//      << "PC0minus3std." << meanImageFormat;
//
//      elxout << "  Writing statistical model principal component 0 minus 3 standard deviations for model " << i << " for " << this->GetComponentLabel() << " to " << makeFileNamePC0M3STD.str() << std::endl;
//      imageWriter->SetInput( this->GetStatisticalModelContainer()->GetElement( i )->DrawSample( pc0minus3std ) );
//      imageWriter->SetFileName( makeFileNamePC0M3STD.str() );
//      imageWriter->Update();
//
//      // 2nd principal component
//      StatisticalModelVectorType pc1plus3std = StatisticalModelVectorType( this->GetStatisticalModelContainer()->GetElement( i )->GetNumberOfPrincipalComponents(), 0.0 );
//      pc1plus3std[ 1 ] = 3.0;
//
//      std::ostringstream makeFileNamePC1P3STD( "" );
//      makeFileNamePC1P3STD
//      << this->m_Configuration->GetCommandLineArgument( "-out" )
//      << "IntensityModel" << i
//      << "Metric" << this->GetMetricNumber()
//      << "PC1plus3std." << meanImageFormat;
//
//      elxout << "  Writing statistical model principal component 1 plus 3 standard deviations for model " << i << " for " << this->GetComponentLabel() << " to " << makeFileNamePC1P3STD.str() << std::endl;
//      imageWriter->SetInput( this->GetStatisticalModelContainer()->GetElement( i )->DrawSample( pc1plus3std ) );
//      imageWriter->SetFileName( makeFileNamePC1P3STD.str() );
//      imageWriter->Update();
//
//      //
//      StatisticalModelVectorType pc1minus3std = StatisticalModelVectorType( this->GetStatisticalModelContainer()->GetElement( i )->GetNumberOfPrincipalComponents(), 0.0 );
//      pc1minus3std[ 1 ] = -3.0;
//
//      std::ostringstream makeFileNamePC1M3STD( "" );
//      makeFileNamePC1M3STD
//      << this->m_Configuration->GetCommandLineArgument( "-out" )
//      << "IntensityModel" << i
//      << "Metric" << this->GetMetricNumber()
//      << "PC1minus3std." << meanImageFormat;
//
//      elxout << "  Writing statistical model principal component 1 minus 3 standard deviations for model " << i << " for " << this->GetComponentLabel() << " to " << makeFileNamePC1M3STD.str() << std::endl;
//      imageWriter->SetInput( this->GetStatisticalModelContainer()->GetElement( i )->DrawSample( pc1minus3std ) );
//      imageWriter->SetFileName( makeFileNamePC1M3STD.str() );
//      imageWriter->Update();
//
//      // 3rd principal component
//      StatisticalModelVectorType pc2plus3std = StatisticalModelVectorType( this->GetStatisticalModelContainer()->GetElement( i )->GetNumberOfPrincipalComponents(), 0.0 );
//      pc2plus3std[ 2 ] = 3.0;
//
//      std::ostringstream makeFileNamePC2P3STD( "" );
//      makeFileNamePC2P3STD
//      << this->m_Configuration->GetCommandLineArgument( "-out" )
//      << "IntensityModel" << i
//      << "Metric" << this->GetMetricNumber()
//      << "PC2plus3std." << meanImageFormat;
//
//      elxout << "  Writing statistical model principal component 2 plus 3 standard deviations for model " << i << " for " << this->GetComponentLabel() << " to " << makeFileNamePC2P3STD.str() << std::endl;
//      imageWriter->SetInput( this->GetStatisticalModelContainer()->GetElement( i )->DrawSample( pc2plus3std ) );
//      imageWriter->SetFileName( makeFileNamePC2P3STD.str() );
//      imageWriter->Update();
//
//      //
//      StatisticalModelVectorType pc2minus3std = StatisticalModelVectorType( this->GetStatisticalModelContainer()->GetElement( i )->GetNumberOfPrincipalComponents(), 0.0 );
//      pc2minus3std[ 2 ] = -3.0;
//
//      std::ostringstream makeFileNamePC2M3STD( "" );
//      makeFileNamePC2M3STD
//      << this->m_Configuration->GetCommandLineArgument( "-out" )
//      << "IntensityModel" << i
//      << "Metric" << this->GetMetricNumber()
//      << "PC2minus3std." << meanImageFormat;
//
//      elxout << "  Writing statistical model principal component 2 minus 3 standard deviations for model " << i << " for " << this->GetComponentLabel() << " to " << makeFileNamePC2M3STD.str() << std::endl;
//      imageWriter->SetInput( this->GetStatisticalModelContainer()->GetElement( i )->DrawSample( pc2minus3std ) );
//      imageWriter->SetFileName( makeFileNamePC2M3STD.str() );
//      imageWriter->Update();
//    }
//  }
} // end AfterRegistration()



///**
// * ***************** AfterRegistration ***********************
// */
//
//template< class TElastix >
//void
//ActiveRegistrationModelIntensityMetric< TElastix >
//::AfterEachIteration( void ) {
//
//  /** What is the current resolution level? */
//  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();
//
//  /** What is the current iteration number? */
//  const unsigned int iter = this->m_Elastix->GetIterationCounter();
//
//  /** Decide whether or not to write the result mesh this iteration. */
//  bool writeModelReconstructionThisIteration = false;
//  this->m_Configuration->ReadParameter( writeModelReconstructionThisIteration,
//                                        "WriteModelReconstructionAfterEachIteration", "", level, 0, false );
//
//  if( writeModelReconstructionThisIteration )
//  {
//
//    StatisticalModelPointValueListType intensityPointValueList;
//    typedef typename itk::ImageFileWriter< FixedImageType > WriterType;
//    typename WriterType::Pointer writer = WriterType::New();
//
//    // Loop over models
//    StatisticalModelContainerConstIterator statisticalModelIterator = this->GetStatisticalModelContainer()->Begin();
//    StatisticalModelContainerConstIterator statisticalModelIteratorEnd = this->GetStatisticalModelContainer()->End();
//    while( statisticalModelIterator != statisticalModelIteratorEnd )
//    {
//
//      /** Create iterator over the sample container. */
//      ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
//      typename ImageSampleContainerType::ConstIterator fixedSampleContainerIterator = sampleContainer->Begin();
//      typename ImageSampleContainerType::ConstIterator fixedSampleContainerIteratorEnd = sampleContainer->End();
//
//      while( fixedSampleContainerIterator != fixedSampleContainerIteratorEnd )
//      {
//        const FixedImagePointType& fixedPoint = fixedSampleContainerIterator->Value().m_ImageCoordinates;
//        MovingImagePointType       movingPoint;
//        RealType                   movingIntensityValue;
//        bool sampleOk = this->TransformPoint( fixedPoint, movingPoint );
//
//        if( sampleOk )
//        {
//          sampleOk = this->EvaluateMovingImageValueAndDerivative( movingPoint, movingIntensityValue, 0 );
//        }
//
//        if( sampleOk )
//        {
//          intensityPointValueList.push_back( StatisticalModelPointValuePairType( movingPoint, movingIntensityValue) );
//        }
//
//        ++fixedSampleContainerIterator;
//      }
//
//      // Write model reconstruction to disk
//      ostringstream os;
//      os << "StatisticalImageMetric" << std::setfill( '0' ) << std::setw( 7 ) << this->m_MetricNumber << ".Level" << level << ".Iteration" << this->m_Elastix->GetIterationCounter() << ".nii";
//      writer->SetFileName( os.str() );
//
//      const StatisticalModelVectorType modelCoefficients = statisticalModelIterator->Value()->ComputeCoefficientsForPointValues( intensityPointValueList, statisticalModelIterator->Value()->GetNoiseVariance() );
//      typename FixedImageType::Pointer image = statisticalModelIterator->Value()->DrawSample( modelCoefficients, false );
//      writer->SetInput( image );
//      writer->Update();
//
//      ++statisticalModelIterator;
//    }
//  }
//}

} // end namespace elastix

#endif // end #ifndef __elxActiveRegistrationModelIntensityMetric_hxx__
