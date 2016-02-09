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
#ifndef elxElastixFilter_hxx
#define elxElastixFilter_hxx

#include "itkImageDuplicator.h"

namespace elastix
{

template< typename TFixedImage, typename TMovingImage >
ElastixFilter< TFixedImage, TMovingImage >
::ElastixFilter( void )
{
  this->AddRequiredInputName( "FixedImage" );
  this->AddRequiredInputName( "MovingImage" );
  this->AddRequiredInputName( "ParameterObject" );

  this->SetPrimaryInputName( "FixedImage" );
  this->SetPrimaryOutputName( "ResultImage" );

  this->m_FixedImageContainer = DataObjectContainerType::New();
  this->m_MovingImageContainer = DataObjectContainerType::New();

  this->m_FixedPointSetFileName = "";
  this->m_MovingPointSetFileName = "";

  this->m_OutputDirectory = ".";
  this->m_LogFileName = "";

  this->LogToConsoleOff();
  this->LogToFileOff();

  ParameterObjectPointer defaultParameterObject = ParameterObject::New();
  defaultParameterObject->AddParameterMap( "rigid" );
  defaultParameterObject->AddParameterMap( "affine" );
  defaultParameterObject->AddParameterMap( "nonrigid" );
  this->SetParameterObject( defaultParameterObject );
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::GenerateData( void )
{
  // TODO: Fix elastix destroying fixed image during registration so we 
  // don't need to deep copy every fixed image before every registration
  typedef itk::ImageDuplicator< TFixedImage > DuplicatorType;
  typedef typename DuplicatorType::Pointer DuplicatorPointer;
  DuplicatorPointer duplicator = DuplicatorType::New();

  // Initialize variables here so they don't go out of scope between iterations of the main loop
  ElastixMainObjectPointer    transform            = 0;
  DataObjectContainerPointer  fixedImageContainer  = DataObjectContainerType::New();
  DataObjectContainerPointer  movingImageContainer = DataObjectContainerType::New();
  DataObjectContainerPointer  fixedMaskContainer   = 0;
  DataObjectContainerPointer  movingMaskContainer  = 0;
  DataObjectContainerPointer  resultImageContainer = 0;
  ParameterMapVectorType      transformParameterMapVector;
  FlatDirectionCosinesType    fixedImageOriginalDirection;

  // Split inputs into separate containers
  const InputNameArrayType inputNames = this->GetInputNames();
  for( unsigned int i = 0; i < inputNames.size(); ++i )
  {
    if( this->IsInputType( "FixedImage", inputNames[ i ] ) )
    { 
      duplicator->SetInputImage( static_cast< TFixedImage* >( this->ProcessObject::GetInput( inputNames[ i ] ) ) );
      duplicator->Update();

      fixedImageContainer->push_back( this->GetInput( inputNames[ i ] ) );
      continue;
    }

    if( this->IsInputType( "MovingImage", inputNames[ i ] ) )
    {
      movingImageContainer->push_back( this->GetInput( inputNames[ i ] ) );
      continue;
    }

    if( this->IsInputType( "FixedMask", inputNames[ i ] ) )
    {
      if( fixedMaskContainer.IsNull() )
      {
        fixedMaskContainer = DataObjectContainerType::New();
      }

      fixedMaskContainer->push_back( this->GetInput( inputNames[ i ] ) );
      continue;
    }

    if( this->IsInputType( "MovingMask", inputNames[ i ] ) )
    {
      if( movingMaskContainer.IsNull() )
      {
          movingMaskContainer = DataObjectContainerType::New();
      }

      movingMaskContainer->push_back( this->GetInput( inputNames[ i ] ) );
    }
  }

  // Set ParameterMap
  ParameterObjectConstPointer parameterObject = static_cast< const ParameterObject* >( this->GetInput( "ParameterObject" ) );
  ParameterMapVectorType parameterMapVector = parameterObject->GetParameterMap();

  if( parameterMapVector.size() == 0 )
  {
    itkExceptionMacro( "Empty parameter map in parameter object." );
  }

  // Elastix must always write result image to guarantee that the ITK pipeline is in a consistent state
  parameterMapVector[ parameterMapVector.size()-1 ][ "WriteResultImage" ] = ParameterValueVectorType( 1, "true" );

  ArgumentMapType argumentMap;

  // Initial transform
  if( !this->m_InitialTransformParameterFileName.empty() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-t0", this->m_InitialTransformParameterFileName ) );
  }

  // Fixed mesh
  if( !this->m_FixedPointSetFileName.empty() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-fp", this->m_FixedPointSetFileName ) );
  }

  // Moving mesh
  if( !this->m_MovingPointSetFileName.empty() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-mp", this->m_MovingPointSetFileName ) );
  }

  // Logging, disk I/O
  if( this->GetOutputDirectory().empty() ) {
    if( this->GetLogToFile() )
    {
      itkExceptionMacro( "LogToFileOn() requires an output directory to be specified.")
    }

    // There must be an "-out" as this is checked later in the code
    argumentMap.insert( ArgumentMapEntryType( "-out", "output_path_not_set" ) );
  }
  else
  {
    if( !itksys::SystemTools::FileExists( this->GetOutputDirectory() ) )
    {
       itkExceptionMacro( "Output directory \"" << this->GetOutputDirectory() << "\" does not exist." );
    }

    if( this->GetOutputDirectory()[ this->GetOutputDirectory().size()-1 ] != '/' || this->GetOutputDirectory()[ this->GetOutputDirectory().size()-1 ] != '\\' )
    {
      this->SetOutputDirectory( this->GetOutputDirectory() + "/" );
    }

    argumentMap.insert( ArgumentMapEntryType( "-out", this->GetOutputDirectory() ) );
  }

  // Setup xout
  std::string logFileName;
  if( this->GetLogToFile() )
  {
    if( this->GetLogFileName().empty() )
    {
      logFileName = this->GetOutputDirectory() + "transformix.log";
    }
    else
    {
      logFileName = this->GetOutputDirectory() + this->GetLogFileName();
    }
  }

  if( elx::xoutSetup( logFileName.c_str(), this->GetLogToFile(), this->GetLogToConsole() ) )
  {
    itkExceptionMacro( "Error while setting up xout" );
  }

  // Run the (possibly multiple) registration(s)
  for( unsigned int i = 0; i < parameterMapVector.size(); ++i )
  {
    // Instantiated pixeltypes are the groundtruth
    parameterMapVector[ i ][ "FixedInternalImagePixelType" ] = ParameterValueVectorType( 1, PixelTypeName< typename TFixedImage::PixelType >::ToString() );
    parameterMapVector[ i ][ "FixedImageDimension" ] = ParameterValueVectorType( 1, ParameterObject::ToString( FixedImageDimension ) );
    parameterMapVector[ i ][ "MovingInternalImagePixelType" ] = ParameterValueVectorType( 1, PixelTypeName< typename TMovingImage::PixelType >::ToString() );
    parameterMapVector[ i ][ "MovingImageDimension" ] = ParameterValueVectorType( 1, ParameterObject::ToString( MovingImageDimension ) );
    parameterMapVector[ i ][ "ResultImagePixelType" ] = ParameterValueVectorType( 1, PixelTypeName< typename TFixedImage::PixelType >::ToString() );

    // Create another instance of ElastixMain
    ElastixMainPointer elastix = ElastixMainType::New();

    // Set the current elastix-level
    elastix->SetElastixLevel( i );
    elastix->SetTotalNumberOfElastixLevels( parameterMapVector.size() );

    // Set stuff we get from a previous registration
    elastix->SetInitialTransform( transform );
    elastix->SetFixedImageContainer( fixedImageContainer );
    elastix->SetMovingImageContainer( movingImageContainer );
    elastix->SetFixedMaskContainer( fixedMaskContainer );
    elastix->SetMovingMaskContainer( movingMaskContainer );
    elastix->SetResultImageContainer( resultImageContainer );
    elastix->SetOriginalFixedImageDirectionFlat( fixedImageOriginalDirection );

    // Start registration
    unsigned int isError = 0;
    try
    {
      isError = elastix->Run( argumentMap, parameterMapVector[ i ] );
    }
    catch( itk::ExceptionObject &e )
    {
      itkExceptionMacro( << "Errors occurred during registration: " << e.what() );
    }

    if( isError != 0 )
    {
      itkExceptionMacro( << "Uncought errors occurred during registration." );
    }

    // Get stuff in order to put it in the next registration
    transform                   = elastix->GetFinalTransform();
    fixedImageContainer         = elastix->GetFixedImageContainer();
    movingImageContainer        = elastix->GetMovingImageContainer();
    fixedMaskContainer          = elastix->GetFixedMaskContainer();
    movingMaskContainer         = elastix->GetMovingMaskContainer();
    resultImageContainer        = elastix->GetResultImageContainer();
    fixedImageOriginalDirection = elastix->GetOriginalFixedImageDirectionFlat();

    transformParameterMapVector.push_back( elastix->GetTransformParametersMap() );

    // Set initial transform to an index number instead of a parameter filename
    if( i > 0 )
    {
      std::stringstream index;
      index << ( i - 1 );
      transformParameterMapVector[ i ][ "InitialTransformParametersFileName" ][ 0 ] = index.str();
    }
  } // End loop over registrations

  // Save result image
  if( resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 )
  {
    this->SetPrimaryOutput( resultImageContainer->ElementAt( 0 ) );
  }
  else
  {
    itkExceptionMacro( "Errors occured during registration: Result image not available." );
  }

  // Assert that result is not empty
  FixedImagePointer resultImage = dynamic_cast< TFixedImage* >( this->GetPrimaryOutput() );
  typename TFixedImage::RegionType region = resultImage->GetLargestPossibleRegion();
  typename TFixedImage::SizeType size = region.GetSize();
  if( size[ 0 ] == 0 && size[ 1 ] == 0 )
  {
    itkExceptionMacro( "Result image is empty (size: " << size << "." );
  }

  // Save parameter map
  ParameterObject::Pointer transformParameterObject = ParameterObject::New();
  transformParameterObject->SetParameterMap( transformParameterMapVector );
  this->SetOutput( "TransformParameterObject", static_cast< itk::DataObject* >( transformParameterObject ) );

  // Override user pixeltype settings
  parameterObject->SetParameterMap( parameterMapVector);
  this->SetParameterObject( parameterObject );
  
  // Close the modules
  ElastixMainType::UnloadComponents();
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetParameterObject( ParameterObjectPointer parameterObject )
{
  this->SetInput( "ParameterObject", static_cast< itk::DataObject* >( parameterObject ) );
}

template< typename TFixedImage, typename TMovingImage >
typename ElastixFilter< TFixedImage, TMovingImage >::ParameterObjectPointer
ElastixFilter< TFixedImage, TMovingImage >
::GetParameterObject( void )
{
  return static_cast< ParameterObject* >( this->GetInput( "ParameterObject" ) );
}

template< typename TFixedImage, typename TMovingImage >
typename ElastixFilter< TFixedImage, TMovingImage >::FixedImagePointer
ElastixFilter< TFixedImage, TMovingImage >
::GetOutput( void )
{
  try {
    this->Update();
  }
  catch( itk::ExceptionObject &e )
  {
    itkExceptionMacro( "Errors occured during execution of ElastixFilter: " << e.what( ));
  }

  return static_cast< TFixedImage* >( itk::ProcessObject::GetOutput( "ResultImage" ) );
}

template< typename TFixedImage, typename TMovingImage >
typename ElastixFilter< TFixedImage, TMovingImage >::ParameterObjectPointer
ElastixFilter< TFixedImage, TMovingImage >
::GetTransformParameterObject( void )
{
  // Make sure the transform parameters have been generated and/or are up to date
  try {
    this->Update();
  }
  catch( itk::ExceptionObject &e )
  {
    itkExceptionMacro( "Errors occured during execution of ElastixFilter: " << e.what( ));
  }

  return static_cast< ParameterObject* >( itk::ProcessObject::GetOutput( "TransformParameterObject" ) );
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetFixedImage( FixedImagePointer fixedImage )
{
  // Free references to fixed images that has already been set
  this->RemoveInputType( "FixedImage" );

  this->SetInput( "FixedImage", static_cast< itk::DataObject* >( fixedImage ) );
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetFixedImage( DataObjectContainerPointer fixedImages )
{
  if( fixedImages->Size() == 0 )
  {
    itkExceptionMacro( "Cannot set fixed images from empty container.")
  }

  // Free references to fixed images that has already been set
  this->RemoveInputType( "FixedImage" );

  // The first image will be named "FixedImage" while the rest of
  // the images will be appended to the input container suffixed 
  // with _1, _2, etc. The common prefix allows us to read out only
  // the fixed images for elastix fixed image container at a later
  // stage while the ITK filter can find its required "FixedImage" input 
  DataObjectContainerIterator fixedImageIterator = fixedImages->Begin();
  this->SetInput( "FixedImage", fixedImageIterator->Value() );
  ++fixedImageIterator;

  while( fixedImageIterator != fixedImages->End() )
  {
    this->AddInputAndAutoIncrementName( "FixedImage", fixedImageIterator->Value() );
    ++fixedImageIterator;
  }
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::AddFixedImage( FixedImagePointer fixedImage )
{
  if( this->GetInput( "FixedImage" ) == ITK_NULLPTR )
  {
    this->SetFixedImage( fixedImage );
  }
  else
  {
    this->AddInputAndAutoIncrementName( "FixedImage", static_cast< itk::DataObject* >( fixedImage ) );
  }
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetMovingImage( MovingImagePointer movingImage )
{
  // Free references to moving images that has already been set
  this->RemoveInputType( "MovingImage" );

  this->SetInput( "MovingImage", static_cast< itk::DataObject* >( movingImage ) );
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetMovingImage( DataObjectContainerPointer movingImages )
{
  if( movingImages->Size() == 0 )
  {
    itkExceptionMacro( "Cannot set moving images from empty container.")
  }

  // Free references to fixed images that has already been set
  this->RemoveInputType( "MovingImage" );

  // The first image will be named "MovingImage" while the rest of
  // the images will be appended to the input container suffixed 
  // with _1, _2, etc. The common prefix allows us to read out only
  // the moving images for elastix moving image container at a later
  // stage while the ITK filter can find its required "MovingImage" input 
  DataObjectContainerIterator movingImageIterator = movingImages->Begin();
  this->SetInput( "MovingImage", movingImageIterator->Value() );
  ++movingImageIterator;

  while( movingImageIterator != movingImages->End() )
  {
    this->AddInputAndAutoIncrementName( "MovingImage", static_cast< itk::DataObject* >( movingImageIterator->Value() ) );
    ++movingImageIterator;
  }
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::AddMovingImage( MovingImagePointer movingImage )
{
  if( this->GetInput( "MovingImage" ) == ITK_NULLPTR )
  {
    this->SetMovingImage( movingImage );
  }
  else
  {
    this->AddInputAndAutoIncrementName( "MovingImage", static_cast< itk::DataObject* >( movingImage ) );
  }
}


template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetFixedMask( FixedImagePointer fixedMask )
{
  // Free references to fixed masks that has already been set
  this->RemoveInputType( "FixedMask" );

  this->SetInput( "FixedMask", static_cast< itk::DataObject* >( fixedMask ) );
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetFixedMask( DataObjectContainerPointer fixedMasks )
{
  if( fixedMasks->Size() == 0 )
  {
    itkExceptionMacro( "Cannot set fixed masks from empty container.")
  }

  // Free references to fixed images that has already been set
  this->RemoveInputType( "FixedMask" );

  DataObjectContainerIterator fixedMaskIterator = fixedMasks->Begin();
  while( fixedMaskIterator != fixedMasks->End() )
  {
    this->AddInputAndAutoIncrementName( "FixedMask", fixedMaskIterator->Value() );
    ++fixedMaskIterator;
  }
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::AddFixedMask( FixedImagePointer fixedMask )
{
  this->AddInputAndAutoIncrementName( "FixedMask", static_cast< itk::DataObject* >( fixedMask ) );
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::RemoveFixedMask( void )
{
  this->RemoveInputType( "FixedMask" );
  this->m_FixedMaskContainer = 0;
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetMovingMask( MovingImagePointer movingMask )
{
  // Free references to moving masks that has already been set
  this->RemoveInputType( "MovingMask" );

  this->SetInput( "MovingMask", static_cast< itk::DataObject* >( movingMask ) );
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetMovingMask( DataObjectContainerPointer movingMasks )
{
  if( movingMasks->Size() == 0 )
  {
    itkExceptionMacro( "Cannot set moving images from empty container.")
  }

  // Free references to moving masks that has already been set
  this->RemoveInputType( "MovingMask" );

  DataObjectContainerIterator movingMaskIterator = movingMasks->Begin();
  while( movingMaskIterator != movingMasks->End() )
  {
    this->AddInputAndAutoIncrementName( "MovingMask", movingMaskIterator->Value() );
    ++movingMaskIterator;
  }
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::AddMovingMask( MovingImagePointer movingMask )
{
  this->AddInputAndAutoIncrementName( "MovingMask", static_cast< itk::DataObject* >( movingMask ) );
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::RemoveMovingMask( void )
{
  this->RemoveInputType( "MovingMask" );
  this->m_MovingMaskContainer = 0;
}

/*
 * Adds a named input to the first null position in the input list
 * and expands the list memory if necessary
 */
template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::AddInputAndAutoIncrementName( DataObjectIdentifierType inputName, itk::DataObject* input )
{
  for ( unsigned idx = 0; idx < this->GetNumberOfIndexedInputs(); ++idx )
  {
    if ( !this->GetInput( idx ) )
    {
      // Append number to name (e.g. append '_2' to inputName for idx = 2)
      inputName += this->MakeNameFromInputIndex( idx );
      this->SetInput( inputName, input );
      return;
    }
  }

  inputName += this->MakeNameFromInputIndex( this->GetNumberOfIndexedInputs() );
  this->SetInput( inputName, input );
  return;
}

template< typename TFixedImage, typename TMovingImage >
bool
ElastixFilter< TFixedImage, TMovingImage >
::IsInputType( DataObjectIdentifierType inputType, DataObjectIdentifierType inputName )
{
  return std::strncmp( inputType.c_str(), inputName.c_str(), std::min( inputType.size(), inputName.size() ) ) == 0;
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::RemoveInputType( DataObjectIdentifierType inputType )
{
  // Free references to inputType images that has already been set
  InputNameArrayType inputNames = this->GetInputNames();
  for( unsigned int i = 0; i < inputNames.size(); ++i )
  {
    if( this->IsInputType( inputType, inputNames[ i ] ) )
    {
      this->RemoveInput( inputNames[ i ] );
    }
  }
}

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::VerifyInputInformation()
{
  // Override superclass and let elastix handle input verification
}

} // namespace elx

#endif // elxElastixFilter_hxx
