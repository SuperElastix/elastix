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

namespace elastix
{

/**
 * ********************* Constructor *********************
 */

template< typename TFixedImage, typename TMovingImage >
ElastixFilter< TFixedImage, TMovingImage >
::ElastixFilter( void )
{
  this->SetPrimaryInputName( "FixedImage" );
  this->SetPrimaryOutputName( "ResultImage" );

  this->AddRequiredInputName( "FixedImage" );
  this->AddRequiredInputName( "MovingImage" );
  this->AddRequiredInputName( "ParameterObject" );

  this->SetFixedPointSetFileName( "" );
  this->SetMovingPointSetFileName( "" );

  this->SetOutputDirectory( "." );
  this->SetLogFileName( "" );

  this->LogToConsoleOff();
  this->LogToFileOff();

  ParameterObjectPointer defaultParameterObject = ParameterObject::New();
  defaultParameterObject->AddParameterMap( "translation" );
  defaultParameterObject->AddParameterMap( "affine" );
  defaultParameterObject->AddParameterMap( "bspline" );
  this->SetParameterObject( defaultParameterObject );

  this->m_InputUID = 0;
} // end Constructor

/**
 * ********************* GenerateData *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::GenerateData( void )
{
  // Initialize variables here so they don't go out of scope between iterations of the main loop
  DataObjectContainerPointer  fixedImageContainer  = DataObjectContainerType::New();
  DataObjectContainerPointer  movingImageContainer = DataObjectContainerType::New();
  DataObjectContainerPointer  fixedMaskContainer   = 0;
  DataObjectContainerPointer  movingMaskContainer  = 0;
  DataObjectContainerPointer  resultImageContainer = 0;
  ElastixMainObjectPointer    transform            = 0;
  ParameterMapVectorType      transformParameterMapVector;
  FlatDirectionCosinesType    fixedImageOriginalDirection;

  // Split inputs into separate containers
  const InputNameArrayType inputNames = this->GetInputNames();
  for( unsigned int i = 0; i < inputNames.size(); ++i )
  {
    if( this->IsInputType( "FixedImage", inputNames[ i ] ) )
    {
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
  ParameterObjectPointer parameterObject = static_cast< ParameterObject* >( this->GetInput( "ParameterObject" ) );
  ParameterMapVectorType& parameterMapVector = parameterObject->GetParameterMap();

  if( parameterMapVector.size() == 0 )
  {
    itkExceptionMacro( "Empty parameter map in parameter object." );
  }

  // Elastix must always write result image to guarantee that the ITK pipeline is in a consistent state
  parameterMapVector[ parameterMapVector.size() - 1 ][ "WriteResultImage" ] = ParameterValueVectorType( 1, "true" );

  // Setup argument map
  ArgumentMapType argumentMap;

  if( !this->m_InitialTransformParameterFileName.empty() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-t0", this->m_InitialTransformParameterFileName ) );
  }

  if( !this->m_FixedPointSetFileName.empty() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-fp", this->m_FixedPointSetFileName ) );
  }

  if( !this->m_MovingPointSetFileName.empty() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-mp", this->m_MovingPointSetFileName ) );
  }

  // Setup output directory
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

    if( this->GetOutputDirectory()[ this->GetOutputDirectory().size()-1 ] != '/' 
     || this->GetOutputDirectory()[ this->GetOutputDirectory().size()-1 ] != '\\' )
    {
      this->SetOutputDirectory( this->GetOutputDirectory() + "/" );
    }

    argumentMap.insert( ArgumentMapEntryType( "-out", this->GetOutputDirectory() ) );
  }

  // Setup log file
  std::string logFileName;
  if( this->GetLogToFile() )
  {
    if( this->GetLogFileName().empty() )
    {
      logFileName = this->GetOutputDirectory() + "elastix.log";
    }
    else
    {
      logFileName = this->GetOutputDirectory() + this->GetLogFileName();
    }
  }

  // Setup xout
  if( elx::xoutSetup( logFileName.c_str(), this->GetLogToFile(), this->GetLogToConsole() ) )
  {
    itkExceptionMacro( "Error while setting up xout" );
  }

  // Run the (possibly multiple) registration(s)
  for( unsigned int i = 0; i < parameterMapVector.size(); ++i )
  {
    // Set pixel types from input images, override user settings
    parameterMapVector[ i ][ "FixedInternalImagePixelType" ] = ParameterValueVectorType( 1, PixelType< typename TFixedImage::PixelType >::ToString() );
    parameterMapVector[ i ][ "FixedImageDimension" ] = ParameterValueVectorType( 1, ParameterObject::ToString( FixedImageDimension ) );
    parameterMapVector[ i ][ "MovingInternalImagePixelType" ] = ParameterValueVectorType( 1, PixelType< typename TMovingImage::PixelType >::ToString() );
    parameterMapVector[ i ][ "MovingImageDimension" ] = ParameterValueVectorType( 1, ParameterObject::ToString( MovingImageDimension ) );
    parameterMapVector[ i ][ "ResultImagePixelType" ] = ParameterValueVectorType( 1, PixelType< typename TFixedImage::PixelType >::ToString() );

    // Create new instance of ElastixMain
    ElastixMainPointer elastix = ElastixMainType::New();

    // Set elastix levels
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
      itkExceptionMacro( << "Internal elastix error: See elastix log (use LogToConsoleOn() or LogToFileOn())." );
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

    // TODO: Fix elastix corrupting default pixel value
    transformParameterMapVector[ transformParameterMapVector.size() - 1 ][ "DefaultPixelValue" ] = parameterMapVector[ i ][ "DefaultPixelValue" ];

    // Set initial transform to an index number instead of a parameter filename
    if( i > 0 )
    {
      std::stringstream index;
      index << ( i - 1 ); // MS: Can this be done in the constructor of stringstream?
      transformParameterMapVector[ i ][ "InitialTransformParametersFileName" ][ 0 ] = index.str();
    }
  } // End loop over registrations

  // Save result image
  if( resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 )
  {
    this->GraftOutput( "ResultImage", resultImageContainer->ElementAt( 0 ) );
  }
  else
  {
    itkExceptionMacro( "Errors occured during registration: Could not read result image." );
  }

  // Save parameter map
  ParameterObject::Pointer transformParameterObject = ParameterObject::New();
  transformParameterObject->SetParameterMap( transformParameterMapVector );
  this->SetOutput( "TransformParameterObject", static_cast< itk::DataObject* >( transformParameterObject ) );
}

/**
 * ********************* SetParameterObject *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetParameterObject( ParameterObjectPointer parameterObject )
{
  this->SetInput( "ParameterObject", static_cast< itk::DataObject* >( parameterObject ) );
} // end SetParameterObject()

/**
 * ********************* GetParameterObject *********************
 */

template< typename TFixedImage, typename TMovingImage >
typename ElastixFilter< TFixedImage, TMovingImage >::ParameterObjectPointer
ElastixFilter< TFixedImage, TMovingImage >
::GetParameterObject( void )
{
  return static_cast< ParameterObject* >( this->GetInput( "ParameterObject" ) );
} // end GetParameterObject()

/**
 * ********************* GetTransformParameterObject *********************
 */

template< typename TFixedImage, typename TMovingImage >
typename ElastixFilter< TFixedImage, TMovingImage >::ParameterObjectPointer
ElastixFilter< TFixedImage, TMovingImage >
::GetTransformParameterObject( void )
{
  // The transform parameters are not the primary output so we have to manually ensure they are up to date
  this->Update();
  return static_cast< ParameterObject* >( itk::ProcessObject::GetOutput( "TransformParameterObject" ) );
} // end GetTransformParameterObject()

/**
 * ********************* SetFixedImage *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetFixedImage( FixedImagePointer fixedImage )
{
  // Free references to fixed images that has already been set
  this->RemoveInputType( "FixedImage" );
  this->SetInput( "FixedImage", static_cast< itk::DataObject* >( fixedImage ) );
} // end SetFixedImage()

/**
 * ********************* SetFixedImage *********************
 */

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
    this->SetInputWithUniqueName( "FixedImage", fixedImageIterator->Value() );
    ++fixedImageIterator;
  }
} // end SetFixedImage()

/**
 * ********************* AddFixedImage *********************
 */

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
    this->SetInputWithUniqueName( "FixedImage", static_cast< itk::DataObject* >( fixedImage ) );
  }
} // end AddFixedImage()

/**
 * ********************* SetMovingImage *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetMovingImage( MovingImagePointer movingImage )
{
  // Free references to moving images that has already been set
  this->RemoveInputType( "MovingImage" );

  this->SetInput( "MovingImage", static_cast< itk::DataObject* >( movingImage ) );
} // end SetMovingImage()

/**
 * ********************* SetMovingImage *********************
 */

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
    this->SetInputWithUniqueName( "MovingImage", static_cast< itk::DataObject* >( movingImageIterator->Value() ) );
    ++movingImageIterator;
  }
} // end SetMovingImage()

/**
 * ********************* AddMovingImage *********************
 */

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
    this->SetInputWithUniqueName( "MovingImage", static_cast< itk::DataObject* >( movingImage ) );
  }
} // end AddMovingImage()

/**
 * ********************* SetFixedMask *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetFixedMask( FixedMaskPointer fixedMask )
{
  // Free references to fixed masks that has already been set
  this->RemoveInputType( "FixedMask" );

  this->SetInput( "FixedMask", static_cast< itk::DataObject* >( fixedMask ) );
} // end SetFixedMask()

/**
 * ********************* SetFixedMask *********************
 */

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
    this->SetInputWithUniqueName( "FixedMask", fixedMaskIterator->Value() );
    ++fixedMaskIterator;
  }
} // end SetFixedMask()

/**
 * ********************* AddFixedMask *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::AddFixedMask( FixedMaskPointer fixedMask )
{
  this->SetInputWithUniqueName( "FixedMask", static_cast< itk::DataObject* >( fixedMask ) );
} // end AddFixedMask()

/**
 * *********************  *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::RemoveFixedMask( void )
{
  this->RemoveInputType( "FixedMask" );
  this->m_FixedMaskContainer = 0;
} // end RemoveFixedMask()

/**
 * ********************* RemoveFixedMask *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetMovingMask( MovingMaskPointer movingMask )
{
  // Free references to moving masks that has already been set
  this->RemoveInputType( "MovingMask" );
  this->AddMovingMask( movingMask );
} // end SetMovingMask()

/**
 * ********************* SetMovingMask *********************
 */

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
    this->SetInputWithUniqueName( "MovingMask", movingMaskIterator->Value() );
    ++movingMaskIterator;
  }
} // end SetMovingMask()

/**
 * ********************* AddMovingMask *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::AddMovingMask( MovingMaskPointer movingMask )
{
  this->SetInputWithUniqueName( "MovingMask", static_cast< itk::DataObject* >( movingMask ) );
} // end AddMovingMask()

/**
 * ********************* RemoveMovingMask *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::RemoveMovingMask( void )
{
  this->RemoveInputType( "MovingMask" );
  this->m_MovingMaskContainer = 0;
} // end RemoveMovingMask()

/**
 * ********************* SetInputWithUniqueName *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetInputWithUniqueName( const DataObjectIdentifierType& inputName, itk::DataObject* input )
{
  std::string uniqueInputName = inputName + "_" + ParameterObject::ToString( this->m_InputUID++ );
  this->SetInput( uniqueInputName, input );
} // end SetInputWithUniqueName()

/**
 * ********************* IsInputType *********************
 */

template< typename TFixedImage, typename TMovingImage >
bool
ElastixFilter< TFixedImage, TMovingImage >
::IsInputType( const DataObjectIdentifierType& inputType, DataObjectIdentifierType inputName )
{
  return std::strncmp( inputType.c_str(), inputName.c_str(), std::min( inputType.size(), inputName.size() ) ) == 0;
} // end IsInputType()

/**
 * ********************* RemoveInputType *********************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::RemoveInputType( const DataObjectIdentifierType& inputType )
{
  InputNameArrayType inputNames = this->GetInputNames();
  for( unsigned int i = 0; i < inputNames.size(); ++i )
  {
    if( this->IsInputType( inputType, inputNames[ i ] ) )
    {
      this->RemoveInput( inputNames[ i ] );
    }
  }
} // end RemoveInputType()

/**
 * ********************* SetLogFileName ****************************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::SetLogFileName( std::string logFileName )
{
  this->m_LogFileName = logFileName;
  this->LogToFileOn();
} // end SetLogFileName()

/**
 * ********************* RemoveLogFileName ****************************
 */

template< typename TFixedImage, typename TMovingImage >
void
ElastixFilter< TFixedImage, TMovingImage >
::RemoveLogFileName( void ) {
  this->SetLogFileName( "" );
  this->LogToFileOff();
} // end RemoveLogFileName()

} // namespace elx

#endif // elxElastixFilter_hxx
