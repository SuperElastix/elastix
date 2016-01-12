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
#ifndef elxTransformixFilter_hxx
#define elxTransformixFilter_hxx

namespace elastix
{

template< typename TInputImage  >
TransformixFilter< TInputImage >
::TransformixFilter( void )
{
  this->AddRequiredInputName( "TransformParameterObject");

  this->SetPrimaryInputName( "InputImage" );
  this->SetPrimaryOutputName( "ResultImage" );

  // The filter must have an input image set for the ITK pipeline
  // to be in a consistent state even if it is not used
  this->SetInputImage( TInputImage::New() );

  this->ComputeSpatialJacobianOff();
  this->ComputeDeterminantOfSpatialJacobianOff();
  this->ComputeDeformationFieldOff();
  this->m_InputPointSetFileName = std::string();

  this->m_OutputDirectory = std::string();
  this->m_LogFileName = std::string();

  this->LogToConsoleOff();
  this->LogToFileOff();
}

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::GenerateData( void )
{
  // Assert that at least one output has been requested
  if( !this->HasInput( "InputImage" ) &&
      !this->GetComputeSpatialJacobian() &&
      !this->GetComputeDeterminantOfSpatialJacobian() &&
      !this->GetComputeDeformationField() &&
      this->GetInputPointSetFileName().empty() )
  {
    itkExceptionMacro( "Expected at least one of SetInputImage(\"path/to/image\"), ComputeSpatialJacobianOn(), "
                    << "ComputeDeterminantOfSpatialJacobianOn(), ComputeDeformationFieldOn() or "
                    << "SetInputPointSetFileName(\"path/to/points\") or to bet set." );
  }

  // Check if an output directory is needed
  // TODO: Change behaviour upstream to have transformix save all outputs as data objects
  if( ( this->GetComputeSpatialJacobian() ||
        this->GetComputeDeterminantOfSpatialJacobian() ||
        this->GetComputeDeformationField() ||
        !this->GetInputPointSetFileName().empty() ||
        this->GetLogToFile() ) &&
      this->GetOutputDirectory().empty() )
  {
    itkExceptionMacro( "The requested outputs require an output directory to be specified."
                    << "Use SetOutputDirectory()." )
  }

  // Check if output directory exists
  if( ( this->GetComputeSpatialJacobian() ||
        this->GetComputeDeterminantOfSpatialJacobian() ||
        this->GetComputeDeformationField() ||
        !this->GetInputPointSetFileName().empty() ||
        this->GetLogToFile() ) &&
      !itksys::SystemTools::FileExists( this->GetOutputDirectory() ) )
  {
    itkExceptionMacro( "Output directory \"" << this->GetOutputDirectory() << "\" does not exist." )
  }

  // Transformix uses "-def" for path to point sets AND as flag for writing deformation field
  // TODO: Change behaviour upstream: Split into seperate arguments
  if( this->GetComputeDeformationField() && !this->GetInputPointSetFileName().empty() )
  {
    itkExceptionMacro( << "For backwards compatibility, only one of ComputeDeformationFieldOn() "
                       << "or SetInputPointSetFileName() can be active at any one time." )
  }

  // Setup argument map which transformix uses internally ito figure out what needs to be done
  ArgumentMapType argumentMap;
  if( this->GetOutputDirectory().empty() ) {
    // There must be an "-out", this is checked later in the code
    argumentMap.insert( ArgumentMapEntryType( "-out", "output_path_not_set" ) );
  }
  else
  {
    if( this->GetOutputDirectory().back() != '/' || this->GetOutputDirectory().back() != '\\' )
    {
      this->SetOutputDirectory( this->GetOutputDirectory() + "/" );
    }

    argumentMap.insert( ArgumentMapEntryType( "-out", this->GetOutputDirectory() ) );
  }

  if( this->GetComputeSpatialJacobian() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-jacmat", "all" ) );
  }

  if( this->GetComputeDeterminantOfSpatialJacobian() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-jac", "all" ) );
  }

  if( this->GetComputeDeformationField() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-def" , "all" ) );
  }

  if( !this->GetInputPointSetFileName().empty() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-def", this->GetInputPointSetFileName() ) );
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

  // Instantiate transformix
  TransformixMainPointer transformix = TransformixMainType::New();

  DataObjectContainerPointer inputImageContainer = 0;
  DataObjectContainerPointer resultImageContainer = 0;

  // Normally we would use HasInput( "InputImage" ) to check if the input image
  // is available. However, an empty input image is set in the constructor because
  // a primary input is needed for the ITK pipeline to be in a consistent state.
  // Here, we assume that an input image is given if its size is non-empty instead.
  // We have to cast the data object to an image in order to perform this check
  typename TInputImage::Pointer inputImage = static_cast< TInputImage* >( this->GetInput( "InputImage" ) );
  typename TInputImage::RegionType region = inputImage->GetLargestPossibleRegion();
  typename TInputImage::SizeType size = region.GetSize();
  if( size[ 0 ] > 0 && size[ 1 ] > 0 ) {
    DataObjectContainerPointer inputImageContainer = DataObjectContainerType::New();
    inputImageContainer->CreateElementAt( 0 ) = this->GetInput("InputImage");
    transformix->SetInputImageContainer( inputImageContainer );
    transformix->SetResultImageContainer( resultImageContainer );
  }

  // Get ParameterMap
  ParameterObjectConstPointer transformParameterObject = static_cast< const ParameterObject* >( this->GetInput( "TransformParameterObject" ) );
  ParameterMapVectorType transformParameterMapVector = transformParameterObject->GetParameterMap();

  for( unsigned int i = 0; i < transformParameterMapVector.size(); ++i )
  {
    // Transformix reads type information from parameter files. We set this information automatically and overwrite
    // user settings in case they are incorrect (in which case elastix will segfault or throw exception)
    transformParameterMapVector[ i ][ "FixedInternalImagePixelType" ] = ParameterValueVectorType( 1, PixelTypeName< typename TInputImage::PixelType >::ToString() );
    transformParameterMapVector[ i ][ "FixedImageDimension" ] = ParameterValueVectorType( 1, ParameterObject::ToString( InputImageDimension ) );
    transformParameterMapVector[ i ][ "MovingInternalImagePixelType" ] = ParameterValueVectorType( 1, PixelTypeName< typename TInputImage::PixelType >::ToString() );
    transformParameterMapVector[ i ][ "MovingImageDimension" ] = ParameterValueVectorType( 1, ParameterObject::ToString( InputImageDimension ) );
    transformParameterMapVector[ i ][ "ResultImagePixelType" ] = ParameterValueVectorType( 1, PixelTypeName< typename TInputImage::PixelType >::ToString() );
  }

  // Run transformix
  unsigned int isError = 0;
  try
  {
    isError = transformix->Run( argumentMap, transformParameterMapVector );
  }
  catch( itk::ExceptionObject &e )
  {
    itkExceptionMacro( << "Errors occured during registration: " << e.what() );
  }

  if( isError != 0 )
  {
    itkExceptionMacro( << "Uncought errors occured during registration." );
  }

  // Save result image
  resultImageContainer = transformix->GetResultImageContainer();
  if( resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 )
  {
    this->GraftOutput( "ResultImage", resultImageContainer->ElementAt( 0 ) );
  }

  // Clean up
  TransformixMainType::UnloadComponents();
}

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::SetInputImage( InputImagePointer inputImage )
{
  this->SetInput( "InputImage", static_cast< itk::DataObject* >( inputImage ) );
}

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::SetTransformParameterObject( ParameterObjectPointer parameterObject )
{
  this->SetInput( "TransformParameterObject", static_cast< itk::DataObject* >( parameterObject ) );
}

template< typename TInputImage >
typename TransformixFilter< TInputImage >::ParameterObjectPointer
TransformixFilter< TInputImage >
::GetTransformParameterObject( void )
{
  return static_cast< ParameterObject* >( this->GetInput( "TransformParameterObject" ) );
}

} // namespace elx

#endif // elxTransformixFilter_hxx
