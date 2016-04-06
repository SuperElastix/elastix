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
  this->AddRequiredInputName( "TransformParameterObject" );

  this->SetPrimaryInputName( "InputImage" );
  this->SetPrimaryOutputName( "ResultImage" );

  this->SetInputPointSetFileName( "" );
  this->ComputeSpatialJacobianOff();
  this->ComputeDeterminantOfSpatialJacobianOff();
  this->ComputeDeformationFieldOff();

  this->SetOutputDirectory( "" );
  this->SetLogFileName( "" );

  this->LogToConsoleOff();
  this->LogToFileOff();

  this->SetInputImage( TInputImage::New() );
}

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::GenerateData( void )
{
  if( this->IsEmpty( static_cast< TInputImage* >( this->GetInput( "InputImage" ) ) ) &&
      this->GetInputPointSetFileName().empty() &&
      !this->GetComputeSpatialJacobian() &&
      !this->GetComputeDeterminantOfSpatialJacobian() &&
      !this->GetComputeDeformationField() )
  {
    typename TInputImage::RegionType region = static_cast< TInputImage* >( this->GetInput( "InputImage" ) )->GetLargestPossibleRegion();
    typename TInputImage::SizeType size = region.GetSize();
    itkExceptionMacro( "Expected at least one of SetInputImage(), "
                    << "SetInputPointSetFileName() "
                    << "ComputeSpatialJacobianOn(), "
                    << "ComputeDeterminantOfSpatialJacobianOn() or "
                    << "ComputeDeformationFieldOn(), "
                    << "to be set.\"" );
  }

  // Only the input "InputImage" does not require an output directory
  if( ( this->GetComputeSpatialJacobian() ||
        this->GetComputeDeterminantOfSpatialJacobian() ||
        this->GetComputeDeformationField() ||
        !this->GetInputPointSetFileName().empty() ||
        this->GetLogToFile() ) &&
      this->GetOutputDirectory().empty() )
  {
    this->SetOutputDirectory( "." );
  }

  // Check if output directory exists
  if( !this->GetOutputDirectory().empty() && !itksys::SystemTools::FileExists( this->GetOutputDirectory() ) )
  {
    itkExceptionMacro( "Output directory \"" << this->GetOutputDirectory() << "\" does not exist." )
  }

  // Transformix uses "-def" for path to point sets AND as flag for writing deformation field
  // TODO: Patch upstream transformix to split this into seperate arguments
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
      if( this->GetOutputDirectory()[ this->GetOutputDirectory().size()-1 ] != '/' || this->GetOutputDirectory()[ this->GetOutputDirectory().size()-1 ] != '\\' )
      {
        this->SetOutputDirectory( this->GetOutputDirectory() + "/" );
      }
      logFileName = this->GetOutputDirectory() + this->GetLogFileName();
    }
  }

  if( elx::xoutSetup( logFileName.c_str(), this->GetLogToFile(), this->GetLogToConsole() ) )
  {
    itkExceptionMacro( "Error while setting up xout" );
  }

  // Instantiate transformix
  TransformixMainPointer transformix = TransformixMainType::New();

  // Setup transformix for warping input image if given
  DataObjectContainerPointer inputImageContainer = 0;
  if( !this->IsEmpty( static_cast< TInputImage* >( this->GetInput( "InputImage" ) ) ) ) {
    inputImageContainer = DataObjectContainerType::New();
    inputImageContainer->CreateElementAt( 0 ) = this->GetInput( "InputImage" );
    transformix->SetInputImageContainer( inputImageContainer );
  }

  // Get ParameterMap
  ParameterObjectPointer transformParameterObject = static_cast< ParameterObject* >( this->GetInput( "TransformParameterObject" ) );
  ParameterMapVectorType transformParameterMapVector = transformParameterObject->GetParameterMap();

  // Assert user did not set empty parameter map
  if( transformParameterMapVector.size() == 0 )
  {
    itkExceptionMacro( "Empty parameter map in parameter object." );
  }
  
  // Instantiated pixel types are the groundtruth
  for( unsigned int i = 0; i < transformParameterMapVector.size(); ++i )
  {
    // Set pixel types from input image, override user settings
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
    itkExceptionMacro( "Errors occured during registration: " << e.what() );
  }

  if( isError != 0 )
  {
    itkExceptionMacro( "Internal transformix error: See transformix log." );
  }

  // Save result image
  DataObjectContainerPointer resultImageContainer = transformix->GetResultImageContainer();
  if( resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 )
  {
    this->GraftOutput( "ResultImage", resultImageContainer->ElementAt( 0 ) );

    if( this->IsEmpty( static_cast< TInputImage* >( this->GetPrimaryOutput( ) ) ) )
    {
      itkExceptionMacro( "Result image is empty (size: [0, 0])." );
    }
  }
}

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::SetInputImage( InputImagePointer inputImage )
{
  this->SetInput( "InputImage", static_cast< itk::DataObject* >( inputImage ) );
}

template< typename TInputImage >
typename TransformixFilter< TInputImage >::InputImagePointer
TransformixFilter< TInputImage >
::GetInputImage( void )
{
  return static_cast< TInputImage* >( this->GetInput( "InputImage" ) );
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

template< typename TInputImage >
bool
TransformixFilter< TInputImage >
::IsEmpty( InputImagePointer inputImage )
{
  typename TInputImage::RegionType region = inputImage->GetLargestPossibleRegion();
  typename TInputImage::SizeType size = region.GetSize();
  return size[ 0 ] == 0 && size[ 1 ] == 0;
}

} // namespace elx

#endif // elxTransformixFilter_hxx
