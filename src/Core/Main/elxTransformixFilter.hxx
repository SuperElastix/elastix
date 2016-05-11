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

/**
 * ********************* Constructor *********************
 */

template< typename TInputImage  >
TransformixFilter< TInputImage >
::TransformixFilter( void )
{
  this->SetPrimaryInputName( "InputImage" );
  this->SetPrimaryOutputName( "ResultImage" );

  this->AddRequiredInputName( "TransformParameterObject" );

  this->m_InputPointSetFileName = "";
  this->m_ComputeSpatialJacobian = false;
  this->m_ComputeDeterminantOfSpatialJacobian = false;
  this->m_ComputeDeformationField = false;

  this->m_OutputDirectory = "";
  this->m_LogFileName = "";

  this->m_LogToConsole = false;
  this->m_LogToFile = false;

  // TransformixFilter must have an input image
  this->SetInput( "InputImage", TInputImage::New() );
} // end Constructor

/**
 * ********************* GenerateData *********************
 */

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
    itkExceptionMacro( "Expected at least one of SetInputImage(), "
                    << "SetInputPointSetFileName() "
                    << "ComputeSpatialJacobianOn(), "
                    << "ComputeDeterminantOfSpatialJacobianOn() or "
                    << "ComputeDeformationFieldOn(), "
                    << "to be set.\"" );
  }

  // TODO: Patch upstream transformix to split this into seperate arguments
  // Transformix uses "-def" for path to point sets AND as flag for writing deformation field
  if( this->GetComputeDeformationField() && !this->GetInputPointSetFileName().empty() )
  {
    itkExceptionMacro( << "For backwards compatibility, only one of ComputeDeformationFieldOn() "
                       << "or SetInputPointSetFileName() can be active at any one time." )
  }

  // Setup argument map which transformix uses internally ito figure out what needs to be done
  ArgumentMapType argumentMap;

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

  // Setup output directory
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
  
  if( !this->GetOutputDirectory().empty() && !itksys::SystemTools::FileExists( this->GetOutputDirectory() ) )
  {
    itkExceptionMacro( "Output directory \"" << this->GetOutputDirectory() << "\" does not exist." )
  }

  if( this->GetOutputDirectory().empty() ) {
    // There must be an "-out", this is checked later in the code
    argumentMap.insert( ArgumentMapEntryType( "-out", "output_path_not_set" ) );
  }
  else
  {
    if( this->GetOutputDirectory()[ this->GetOutputDirectory().size() - 1 ] != '/' 
     && this->GetOutputDirectory()[ this->GetOutputDirectory().size() - 1 ] != '\\' )
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
      logFileName = this->GetOutputDirectory() + "transformix.log";
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
  
  // Set pixel types from input image, override user settings
  for( unsigned int i = 0; i < transformParameterMapVector.size(); ++i )
  {
    transformParameterMapVector[ i ][ "FixedInternalImagePixelType" ] = ParameterValueVectorType( 1, PixelType< typename TInputImage::PixelType >::ToString() );
    transformParameterMapVector[ i ][ "FixedImageDimension" ] = ParameterValueVectorType( 1, ParameterObject::ToString( InputImageDimension ) );
    transformParameterMapVector[ i ][ "MovingInternalImagePixelType" ] = ParameterValueVectorType( 1, PixelType< typename TInputImage::PixelType >::ToString() );
    transformParameterMapVector[ i ][ "MovingImageDimension" ] = ParameterValueVectorType( 1, ParameterObject::ToString( InputImageDimension ) );
    transformParameterMapVector[ i ][ "ResultImagePixelType" ] = ParameterValueVectorType( 1, PixelType< typename TInputImage::PixelType >::ToString() );
  }

  // Run transformix
  unsigned int isError = 0;
  try
  {
    isError = transformix->Run( argumentMap, transformParameterMapVector );
  }
  catch( itk::ExceptionObject &e )
  {
    itkExceptionMacro( "Errors occured during execution: " << e.what() );
  }

  if( isError != 0 )
  {
    itkExceptionMacro( "Internal transformix error: See transformix log (use LogToConsoleOn() or LogToFileOn())" );
  }

  // Save result image
  DataObjectContainerPointer resultImageContainer = transformix->GetResultImageContainer();
  if( resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 )
  {
    this->GraftOutput( "ResultImage", resultImageContainer->ElementAt( 0 ) );
  }
} // end GenerateData()

/**
 * ********************* SetInput *********************
 */

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::SetInput( TInputImage* inputImage )
{
  this->SetInput( "InputImage", dynamic_cast< itk::DataObject* >( inputImage ) );
} // end SetInput()

/**
 * ********************* GetInput *********************
 */

template< typename TInputImage >
typename TransformixFilter< TInputImage >::InputImageConstPointer
TransformixFilter< TInputImage >
::GetInput( void )
{
  return static_cast< TInputImage* >( this->GetInput( "InputImage" ) );
} // end GetInput()

/**
 * ********************* RemoveInput *********************
 */

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::RemoveInput( void )
{
  this->SetInput( TInputImage::New() );
} // end RemoveInput

/**
 * ********************* SetTransformParameterObject *********************
 */

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::SetTransformParameterObject( ParameterObjectPointer parameterObject )
{
  this->SetInput( "TransformParameterObject", static_cast< itk::DataObject* >( parameterObject ) );
} // end SetTransformParameterObject()

/**
 * ********************* GetTransformParameterObject *********************
 */

template< typename TInputImage >
typename TransformixFilter< TInputImage >::ParameterObjectType*
TransformixFilter< TInputImage >
::GetTransformParameterObject( void )
{
  this->Update();
  return static_cast< ParameterObjectType* >( this->GetInput( "TransformParameterObject" ) );
} // end GetTransformParameterObject()

/**
 * ********************* GetTransformParameterObject *********************
 */

template< typename TInputImage >
const typename TransformixFilter< TInputImage >::ParameterObjectType*
TransformixFilter< TInputImage >
::GetTransformParameterObject( void ) const
{
  this->Update();
  return static_cast< const ParameterObjectType* >( this->GetInput( "TransformParameterObject" ) );
} // end GetTransformParameterObject()

/**
* ********************* IsEmpty ****************************
*/

template< typename TInputImage >
bool
TransformixFilter< TInputImage >
::IsEmpty( const InputImagePointer inputImage )
{
  typename TInputImage::RegionType region = inputImage->GetLargestPossibleRegion();
  return region.GetNumberOfPixels() == 0;
} // end IsEmpty()

/**
 * ********************* SetLogFileName ****************************
 */

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::SetLogFileName( std::string logFileName )
{
  this->m_LogFileName = logFileName;
  this->LogToFileOn();
} // end SetLogFileName()

/**
 * ********************* RemoveLogFileName ****************************
 */

template< typename TInputImage >
void
TransformixFilter< TInputImage >
::RemoveLogFileName( void ) {
  this->SetLogFileName( "" );
  this->LogToFileOff();
} // end RemoveLogFileName()

} // namespace elx

#endif // elxTransformixFilter_hxx
