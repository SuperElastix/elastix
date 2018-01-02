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

template< typename TMovingImage  >
TransformixFilter< TMovingImage >
::TransformixFilter( void )
{
  this->SetPrimaryInputName( "InputImage" );
  this->SetPrimaryOutputName( "ResultImage" );
  this->SetOutput( "ResultDeformationField", this->MakeOutput( "ResultDeformationField" ) );

  this->AddRequiredInputName( "TransformParameterObject" );

  this->m_FixedPointSetFileName               = "";
  this->m_ComputeSpatialJacobian              = false;
  this->m_ComputeDeterminantOfSpatialJacobian = false;
  this->m_ComputeDeformationField             = false;

  this->m_OutputDirectory = "";
  this->m_LogFileName     = "";

  this->m_LogToConsole = false;
  this->m_LogToFile    = false;

  // TransformixFilter must have an input image
  this->SetInput( "InputImage", TMovingImage::New() );
} // end Constructor


/**
 * ********************* GenerateData *********************
 */

template< typename TMovingImage >
void
TransformixFilter< TMovingImage >
::GenerateData( void )
{
  // Force compiler to instantiate the image dimension, otherwise we may get
  //   Undefined symbols for architecture x86_64:
  //     "elastix::TransformixFilter<itk::Image<float, 2u> >::MovingImageDimension"
  // on some platforms.
  const unsigned int movingImageDimension = MovingImageDimension;

  if( this->IsEmpty( itkDynamicCastInDebugMode< TMovingImage* >( this->GetInput( "InputImage" ) ) ) &&
      this->GetFixedPointSetFileName().empty() &&
      !this->GetComputeSpatialJacobian() &&
      !this->GetComputeDeterminantOfSpatialJacobian() &&
      !this->GetComputeDeformationField() )
  {
    itkExceptionMacro( "Expected at least one of SeTMovingImage(), "
                    << "SetFixedPointSetFileName() "
                    << "ComputeSpatialJacobianOn(), "
                    << "ComputeDeterminantOfSpatialJacobianOn() or "
                    << "ComputeDeformationFieldOn(), "
                    << "to be active.\"" );
  }

  // TODO: Patch upstream transformix to split this into seperate arguments
  // Transformix uses "-def" for path to point sets AND as flag for writing deformation field
  if( this->GetComputeDeformationField() && !this->GetFixedPointSetFileName().empty() )
  {
    itkExceptionMacro( << "For backwards compatibility, only one of ComputeDeformationFieldOn() "
                       << "or SetFixedPointSetFileName() can be active at any one time." )
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
    argumentMap.insert( ArgumentMapEntryType( "-def", "all" ) );
  }

  if( !this->GetFixedPointSetFileName().empty() )
  {
    argumentMap.insert( ArgumentMapEntryType( "-def", this->GetFixedPointSetFileName() ) );
  }

  // Setup output directory
  // Only the input "InputImage" does not require an output directory
  if( ( this->GetComputeSpatialJacobian()
    || this->GetComputeDeterminantOfSpatialJacobian()
    || this->GetComputeDeformationField()
    || !this->GetFixedPointSetFileName().empty()
    || this->GetLogToFile() )
    && this->GetOutputDirectory().empty() )
  {
    this->SetOutputDirectory( "." );
  }

  if( !this->GetOutputDirectory().empty() && !itksys::SystemTools::FileExists( this->GetOutputDirectory() ) )
  {
    itkExceptionMacro( "Output directory \"" << this->GetOutputDirectory() << "\" does not exist." )
  }

  if( this->GetOutputDirectory().empty() )
  {
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
  if( !this->IsEmpty( itkDynamicCastInDebugMode< TMovingImage* >( this->GetInput( "InputImage" ) ) ) ) {
    inputImageContainer = DataObjectContainerType::New();
    inputImageContainer->CreateElementAt( 0 ) = this->GetInput( "InputImage" );
    transformix->SetInputImageContainer( inputImageContainer );
  }

  // Get ParameterMap
  ParameterObjectPointer transformParameterObject = itkDynamicCastInDebugMode< ParameterObject * >( this->GetInput( "TransformParameterObject" ) );
  ParameterMapVectorType transformParameterMapVector = transformParameterObject->GetParameterMap();

  // Assert user did not set empty parameter map
  if( transformParameterMapVector.size() == 0 )
  {
    itkExceptionMacro( "Empty parameter map in parameter object." );
  }

  // Set pixel types from input image, override user settings
  for( unsigned int i = 0; i < transformParameterMapVector.size(); ++i )
  {
    transformParameterMapVector[ i ][ "FixedImageDimension" ]
      = ParameterValueVectorType( 1, ParameterObject::ToString( movingImageDimension ) );
    transformParameterMapVector[ i ][ "MovingImageDimension" ]
      = ParameterValueVectorType( 1, ParameterObject::ToString( movingImageDimension ) );
    transformParameterMapVector[ i ][ "ResultImagePixelType" ]
      = ParameterValueVectorType( 1, ParameterObject::ToString( PixelType< typename TMovingImage::PixelType >::ToString() ) );

    if( i > 0 )
    {
      transformParameterMapVector[ i ][ "InitialTransformParametersFileName" ]
        = ParameterValueVectorType( 1, ParameterObject::ToString( i - 1 ) );
    }
  }

  // Run transformix
  unsigned int isError = 0;
  try
  {
    isError = transformix->Run( argumentMap, transformParameterMapVector );
  }
  catch( itk::ExceptionObject & e )
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
  // Optionally, save result deformation field
  DataObjectContainerPointer resultDeformationFieldContainer = transformix->GetResultDeformationFieldContainer();
  if ( resultDeformationFieldContainer.IsNotNull() && resultDeformationFieldContainer->Size() > 0 )
  {
    this->GraftOutput( "ResultDeformationField", resultDeformationFieldContainer->ElementAt( 0 ) );
  }
} // end GenerateData()


/**
* ********************* MakeOutput *********************
*/
template< typename TMovingImage >
typename TransformixFilter< TMovingImage >::DataObjectPointer
TransformixFilter< TMovingImage >
::MakeOutput( const DataObjectIdentifierType & key )
{
  if ( key == "ResultImage" )
  {
    return TMovingImage::New().GetPointer();
  }
  else if ( key == "ResultDeformationField" )
  {
    return OutputDeformationFieldType::New().GetPointer();
  }
  else
  {
    // Primary and all other outputs default to ResultImage.
    return TMovingImage::New().GetPointer();
  }
} // end MakeOutput()

/**
 * ********************* SetMovingImage *********************
 */

template< typename TMovingImage >
void
TransformixFilter< TMovingImage >
::SetMovingImage( TMovingImage * inputImage )
{
  this->SetInput( "InputImage", inputImage );
} // end SetMovingImage()


/**
 * ********************* GetMovingImage *********************
 */

template< typename TMovingImage >
typename TransformixFilter< TMovingImage >::InputImageConstPointer
TransformixFilter< TMovingImage >
::GetMovingImage( void )
{
  return itkDynamicCastInDebugMode< TMovingImage * >( this->GetInput( "InputImage" ) );
} // end GetMovingImage()


/**
 * ********************* RemoveMovingImage *********************
 */

template< typename TMovingImage >
void
TransformixFilter< TMovingImage >
::RemoveMovingImage( void )
{
  this->RemoveInput( "InputImage" );
} // end RemoveMovingImage


/**
 * ********************* SetTransformParameterObject *********************
 */

template< typename TMovingImage >
void
TransformixFilter< TMovingImage >
::SetTransformParameterObject( ParameterObjectPointer parameterObject )
{
  this->SetInput( "TransformParameterObject", parameterObject );
} // end SetTransformParameterObject()


/**
 * ********************* GetTransformParameterObject *********************
 */

template< typename TMovingImage >
typename TransformixFilter< TMovingImage >::ParameterObjectType *
TransformixFilter< TMovingImage >
::GetTransformParameterObject( void )
{
  return dynamic_cast< ParameterObjectType * >( this->GetInput( "TransformParameterObject" ) );
} // end GetTransformParameterObject()


/**
 * ********************* GetTransformParameterObject *********************
 */

template< typename TMovingImage >
const typename TransformixFilter< TMovingImage >::ParameterObjectType *
TransformixFilter< TMovingImage >
::GetTransformParameterObject( void ) const
{
  return dynamic_cast< const ParameterObjectType * >( this->GetInput( "TransformParameterObject" ) );
} // end GetTransformParameterObject()


/**
*  ********************* GetOutputDeformationField *********************
*/
template< typename TMovingImage >
typename TransformixFilter< TMovingImage >::OutputDeformationFieldType *
TransformixFilter< TMovingImage >
::GetOutputDeformationField()
{

  return itkDynamicCastInDebugMode< OutputDeformationFieldType * >(
    this->itk::ProcessObject::GetOutput( "ResultDeformationField" ) );
} // end GetOutputDeformationField

/**
*  ********************* GetOutputDeformationField *********************
*/
template< typename TMovingImage >
const typename TransformixFilter< TMovingImage >::OutputDeformationFieldType *
TransformixFilter< TMovingImage >
::GetOutputDeformationField() const
{

  return itkDynamicCastInDebugMode< const OutputDeformationFieldType * >(
    this->itk::ProcessObject::GetOutput( "ResultDeformationField" ) );
} // end GetOutputDeformationField


/**
* ********************* IsEmpty ****************************
*/

template< typename TMovingImage >
bool
TransformixFilter< TMovingImage >
::IsEmpty( const InputImagePointer inputImage )
{
  typename TMovingImage::RegionType region = inputImage->GetLargestPossibleRegion();
  return region.GetNumberOfPixels() == 0;
} // end IsEmpty()


/**
 * ********************* SetLogFileName ****************************
 */

template< typename TMovingImage >
void
TransformixFilter< TMovingImage >
::SetLogFileName( std::string logFileName )
{
  this->m_LogFileName = logFileName;
  this->LogToFileOn();
} // end SetLogFileName()


/**
 * ********************* RemoveLogFileName ****************************
 */

template< typename TMovingImage >
void
TransformixFilter< TMovingImage >
::RemoveLogFileName( void )
{
  this->m_LogFileName = "";
  this->LogToFileOff();
} // end RemoveLogFileName()


} // namespace elx

#endif // elxTransformixFilter_hxx
