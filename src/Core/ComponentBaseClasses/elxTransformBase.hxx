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
#ifndef __elxTransformBase_hxx
#define __elxTransformBase_hxx

#include "elxTransformBase.h"

#include "itkPointSet.h"
#include "itkDefaultStaticMeshTraits.h"
#include "itkTransformixInputPointFileReader.h"
#include "vnl/vnl_math.h"
#include <itksys/SystemTools.hxx>
#include "itkVector.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "itkTransformToDeterminantOfSpatialJacobianSource.h"
#include "itkTransformToSpatialJacobianSource.h"
#include "itkImageFileWriter.h"
#include "itkImageGridSampler.h"
#include "itkContinuousIndex.h"
#include "itkChangeInformationImageFilter.h"
#include "itkMesh.h"
#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"
#include "itkTransformMeshFilter.h"

namespace itk
{

/** \class PixelTypeChangeCommand
 * \brief Command that modifies the PixelType of an ImageIO object.
 *
 * This class is used for writing the fullSpatialJacobian image.
 * It is a hack to ensure that a matrix image is seen as a
 * vector image, which most IO classes understand.
 *
 * \ingroup ITKSystemObjects
 */
template< class T >
class PixelTypeChangeCommand : public Command
{
public:

  /** Standard class typedefs. */
  typedef PixelTypeChangeCommand    Self;
  typedef itk::SmartPointer< Self > Pointer;

  /** This is supposed to be an ImageFileWriter */
  typedef T CallerType;

  /** Run-time type information (and related methods). */
  itkTypeMacro( PixelTypeChangeCommand, Command );

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Set the pixel type to VECTOR */
  virtual void Execute( Object * caller, const EventObject & )
  {
    CallerType * castcaller = dynamic_cast< CallerType * >( caller );
    castcaller->GetImageIO()->SetPixelType( ImageIOBase::VECTOR );
  }


  virtual void Execute( const Object * caller, const EventObject & )
  {
    CallerType * castcaller = const_cast< CallerType * >(
      dynamic_cast< const CallerType * >( caller ) );
    castcaller->GetImageIO()->SetPixelType( ImageIOBase::VECTOR );
  }


protected:

  PixelTypeChangeCommand() {}
  virtual ~PixelTypeChangeCommand() {}

private:

  PixelTypeChangeCommand( const Self & ); // purposely not implemented
  void operator=( const Self & );         // purposely not implemented

};

} // end namespace itk

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
TransformBase< TElastix >
::TransformBase()
{
  /** Initialize. */
  this->m_TransformParametersPointer   = 0;
  this->m_ReadWriteTransformParameters = true;

} // end Constructor()


/**
 * ********************** Destructor ****************************
 */

template< class TElastix >
TransformBase< TElastix >
::~TransformBase()
{
  /** Delete. */
  if( this->m_TransformParametersPointer )
  {
    delete this->m_TransformParametersPointer;
  }

} // end Destructor()


/**
 * ******************** BeforeAllBase ***************************
 */

template< class TElastix >
int
TransformBase< TElastix >
::BeforeAllBase( void )
{
  /** Check Command line options and print them to the logfile. */
  elxout << "Command line options from TransformBase:" << std::endl;
  std::string check( "" );

  /** Check for appearance of "-t0". */
  check = this->m_Configuration->GetCommandLineArgument( "-t0" );
  if( check.empty() )
  {
    elxout << "-t0       unspecified, so no initial transform used" << std::endl;
  }
  else
  {
    elxout << "-t0       " << check << std::endl;
  }

  /** Return a value. */
  return 0;

} // end BeforeAllBase()


/**
 * ******************** BeforeAllTransformix ********************
 */

template< class TElastix >
int
TransformBase< TElastix >
::BeforeAllTransformix( void )
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Declare check. */
  std::string check = "";

  /** Check for appearance of "-ipp". */
  check = this->m_Configuration->GetCommandLineArgument( "-ipp" );
  if( check != "" )
  {
    elxout << "-ipp      " << check << std::endl;
    // Deprecated since elastix 4.3
    xl::xout[ "warning" ] << "WARNING: \"-ipp\" is deprecated, use \"-def\" instead!"
                          << std::endl;
  }

  /** Check for appearance of "-def". */
  check = this->m_Configuration->GetCommandLineArgument( "-def" );
  if( check == "" )
  {
    elxout << "-def      unspecified, so no input points transformed" << std::endl;
  }
  else
  {
    elxout << "-def      " << check << std::endl;
  }

  /** Check for appearance of "-jac". */
  check = this->m_Configuration->GetCommandLineArgument( "-jac" );
  if( check == "" )
  {
    elxout << "-jac      unspecified, so no det(dT/dx) computed" << std::endl;
  }
  else
  {
    elxout << "-jac      " << check << std::endl;
  }

  /** Check for appearance of "-jacmat". */
  check = this->m_Configuration->GetCommandLineArgument( "-jacmat" );
  if( check == "" )
  {
    elxout << "-jacmat   unspecified, so no dT/dx computed" << std::endl;
  }
  else
  {
    elxout << "-jacmat   " << check << std::endl;
  }

  /** Return a value. */
  return returndummy;

} // end BeforeAllTransformix()


/**
 * ******************* BeforeRegistrationBase *******************
 */

template< class TElastix >
void
TransformBase< TElastix >
::BeforeRegistrationBase( void )
{
  /** Read from the configuration file how to combine the initial
   * transform with the current transform.
   */
  std::string howToCombineTransforms = "Compose";
  this->m_Configuration->ReadParameter(
    howToCombineTransforms, "HowToCombineTransforms", 0, false );

  /** Check if this is a CombinationTransform. */
  CombinationTransformType * thisAsGrouper
    = dynamic_cast< CombinationTransformType * >( this );
  if( thisAsGrouper )
  {
    if( howToCombineTransforms == "Compose" )
    {
      thisAsGrouper->SetUseComposition( true );
    }
    else
    {
      thisAsGrouper->SetUseComposition( false );
    }
  }

  /** Set the initial transform. Elastix returns an itk::Object, so try to
   * cast it to an InitialTransformType, which is of type itk::Transform.
   * No need to cast to InitialAdvancedTransformType, since InitialAdvancedTransformType
   * inherits from InitialTransformType.
   */
  if( this->m_Elastix->GetInitialTransform() )
  {
    InitialTransformType * testPointer = dynamic_cast< InitialTransformType * >(
      this->m_Elastix->GetInitialTransform() );
    if( testPointer )
    {
      this->SetInitialTransform( testPointer );
    }
  }
  else
  {
    std::string fileName =  this->m_Configuration->GetCommandLineArgument( "-t0" );
    if( !fileName.empty() )
    {
      if( itksys::SystemTools::FileExists( fileName.c_str() ) )
      {
        this->ReadInitialTransformFromFile( fileName.c_str() );
      }
      else
      {
        itkExceptionMacro( << "ERROR: the file " << fileName << " does not exist!" );
      }
    }
  }

} // end BeforeRegistrationBase()


/**
 * ******************* GetInitialTransform **********************
 */

template< class TElastix >
const typename TransformBase< TElastix >::InitialTransformType
* TransformBase< TElastix >
::GetInitialTransform( void ) const
{
  /** Cast to a(n Advanced)CombinationTransform. */
  const CombinationTransformType * thisAsGrouper
    = dynamic_cast< const CombinationTransformType * >( this );

  /** Set the initial transform. */
  if( thisAsGrouper )
  {
    return thisAsGrouper->GetInitialTransform();
  }

  return 0;

} // end GetInitialTransform()

/**
 * ******************* SetInitialTransform **********************
 */

template< class TElastix >
void
TransformBase< TElastix >
::SetInitialTransform( InitialTransformType * _arg )
{
  /** Cast to a(n Advanced)CombinationTransform. */
  CombinationTransformType * thisAsGrouper
    = dynamic_cast< CombinationTransformType * >( this );

  /** Set initial transform. */
  if( thisAsGrouper )
  {
    thisAsGrouper->SetInitialTransform( _arg );
  }

  // \todo AdvancedCombinationTransformType

} // end SetInitialTransform()


/**
 * ******************* SetFinalParameters ********************
 */

template< class TElastix >
void
TransformBase< TElastix >
::SetFinalParameters( void )
{
  /** Make a local copy, since some transforms do not do this,
   * like the B-spline transform.
   */
  this->m_FinalParameters = this->GetElastix()->GetElxOptimizerBase()
    ->GetAsITKBaseType()->GetCurrentPosition();

  /** Set the final Parameters for the resampler. */
  this->GetAsITKBaseType()->SetParameters( this->m_FinalParameters );

} // end SetFinalParameters()


/**
 * ******************* AfterRegistrationBase ********************
 */

template< class TElastix >
void
TransformBase< TElastix >
::AfterRegistrationBase( void )
{
  /** Set the final Parameters. */
  this->SetFinalParameters();

} // end AfterRegistrationBase()


/**
 * ******************* ReadFromFile *****************************
 */

template< class TElastix >
void
TransformBase< TElastix >
::ReadFromFile( void )
{
  /** NOTE:
   * This method assumes this->m_Configuration is initialized with a
   * transform parameter file, so not an elastix parameter file!!
   */

  /** Task 1 - Read the parameters from file. */

  /** Get the number of TransformParameters. */
  unsigned int numberOfParameters = 0;
  this->m_Configuration->ReadParameter( numberOfParameters, "NumberOfParameters", 0 );

  if( this->m_ReadWriteTransformParameters )
  {
    /** Get the TransformParameters pointer. */
    if( this->m_TransformParametersPointer )
    {
      delete this->m_TransformParametersPointer;
    }
    this->m_TransformParametersPointer = new ParametersType( numberOfParameters );

    /** Read the TransformParameters. */
    std::vector< ValueType > vecPar( numberOfParameters,
      itk::NumericTraits< ValueType >::ZeroValue() );
    this->m_Configuration->ReadParameter( vecPar, "TransformParameters",
      0, numberOfParameters - 1, true );

    /** Sanity check. Are the number of found parameters the same as
     * the number of specified parameters?
     * Do not rely on vecPar.size(), since it is unchanged by ReadParameter(),
     * so we cannot use: numberOfParametersFound = vecPar.size().
     */
    const std::size_t numberOfParametersFound
      = this->m_Configuration->CountNumberOfParameterEntries( "TransformParameters" );

    if( numberOfParametersFound != numberOfParameters )
    {
      std::ostringstream makeMessage( "" );
      makeMessage << "\nERROR: Invalid transform parameter file!\n"
                  << "The number of parameters in \"TransformParameters\" is "
                  << numberOfParametersFound
                  << ", which does not match the number specified in \"NumberOfParameters\" ("
                  << numberOfParameters << ").\n"
                  << "The transform parameters should be specified as:\n"
                  << "  (TransformParameters num num ... num)\n"
                  << "with " << numberOfParameters << " parameters." << std::endl;
      itkExceptionMacro( << makeMessage.str().c_str() );

      /** Historical note:
       * The old way of specifying parameters was
       *  - for less than 20 parameters:
       *      (TransformParameters num num ... num)
       *  - Otherwise:
       *      // (TransformParameters)
       *      // num num ... num
       *
       * This behavior was deprecated since elastix 4.2, and removed in elastix 4.5.
       */
    }

    /** Copy to m_TransformParametersPointer. */
    for( unsigned int i = 0; i < numberOfParameters; i++ )
    {
      ( *( this->m_TransformParametersPointer ) )[ i ] = vecPar[ i ];
    }

    /** Set the parameters into this transform. */
    this->GetAsITKBaseType()->SetParameters( *( this->m_TransformParametersPointer ) );

  } // end if this->m_ReadWriteTransformParameters

  /** Task 2 - Get the InitialTransform. */

  /** Get the InitialTransformName. */
  std::string fileName = "NoInitialTransform";
  this->m_Configuration->ReadParameter( fileName,
    "InitialTransformParametersFileName", 0 );

  /** Call the function ReadInitialTransformFromFile. */
  if( fileName != "NoInitialTransform" )
  {
#ifdef _ELASTIX_BUILD_LIBRARY
    /** Get initial transform index number. */
    std::istringstream to_size_t( fileName );
    size_t             index;
    to_size_t >> index;

    /** We can safely read the initial transform. */
    this->ReadInitialTransformFromVector( index );
#else
    /** Check if the initial transform of this transform parameter file
     * is not the same as this transform parameter file. Otherwise,
     * we will have an infinite loop.
     */
    std::string fullFileName1 = itksys::SystemTools::CollapseFullPath(
      fileName.c_str() );
    std::string fullFileName2 = itksys::SystemTools::CollapseFullPath(
      this->GetConfiguration()->GetCommandLineArgument( "-tp" ).c_str() );
    if( fullFileName1 == fullFileName2 )
    {
      itkExceptionMacro( << "ERROR: The InitialTransformParametersFileName "
                         << "is identical to the current TransformParameters filename! "
                         << "An infinite loop is not allowed." );
    }

    /** We can safely read the initial transform. */
    this->ReadInitialTransformFromFile( fileName.c_str() );
#endif
  }

  /** Task 3 - Read from the configuration file how to combine the
   * initial transform with the current transform.
   */
  std::string howToCombineTransforms = "Compose"; // default
  this->m_Configuration->ReadParameter( howToCombineTransforms,
    "HowToCombineTransforms", 0, true );

  /** Convert 'this' to a pointer to a CombinationTransform and set how
   * to combine the current transform with the initial transform.
   */
  CombinationTransformType * thisAsGrouper
    = dynamic_cast< CombinationTransformType * >( this );
  if( thisAsGrouper )
  {
    if( howToCombineTransforms == "Compose" )
    {
      thisAsGrouper->SetUseComposition( true );
    }
    else
    {
      thisAsGrouper->SetUseComposition( false );
    }
  }

  /** Task 4 - Remember the name of the TransformParametersFileName.
   * This will be needed when another transform will use this transform
   * as an initial transform (see the WriteToFile method)
   */
  this->SetTransformParametersFileName(
    this->GetConfiguration()->GetCommandLineArgument( "-tp" ).c_str() );

} // end ReadFromFile()


/**
 * ******************* ReadInitialTransformFromVector *****************************
 */

template< class TElastix >
void
TransformBase< TElastix >
::ReadInitialTransformFromVector( const size_t index )
{
  /** Retrieve configuration object from internally stored vector of configuration objects. */
  ConfigurationPointer configurationInitialTransform
    = this->GetElastix()->GetConfiguration( index );

  /** Read the InitialTransform name. */
  ComponentDescriptionType initialTransformName = "AffineTransform";
  configurationInitialTransform->ReadParameter(
    initialTransformName, "Transform", 0 );

  /** Create an InitialTransform. */
  ObjectType::Pointer initialTransform;

  PtrToCreator testcreator = 0;
  testcreator = this->GetElastix()->GetComponentDatabase()
    ->GetCreator( initialTransformName, this->m_Elastix->GetDBIndex() );
  initialTransform = testcreator ? testcreator() : NULL;

  Self * elx_initialTransform = dynamic_cast< Self * >(
    initialTransform.GetPointer() );

  /** Call the ReadFromFile method of the initialTransform. */
  if( elx_initialTransform )
  {
    //elx_initialTransform->SetTransformParametersFileName(transformParametersFileName);
    elx_initialTransform->SetElastix( this->GetElastix() );
    elx_initialTransform->SetConfiguration( configurationInitialTransform );
    elx_initialTransform->ReadFromFile();

    /** Set initial transform. */
    InitialTransformType * testPointer
      = dynamic_cast< InitialTransformType * >( initialTransform.GetPointer() );
    if( testPointer )
    {
      this->SetInitialTransform( testPointer );
    }

  } // end if
}   // end ReadInitialTransformFromVector()


/**
 * ******************* ReadInitialTransformFromFile *************
 */

template< class TElastix >
void
TransformBase< TElastix >
::ReadInitialTransformFromFile( const char * transformParametersFileName )
{
  /** Create a new configuration, which will be initialized with
   * the transformParameterFileName. */
  ConfigurationPointer configurationInitialTransform = ConfigurationType::New();

  /** Create argmapInitialTransform. */
  CommandLineArgumentMapType argmapInitialTransform;
  argmapInitialTransform.insert( CommandLineEntryType(
    "-tp", transformParametersFileName ) );

  int initfailure = configurationInitialTransform->Initialize(
    argmapInitialTransform );
  if( initfailure != 0 )
  {
    itkGenericExceptionMacro( << "ERROR: Reading initial transform "
                              << "parameters failed: " << transformParametersFileName );
  }

  /** Read the InitialTransform name. */
  ComponentDescriptionType initialTransformName = "AffineTransform";
  configurationInitialTransform->ReadParameter(
    initialTransformName, "Transform", 0 );

  /** Create an InitialTransform. */
  ObjectType::Pointer initialTransform;

  PtrToCreator testcreator = 0;
  testcreator = this->GetElastix()->GetComponentDatabase()
    ->GetCreator( initialTransformName, this->m_Elastix->GetDBIndex() );
  initialTransform = testcreator ? testcreator() : NULL;

  Self * elx_initialTransform = dynamic_cast< Self * >(
    initialTransform.GetPointer() );

  /** Call the ReadFromFile method of the initialTransform. */
  if( elx_initialTransform )
  {
    //elx_initialTransform->SetTransformParametersFileName(transformParametersFileName);
    elx_initialTransform->SetElastix( this->GetElastix() );
    elx_initialTransform->SetConfiguration( configurationInitialTransform );
    elx_initialTransform->ReadFromFile();

    /** Set initial transform. */
    InitialTransformType * testPointer
      = dynamic_cast< InitialTransformType * >( initialTransform.GetPointer() );
    if( testPointer )
    {
      this->SetInitialTransform( testPointer );
    }

  } // end if

} // end ReadInitialTransformFromFile()


/**
 * ******************* WriteToFile ******************************
 */

template< class TElastix >
void
TransformBase< TElastix >
::WriteToFile( void ) const
{
  /** Write the current set parameters to file. */
  this->WriteToFile( this->GetAsITKBaseType()->GetParameters() );

} // end WriteToFile()


/**
 * ******************* WriteToFile ******************************
 */

template< class TElastix >
void
TransformBase< TElastix >
::WriteToFile( const ParametersType & param ) const
{
  using namespace xl;

  /** Write the name of this transform. */
  xout[ "transpar" ] << "(Transform \""
                     << this->elxGetClassName() << "\")" << std::endl;

  /** Get the number of parameters of this transform. */
  unsigned int nrP = param.GetSize();

  /** Write the number of parameters of this transform. */
  xout[ "transpar" ] << "(NumberOfParameters "
                     << nrP << ")" << std::endl;

  /** Write the parameters of this transform. */
  if( this->m_ReadWriteTransformParameters )
  {
    /** In this case, write in a normal way to the parameter file. */
    xout[ "transpar" ] << "(TransformParameters ";
    for( unsigned int i = 0; i < nrP - 1; i++ )
    {
      xout[ "transpar" ] << param[ i ] << " ";
    }
    xout[ "transpar" ] << param[ nrP - 1 ] << ")" << std::endl;
  }

  /** Write the name of the parameters-file of the initial transform. */
  if( this->GetInitialTransform() )
  {
    xout[ "transpar" ] << "(InitialTransformParametersFileName \""
                       << ( dynamic_cast< const Self * >( this->GetInitialTransform() ) )
      ->GetTransformParametersFileName() << "\")" << std::endl;
  }
  else
  {
    xout[ "transpar" ]
      << "(InitialTransformParametersFileName \"NoInitialTransform\")"
      << std::endl;
  }

  /** Write the way Transforms are combined. 
   *  Set it to the default "Compose" when no combination transform is used. */
  std::string                      combinationMethod = "Compose";
  const CombinationTransformType * dummyComboTransform
    = dynamic_cast< const CombinationTransformType * >( this );
  if( dummyComboTransform )
  {
    if( dummyComboTransform->GetUseAddition() )
    {
      combinationMethod = "Add";
    }
  }

  xout[ "transpar" ] << "(HowToCombineTransforms \""
                     << combinationMethod << "\")" << std::endl;

  /** Write image specific things. */
  xout[ "transpar" ] << std::endl << "// Image specific" << std::endl;

  /** Write image dimensions. */
  unsigned int FixDim = FixedImageDimension;
  unsigned int MovDim = MovingImageDimension;
  xout[ "transpar" ] << "(FixedImageDimension "
                     << FixDim << ")" << std::endl;
  xout[ "transpar" ] << "(MovingImageDimension "
                     << MovDim << ")" << std::endl;

  /** Write image pixel types. */
  std::string fixpix = "float";
  std::string movpix = "float";
  this->m_Configuration->ReadParameter( fixpix, "FixedInternalImagePixelType", 0 );
  this->m_Configuration->ReadParameter( movpix, "MovingInternalImagePixelType", 0 );
  xout[ "transpar" ] << "(FixedInternalImagePixelType \""
                     << fixpix << "\")" << std::endl;
  xout[ "transpar" ] << "(MovingInternalImagePixelType \""
                     << movpix << "\")" << std::endl;

  /** Get the Size, Spacing and Origin of the fixed image. */
  typedef typename FixedImageType::SizeType      FixedImageSizeType;
  typedef typename FixedImageType::IndexType     FixedImageIndexType;
  typedef typename FixedImageType::SpacingType   FixedImageSpacingType;
  typedef typename FixedImageType::PointType     FixedImageOriginType;
  typedef typename FixedImageType::DirectionType FixedImageDirectionType;
  FixedImageSizeType size
    = this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  FixedImageIndexType index
    = this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetIndex();
  FixedImageSpacingType spacing
    = this->m_Elastix->GetFixedImage()->GetSpacing();
  FixedImageOriginType origin
    = this->m_Elastix->GetFixedImage()->GetOrigin();
  /** The following line would be logically: */
  //FixedImageDirectionType direction =
  //  this->m_Elastix->GetFixedImage()->GetDirection();
  /** But to support the UseDirectionCosines option, we should do it like this: */
  FixedImageDirectionType direction;
  this->GetElastix()->GetOriginalFixedImageDirection( direction );

  /** Write image Size. */
  xout[ "transpar" ] << "(Size ";
  for( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
  {
    xout[ "transpar" ] << size[ i ] << " ";
  }
  xout[ "transpar" ] << size[ FixedImageDimension - 1 ] << ")" << std::endl;

  /** Write image Index. */
  xout[ "transpar" ] << "(Index ";
  for( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
  {
    xout[ "transpar" ] << index[ i ] << " ";
  }
  xout[ "transpar" ] << index[ FixedImageDimension - 1 ] << ")" << std::endl;

  /** Set the precision of cout to 10, because Spacing and
   * Origin must have at least one digit precision.
   */
  xout[ "transpar" ] << std::setprecision( 10 );

  /** Write image Spacing. */
  xout[ "transpar" ] << "(Spacing ";
  for( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
  {
    xout[ "transpar" ] << spacing[ i ] << " ";
  }
  xout[ "transpar" ] << spacing[ FixedImageDimension - 1 ] << ")" << std::endl;

  /** Write image Origin. */
  xout[ "transpar" ] << "(Origin ";
  for( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
  {
    xout[ "transpar" ] << origin[ i ] << " ";
  }
  xout[ "transpar" ] << origin[ FixedImageDimension - 1 ] << ")" << std::endl;

  /** Write direction cosines. */
  xout[ "transpar" ] << "(Direction";
  for( unsigned int i = 0; i < FixedImageDimension; i++ )
  {
    for( unsigned int j = 0; j < FixedImageDimension; j++ )
    {
      xout[ "transpar" ] << " " << direction( j, i );
    }
  }
  xout[ "transpar" ] << ")" << std::endl;

  /** Set the precision back to default value. */
  xout[ "transpar" ] << std::setprecision( this->m_Elastix->GetDefaultOutputPrecision() );

  /** Write whether the direction cosines should be taken into account.
   * This parameter is written from elastix 4.203.
   */
  std::string useDirectionCosinesBool = "false";
  if( this->GetElastix()->GetUseDirectionCosines() )
  {
    useDirectionCosinesBool = "true";
  }
  xout[ "transpar" ] << "(UseDirectionCosines \""
                     << useDirectionCosinesBool << "\")" << std::endl;

} // end WriteToFile()


/**
 * ******************* CreateTransformParametersMap ******************************
 */

template< class TElastix >
void
TransformBase< TElastix >
::CreateTransformParametersMap(
  const ParametersType & param,
  ParameterMapType * paramsMap ) const
{
  std::ostringstream         tmpStream;
  std::string                parameterName;
  std::vector< std::string > parameterValues;

  /** Write the name of this transform. */
  parameterName = "Transform";
  parameterValues.push_back( this->elxGetClassName() );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Get the number of parameters of this transform. */
  unsigned int nrP = param.GetSize();

  /** Write the number of parameters of this transform. */
  parameterName = "NumberOfParameters";
  tmpStream.str( "" ); tmpStream << nrP;
  parameterValues.push_back( tmpStream.str() );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write the parameters of this transform. */
  if( this->m_ReadWriteTransformParameters )
  {
    /** In this case, write in a normal way to the parameter file. */
    parameterName = "TransformParameters";
    for( unsigned int i = 0; i < nrP; i++ )
    {
      tmpStream.str( "" ); tmpStream << param[ i ];
      parameterValues.push_back( tmpStream.str() );
    }
    paramsMap->insert( make_pair( parameterName, parameterValues ) );
    parameterValues.clear();
  }

  /** Write the name of the parameters-file of the initial transform. */
  if( this->GetInitialTransform() )
  {
    parameterName = "InitialTransformParametersFileName";
    parameterValues.push_back( ( dynamic_cast< const Self * >(
        this->GetInitialTransform() ) )->GetTransformParametersFileName() );
    paramsMap->insert( make_pair( parameterName, parameterValues ) );
    parameterValues.clear();
  }
  else
  {
    parameterName = "InitialTransformParametersFileName";
    parameterValues.push_back( "NoInitialTransform" );
    paramsMap->insert( make_pair( parameterName, parameterValues ) );
    parameterValues.clear();
  }

  /** Write the way Transforms are combined. */
  std::string                      combinationMethod = "Compose";
  const CombinationTransformType * dummyComboTransform
    = dynamic_cast< const CombinationTransformType * >( this );
  if( dummyComboTransform )
  {
    if( dummyComboTransform->GetUseComposition() )
    {
      combinationMethod = "Compose";
    }
  }

  parameterName = "HowToCombineTransforms";
  parameterValues.push_back( combinationMethod );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write image specific things. */
// xout["transpar"] << std::endl << "// Image specific" << std::endl;

  /** Write image dimensions. */
  const unsigned int fixDim = FixedImageDimension;
  const unsigned int movDim = MovingImageDimension;
  parameterName = "FixedImageDimension";
  tmpStream.str( "" ); tmpStream << fixDim;
  parameterValues.push_back( tmpStream.str() );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  parameterName = "MovingImageDimension";
  tmpStream.str( "" ); tmpStream << movDim;
  parameterValues.push_back( tmpStream.str() );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write image pixel types. */
  std::string fixpix = "float";
  std::string movpix = "float";
  this->m_Configuration->ReadParameter( fixpix, "FixedInternalImagePixelType", 0 );
  this->m_Configuration->ReadParameter( movpix, "MovingInternalImagePixelType", 0 );

  parameterName = "FixedInternalImagePixelType";
  parameterValues.push_back( fixpix );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  parameterName = "MovingInternalImagePixelType";
  parameterValues.push_back( movpix );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Get the Size, Spacing and Origin of the fixed image. */
  typedef typename FixedImageType::SizeType      FixedImageSizeType;
  typedef typename FixedImageType::IndexType     FixedImageIndexType;
  typedef typename FixedImageType::SpacingType   FixedImageSpacingType;
  typedef typename FixedImageType::PointType     FixedImageOriginType;
  typedef typename FixedImageType::DirectionType FixedImageDirectionType;
  FixedImageSizeType size
    = this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  FixedImageIndexType index
    = this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetIndex();
  FixedImageSpacingType spacing
    = this->m_Elastix->GetFixedImage()->GetSpacing();
  FixedImageOriginType origin
    = this->m_Elastix->GetFixedImage()->GetOrigin();
  /** The following line would be logically: */
  //FixedImageDirectionType direction =
  //  this->m_Elastix->GetFixedImage()->GetDirection();
  /** But to support the UseDirectionCosines option, we should do it like this: */
  FixedImageDirectionType direction;
  this->GetElastix()->GetOriginalFixedImageDirection( direction );

  /** Write image Size. */
  parameterName = "Size";
  for( unsigned int i = 0; i < FixedImageDimension; i++ )
  {
    tmpStream.str( "" ); tmpStream << size[ i ];
    parameterValues.push_back( tmpStream.str() );
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write image Index. */
  parameterName = "Index";
  for( unsigned int i = 0; i < FixedImageDimension; i++ )
  {
    tmpStream.str( "" ); tmpStream << index[ i ];
    parameterValues.push_back( tmpStream.str() );
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Set the precision of cout to 2, because Spacing and
   * Origin must have at least one digit precision.
   */
//  xout["transpar"] << std::setprecision(10);

  /** Write image Spacing. */
  parameterName = "Spacing";
  for( unsigned int i = 0; i < FixedImageDimension; i++ )
  {
    tmpStream.str( "" ); tmpStream << spacing[ i ];
    parameterValues.push_back( tmpStream.str() );
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write image Origin. */
  parameterName = "Origin";
  for( unsigned int i = 0; i < FixedImageDimension; i++ )
  {
    tmpStream.str( "" ); tmpStream << origin[ i ];
    parameterValues.push_back( tmpStream.str() );
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write direction cosines. */
  parameterName = "Direction";
  for( unsigned int i = 0; i < FixedImageDimension; i++ )
  {
    for( unsigned int j = 0; j < FixedImageDimension; j++ )
    {
      tmpStream.str( "" ); tmpStream << direction( j, i );
      parameterValues.push_back( tmpStream.str() );
    }
  }
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Set the precision back to default value. */
//  xout["transpar"] << std::setprecision( this->m_Elastix->GetDefaultOutputPrecision() );

  /** Write whether the direction cosines should be taken into account.
   * This parameter is written from elastix 4.203.
   */
  std::string useDirectionCosinesBool = "false";
  if( this->GetElastix()->GetUseDirectionCosines() )
  {
    useDirectionCosinesBool = "true";
  }
  parameterName = "UseDirectionCosines";
  parameterValues.push_back( useDirectionCosinesBool );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

} // end CreateTransformParametersMap()


/**
 * ******************* TransformPoints **************************
 *
 * This function reads points from a file (but only if requested)
 * and transforms these fixed-image coordinates to moving-image
 * coordinates.
 */

template< class TElastix >
void
TransformBase< TElastix >
::TransformPoints( void ) const
{
  /** If the optional command "-def" is given in the command
   * line arguments, then and only then we continue.
   */
  const std::string ipp = this->GetConfiguration()->GetCommandLineArgument( "-ipp" );
  std::string       def = this->GetConfiguration()->GetCommandLineArgument( "-def" );

  /** For backwards compatibility def = ipp. */
  if( def != "" && ipp != "" )
  {
    itkExceptionMacro( << "ERROR: Can not use both \"-def\" and \"-ipp\"!\n"
                       << "  \"-ipp\" is deprecated, use only \"-def\".\n" );
  }
  else if( def == "" && ipp != "" )
  {
    def = ipp;
  }

  /** If there is an input point-file? */
  if( def != "" && def != "all" )
  {
    if( itksys::SystemTools::StringEndsWith( def.c_str(), ".vtk" )
      || itksys::SystemTools::StringEndsWith( def.c_str(), ".VTK" ) )
    {
      elxout << "  The transform is evaluated on some points, "
             << "specified in a VTK input point file." << std::endl;
      this->TransformPointsSomePointsVTK( def );
    }
    else
    {
      elxout << "  The transform is evaluated on some points, "
             << "specified in the input point file." << std::endl;
      this->TransformPointsSomePoints( def );
    }
  }
  else if( def == "all" )
  {
    elxout << "  The transform is evaluated on all points. "
           << "The result is a deformation field." << std::endl;
    this->TransformPointsAllPoints();
  }
  else
  {
    // just a message
    elxout << "  The command-line option \"-def\" is not used, "
           << "so no points are transformed" << std::endl;
  }

} // end TransformPoints()


/**
 * ************** TransformPointsSomePoints *********************
 *
 * This function reads points from a file and transforms
 * these fixed-image coordinates to moving-image
 * coordinates.
 *
 * Reads the inputpoints from a text file, either as index or as point.
 * Computes the transformed points, converts them back to an index and compute
 * the deformation vector as the difference between the outputpoint and
 * the input point. Save the results.
 */

template< class TElastix >
void
TransformBase< TElastix >
::TransformPointsSomePoints( const std::string filename ) const
{
  /** Typedef's. */
  typedef typename FixedImageType::RegionType           FixedImageRegionType;
  typedef typename FixedImageType::PointType            FixedImageOriginType;
  typedef typename FixedImageType::SpacingType          FixedImageSpacingType;
  typedef typename FixedImageType::IndexType            FixedImageIndexType;
  typedef typename FixedImageIndexType::IndexValueType  FixedImageIndexValueType;
  typedef typename MovingImageType::IndexType           MovingImageIndexType;
  typedef typename MovingImageIndexType::IndexValueType MovingImageIndexValueType;
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
  std::vector< FixedImageIndexType >   inputindexvec(  nrofpoints );
  std::vector< InputPointType >        inputpointvec(  nrofpoints );
  std::vector< OutputPointType >       outputpointvec( nrofpoints );
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
          itk::Math::Round< double >( fixedcindex[ i ] ) );
      }
    }
  }
  else //so: inputasindex
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
          itk::Math::Round< double >( point[ i ] ) );
      }
      /** Compute the input point in physical coordinates. */
      dummyImage->TransformIndexToPhysicalPoint(
        inputindexvec[ j ], inputpointvec[ j ] );
    }
  }

  /** Apply the transform. */
  elxout << "  The input points are transformed." << std::endl;
  for( unsigned int j = 0; j < nrofpoints; j++ )
  {
    /** Call TransformPoint. */
    outputpointvec[ j ] = this->GetAsITKBaseType()->TransformPoint( inputpointvec[ j ] );

    /** Transform back to index in fixed image domain. */
    dummyImage->TransformPhysicalPointToContinuousIndex(
      outputpointvec[ j ], fixedcindex );
    for( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      outputindexfixedvec[ j ][ i ] = static_cast< FixedImageIndexValueType >(
        itk::Math::Round< double >( fixedcindex[ i ] ) );
    }

    if( alsoMovingIndices )
    {
      /** Transform back to index in moving image domain. */
      movingImage->TransformPhysicalPointToContinuousIndex(
        outputpointvec[ j ], movingcindex );
      for( unsigned int i = 0; i < MovingImageDimension; i++ )
      {
        outputindexmovingvec[ j ][ i ] = static_cast< MovingImageIndexValueType >(
          itk::Math::Round< double >( movingcindex[ i ] ) );
      }
    }

    /** Compute displacement. */
    deformationvec[ j ].CastFrom( outputpointvec[ j ] - inputpointvec[ j ] );
  }

  /** Create filename and file stream. */
  std::string outputPointsFileName = this->m_Configuration
    ->GetCommandLineArgument( "-out" );
  outputPointsFileName += "outputpoints.txt";
  std::ofstream outputPointsFile( outputPointsFileName.c_str() );
  outputPointsFile << std::showpoint << std::fixed;
  elxout << "  The transformed points are saved in: "
         <<  outputPointsFileName << std::endl;

  /** Print the results. */
  for( unsigned int j = 0; j < nrofpoints; j++ )
  {
    /** The input index. */
    outputPointsFile << "Point\t" << j << "\t; InputIndex = [ ";
    for( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      outputPointsFile << inputindexvec[ j ][ i ] << " ";
    }

    /** The input point. */
    outputPointsFile << "]\t; InputPoint = [ ";
    for( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      outputPointsFile << inputpointvec[ j ][ i ] << " ";
    }

    /** The output index in fixed image. */
    outputPointsFile << "]\t; OutputIndexFixed = [ ";
    for( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      outputPointsFile << outputindexfixedvec[ j ][ i ] << " ";
    }

    /** The output point. */
    outputPointsFile << "]\t; OutputPoint = [ ";
    for( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      outputPointsFile << outputpointvec[ j ][ i ] << " ";
    }

    /** The output point minus the input point. */
    outputPointsFile << "]\t; Deformation = [ ";
    for( unsigned int i = 0; i < MovingImageDimension; i++ )
    {
      outputPointsFile << deformationvec[ j ][ i ] << " ";
    }

    if( alsoMovingIndices )
    {
      /** The output index in moving image. */
      outputPointsFile << "]\t; OutputIndexMoving = [ ";
      for( unsigned int i = 0; i < MovingImageDimension; i++ )
      {
        outputPointsFile << outputindexmovingvec[ j ][ i ] << " ";
      }
    }

    outputPointsFile << "]" << std::endl;
  } // end for nrofpoints

} // end TransformPointsSomePoints()


/**
 * ************** TransformPointsSomePointsVTK *********************
 *
 * This function reads points from a .vtk file and transforms
 * these fixed-image coordinates to moving-image
 * coordinates.
 *
 * Reads the inputmesh from a vtk file, assuming world coordinates.
 * Computes the transformed points, save as outputpoints.vtk.
 */

template< class TElastix >
void
TransformBase< TElastix >
::TransformPointsSomePointsVTK( const std::string filename ) const
{
  /** Typedef's. \todo test DummyIPPPixelType=bool. */
  typedef float DummyIPPPixelType;
  typedef itk::DefaultStaticMeshTraits<
    DummyIPPPixelType, FixedImageDimension,
    FixedImageDimension, CoordRepType >                  MeshTraitsType;
  typedef itk::Mesh<
    DummyIPPPixelType, FixedImageDimension, MeshTraitsType > MeshType;
  typedef itk::MeshFileReader< MeshType > MeshReaderType;
  typedef itk::MeshFileWriter< MeshType > MeshWriterType;
  typedef itk::TransformMeshFilter<
    MeshType, MeshType, CombinationTransformType >       TransformMeshFilterType;

  /** Read the input points. */
  typename MeshReaderType::Pointer meshReader = MeshReaderType::New();
  meshReader->SetFileName( filename.c_str() );
  elxout << "  Reading input point file: " << filename << std::endl;
  try
  {
    meshReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    xl::xout[ "error" ] << "  Error while opening input point file." << std::endl;
    xl::xout[ "error" ] << err << std::endl;
  }

  /** Some user-feedback. */
  elxout << "  Input points are specified in world coordinates." << std::endl;
  unsigned long nrofpoints = meshReader->GetOutput()->GetNumberOfPoints();
  elxout << "  Number of specified input points: " << nrofpoints << std::endl;

  /** Apply the transform. */
  elxout << "  The input points are transformed." << std::endl;
  typename TransformMeshFilterType::Pointer meshTransformer = TransformMeshFilterType::New();
  meshTransformer->SetTransform(
    const_cast< CombinationTransformType * >( this->GetAsCombinationTransform() ) );
  meshTransformer->SetInput( meshReader->GetOutput() );
  try
  {
    meshTransformer->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    xl::xout[ "error" ] << "  Error while transforming points." << std::endl;
    xl::xout[ "error" ] << err << std::endl;
  }

  /** Create filename and file stream. */
  std::string outputPointsFileName = this->m_Configuration
    ->GetCommandLineArgument( "-out" );
  outputPointsFileName += "outputpoints.vtk";
  elxout << "  The transformed points are saved in: "
         <<  outputPointsFileName << std::endl;
  typename MeshWriterType::Pointer meshWriter = MeshWriterType::New();
  meshWriter->SetFileName( outputPointsFileName.c_str() );
  meshWriter->SetInput( meshTransformer->GetOutput() );

  try
  {
    meshWriter->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    xl::xout[ "error" ] << "  Error while saving points." << std::endl;
    xl::xout[ "error" ] << err << std::endl;
  }

} // end TransformPointsSomePointsVTK()


/**
 * ************** TransformPointsAllPoints **********************
 *
 * This function transforms all indexes to a physical point.
 * The difference vector (= the deformation at that index) is
 * stored in an image of vectors (of floats).
 */

template< class TElastix >
void
TransformBase< TElastix >
::TransformPointsAllPoints( void ) const
{
  /** Typedef's. */
  typedef typename FixedImageType::DirectionType FixedImageDirectionType;
  typedef itk::Vector<
    float, FixedImageDimension >                      VectorPixelType;
  typedef itk::Image<
    VectorPixelType, FixedImageDimension >            DeformationFieldImageType;
  typedef itk::TransformToDisplacementFieldFilter<
    DeformationFieldImageType, CoordRepType >         DeformationFieldGeneratorType;
  typedef itk::ChangeInformationImageFilter<
    DeformationFieldImageType >                       ChangeInfoFilterType;
  typedef itk::ImageFileWriter<
    DeformationFieldImageType >                       DeformationFieldWriterType;

  /** Create an setup deformation field generator. */
  typename DeformationFieldGeneratorType::Pointer defGenerator
    = DeformationFieldGeneratorType::New();
  defGenerator->SetSize(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetSize() );
  defGenerator->SetOutputSpacing(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputSpacing() );
  defGenerator->SetOutputOrigin(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputOrigin() );
  defGenerator->SetOutputStartIndex(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputStartIndex() );
  defGenerator->SetOutputDirection(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputDirection() );
  defGenerator->SetTransform( const_cast< const ITKBaseType * >( this->GetAsITKBaseType() ) );

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false. */
  typename ChangeInfoFilterType::Pointer infoChanger = ChangeInfoFilterType::New();
  FixedImageDirectionType originalDirection;
  bool                    retdc = this->GetElastix()->GetOriginalFixedImageDirection( originalDirection );
  infoChanger->SetOutputDirection( originalDirection );
  infoChanger->SetChangeDirection( retdc & !this->GetElastix()->GetUseDirectionCosines() );
  infoChanger->SetInput( defGenerator->GetOutput() );

  /** Track the progress of the generation of the deformation field. */
#ifndef _ELASTIX_BUILD_LIBRARY
  typename ProgressCommandType::Pointer progressObserver = ProgressCommandType::New();
  progressObserver->ConnectObserver( defGenerator );
  progressObserver->SetStartString( "  Progress: " );
  progressObserver->SetEndString( "%" );
#endif

  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter( resultImageFormat, "ResultImageFormat", 0, false );
  std::ostringstream makeFileName( "" );
  makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
               << "deformationField." << resultImageFormat;

  /** Write outputImage to disk. */
  typename DeformationFieldWriterType::Pointer defWriter
    = DeformationFieldWriterType::New();
  defWriter->SetInput( infoChanger->GetOutput() );
  defWriter->SetFileName( makeFileName.str().c_str() );

  /** Do the writing. */
  elxout << "  Computing and writing the deformation field ..." << std::endl;
  try
  {
    defWriter->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "TransformBase - TransformPointsAllPoints()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing deformation field image.\n";
    excp.SetDescription( err_str );

    /** Pass the exception to an higher level. */
    throw excp;
  }

} // end TransformPointsAllPoints()


/**
 * ************** ComputeDeterminantOfSpatialJacobian **********************
 */

template< class TElastix >
void
TransformBase< TElastix >
::ComputeDeterminantOfSpatialJacobian( void ) const
{
  /** If the optional command "-jac" is given in the command line arguments,
   * then and only then we continue.
   */
  std::string jac = this->GetConfiguration()->GetCommandLineArgument( "-jac" );
  if( jac == "" )
  {
    elxout << "  The command-line option \"-jac\" is not used, "
           << "so no det(dT/dx) computed." << std::endl;
    return;
  }
  else if( jac != "all" )
  {
    elxout << "  WARNING: The command-line option \"-jac\" should be used as \"-jac all\",\n"
           << "    but is specified as \"-jac " << jac << "\"\n"
           << "    Therefore det(dT/dx) is not computed." << std::endl;
    return;
  }

  /** Typedef's. */
  typedef itk::Image< float, FixedImageDimension > JacobianImageType;
  typedef itk::TransformToDeterminantOfSpatialJacobianSource<
    JacobianImageType, CoordRepType >                 JacobianGeneratorType;
  typedef itk::ImageFileWriter< JacobianImageType > JacobianWriterType;
  typedef itk::ChangeInformationImageFilter<
    JacobianImageType >                               ChangeInfoFilterType;
  typedef typename FixedImageType::DirectionType FixedImageDirectionType;

  /** Create an setup Jacobian generator. */
  typename JacobianGeneratorType::Pointer jacGenerator = JacobianGeneratorType::New();
  jacGenerator->SetTransform( const_cast< const ITKBaseType * >(
      this->GetAsITKBaseType() ) );
  jacGenerator->SetOutputSize(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetSize() );
  jacGenerator->SetOutputSpacing(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputSpacing() );
  jacGenerator->SetOutputOrigin(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputOrigin() );
  jacGenerator->SetOutputIndex(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputStartIndex() );
  jacGenerator->SetOutputDirection(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputDirection() );
  // NOTE: We can not use the following, since the fixed image does not exist in transformix
  //   jacGenerator->SetOutputParametersFromImage(
  //     this->GetRegistration()->GetAsITKBaseType()->GetFixedImage() );

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false. */
  typename ChangeInfoFilterType::Pointer infoChanger = ChangeInfoFilterType::New();
  FixedImageDirectionType originalDirection;
  bool                    retdc = this->GetElastix()->GetOriginalFixedImageDirection( originalDirection );
  infoChanger->SetOutputDirection( originalDirection );
  infoChanger->SetChangeDirection( retdc & !this->GetElastix()->GetUseDirectionCosines() );
  infoChanger->SetInput( jacGenerator->GetOutput() );
#ifndef _ELASTIX_BUILD_LIBRARY
  /** Track the progress of the generation of the deformation field. */
  typename ProgressCommandType::Pointer progressObserver = ProgressCommandType::New();
  progressObserver->ConnectObserver( jacGenerator );
  progressObserver->SetStartString( "  Progress: " );
  progressObserver->SetEndString( "%" );
#endif
  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter( resultImageFormat, "ResultImageFormat", 0, false );
  std::ostringstream makeFileName( "" );
  makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
               << "spatialJacobian." << resultImageFormat;

  /** Write outputImage to disk. */
  typename JacobianWriterType::Pointer jacWriter = JacobianWriterType::New();
  jacWriter->SetInput( infoChanger->GetOutput() );
  jacWriter->SetFileName( makeFileName.str().c_str() );

  /** Do the writing. */
  elxout << "  Computing and writing the spatial Jacobian determinant..." << std::endl;
  try
  {
    jacWriter->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "TransformBase - ComputeDeterminantOfSpatialJacobian()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing spatial Jacobian determinant image.\n";
    excp.SetDescription( err_str );

    /** Pass the exception to an higher level. */
    throw excp;
  }

} // end ComputeDeterminantOfSpatialJacobian()


/**
 * ************** ComputeSpatialJacobian **********************
 */

template< class TElastix >
void
TransformBase< TElastix >
::ComputeSpatialJacobian( void ) const
{
  /** If the optional command "-jacmat" is given in the command line arguments,
   * then and only then we continue.
   */
  std::string jac = this->GetConfiguration()->GetCommandLineArgument( "-jacmat" );
  if( jac != "all" )
  {
    elxout << "  The command-line option \"-jacmat\" is not used, "
           << "so no dT/dx computed." << std::endl;
    return;
  }

  /** Typedef's. */
  typedef float SpatialJacobianComponentType;
  typedef itk::Matrix< SpatialJacobianComponentType,
    MovingImageDimension, FixedImageDimension >        OutputSpatialJacobianType;
  typedef itk::Image< OutputSpatialJacobianType,
    FixedImageDimension >                              JacobianImageType;
  typedef itk::TransformToSpatialJacobianSource<
    JacobianImageType, CoordRepType >                 JacobianGeneratorType;
  typedef itk::ImageFileWriter< JacobianImageType > JacobianWriterType;
  typedef itk::ChangeInformationImageFilter<
    JacobianImageType >                               ChangeInfoFilterType;
  typedef typename FixedImageType::DirectionType FixedImageDirectionType;
  typedef itk::PixelTypeChangeCommand<
    JacobianWriterType >                              PixelTypeChangeCommandType;

  /** Create an setup Jacobian generator. */
  typename JacobianGeneratorType::Pointer jacGenerator = JacobianGeneratorType::New();
  jacGenerator->SetTransform( const_cast< const ITKBaseType * >(
      this->GetAsITKBaseType() ) );
  jacGenerator->SetOutputSize(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetSize() );
  jacGenerator->SetOutputSpacing(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputSpacing() );
  jacGenerator->SetOutputOrigin(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputOrigin() );
  jacGenerator->SetOutputIndex(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputStartIndex() );
  jacGenerator->SetOutputDirection(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputDirection() );
  // NOTE: We can not use the following, since the fixed image does not exist in transformix
  //   jacGenerator->SetOutputParametersFromImage(
  //     this->GetRegistration()->GetAsITKBaseType()->GetFixedImage() );

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false.
   */
  typename ChangeInfoFilterType::Pointer infoChanger = ChangeInfoFilterType::New();
  FixedImageDirectionType originalDirection;
  bool                    retdc = this->GetElastix()->GetOriginalFixedImageDirection( originalDirection );
  infoChanger->SetOutputDirection( originalDirection );
  infoChanger->SetChangeDirection( retdc & !this->GetElastix()->GetUseDirectionCosines() );
  infoChanger->SetInput( jacGenerator->GetOutput() );
#ifndef _ELASTIX_BUILD_LIBRARY
  /** Track the progress of the generation of the deformation field. */
  typename ProgressCommandType::Pointer progressObserver = ProgressCommandType::New();
  progressObserver->ConnectObserver( jacGenerator );
  progressObserver->SetStartString( "  Progress: " );
  progressObserver->SetEndString( "%" );
#endif
  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter( resultImageFormat, "ResultImageFormat", 0, false );
  std::ostringstream makeFileName( "" );
  makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
               << "fullSpatialJacobian." << resultImageFormat;

  /** Write outputImage to disk. */
  typename JacobianWriterType::Pointer jacWriter = JacobianWriterType::New();
  jacWriter->SetInput( infoChanger->GetOutput() );
  jacWriter->SetFileName( makeFileName.str().c_str() );
  /** Hack to change the pixel type to vector. Not necessary for mhd. */
  typename PixelTypeChangeCommandType::Pointer jacStartWriteCommand
    = PixelTypeChangeCommandType::New();
  if( resultImageFormat != "mhd" )
  {
    jacWriter->AddObserver( itk::StartEvent(), jacStartWriteCommand );
  }

  /** Do the writing. */
  elxout << "  Computing and writing the spatial Jacobian..." << std::endl;
  try
  {
    jacWriter->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "TransformBase - ComputeSpatialJacobian()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing spatial Jacobian image.\n";
    excp.SetDescription( err_str );

    /** Pass the exception to an higher level. */
    throw excp;
  }

} // end ComputeSpatialJacobian()


/**
 * ************** SetTransformParametersFileName ****************
 */

template< class TElastix >
void
TransformBase< TElastix >
::SetTransformParametersFileName( const char * filename )
{
  /** Copied from itkSetStringMacro. */
  if( filename && ( filename == this->m_TransformParametersFileName ) )
  {
    return;
  }
  if( filename )
  {
    this->m_TransformParametersFileName = filename;
  }
  else
  {
    this->m_TransformParametersFileName = "";
  }
  ObjectType * thisAsObject = dynamic_cast< ObjectType * >( this );
  thisAsObject->Modified();

} // end SetTransformParametersFileName()


/**
 * ************** SetReadWriteTransformParameters ***************
 */

template< class TElastix >
void
TransformBase< TElastix >
::SetReadWriteTransformParameters( const bool _arg )
{
  /** Copied from itkSetMacro. */
  if( this->m_ReadWriteTransformParameters != _arg  )
  {
    this->m_ReadWriteTransformParameters = _arg;
    ObjectType * thisAsObject = dynamic_cast< ObjectType * >( this );
    thisAsObject->Modified();
  }

} // end SetReadWriteTransformParameters()


/**
 * ************** AutomaticScalesEstimation ***************
 */

template< class TElastix >
void
TransformBase< TElastix >
::AutomaticScalesEstimation( ScalesType & scales ) const
{
  typedef itk::ImageGridSampler< FixedImageType > ImageSamplerType;
  typedef typename ImageSamplerType::Pointer      ImageSamplerPointer;
  typedef typename
    ImageSamplerType::ImageSampleContainerType ImageSampleContainerType;
  typedef typename ImageSampleContainerType::Pointer       ImageSampleContainerPointer;
  typedef typename ITKBaseType::JacobianType               JacobianType;
  typedef typename ITKBaseType::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  const ITKBaseType * const thisITK = this->GetAsITKBaseType();
  const unsigned int        outdim  = MovingImageDimension;
  const unsigned int        N       = thisITK->GetNumberOfParameters();
  scales = ScalesType( N );

  /** Set up grid sampler. */
  ImageSamplerPointer sampler = ImageSamplerType::New();
  sampler->SetInput(
    this->GetRegistration()->GetAsITKBaseType()->GetFixedImage() );
  sampler->SetInputImageRegion(
    this->GetRegistration()->GetAsITKBaseType()->GetFixedImageRegion() );

  /** Compute the grid spacing. */
  unsigned long nrofsamples = 10000;
  sampler->SetNumberOfSamples( nrofsamples );

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  ImageSampleContainerPointer sampleContainer = sampler->GetOutput();
  nrofsamples = sampleContainer->Size();
  if( nrofsamples == 0 )
  {
    /** \todo: should we demand a minimum number (~100) of voxels? */
    itkExceptionMacro( << "No valid voxels found to estimate the scales." );
  }

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end   = sampleContainer->End();

  /** initialize */
  scales.Fill( 0.0 );

  /** Read fixed coordinates and get Jacobian. */
  for( iter = begin; iter != end; ++iter )
  {
    const InputPointType & point = ( *iter ).Value().m_ImageCoordinates;
    //const JacobianType & jacobian = thisITK->GetJacobian( point );
    JacobianType jacobian; NonZeroJacobianIndicesType nzji;
    thisITK->GetJacobian( point, jacobian, nzji );

    /** Square each element of the Jacobian and add each row
     * to the newscales.
     */
    for( unsigned int d = 0; d < outdim; ++d )
    {
      ScalesType jacd( jacobian[ d ], N, false );
      scales += element_product( jacd, jacd );
    }
  }
  scales /= static_cast< double >( nrofsamples );

} // end AutomaticScalesEstimation()


/**
 * ************** AutomaticScalesEstimationStackTransform ***************
 */

template <class TElastix>
void
TransformBase<TElastix>
::AutomaticScalesEstimationStackTransform(
  const unsigned int & numberOfSubTransforms, ScalesType & scales ) const
{
  typedef typename FixedImageType::RegionType               FixedImageRegionType;
  typedef typename FixedImageType::IndexType                FixedImageIndexType;
  typedef typename FixedImageType::SizeType                 SizeType;

  typedef itk::ImageGridSampler< FixedImageType >           ImageSamplerType;
  typedef typename ImageSamplerType::Pointer                ImageSamplerPointer;
  typedef typename
      ImageSamplerType::ImageSampleContainerType            ImageSampleContainerType;
  typedef typename ImageSampleContainerType::Pointer        ImageSampleContainerPointer;
  typedef typename ITKBaseType::JacobianType                JacobianType;
  typedef typename ITKBaseType::NonZeroJacobianIndicesType  NonZeroJacobianIndicesType;

  const ITKBaseType * const thisITK = this->GetAsITKBaseType();
  const unsigned int outdim = FixedImageDimension;
  const unsigned int N = thisITK->GetNumberOfParameters();

  /** initialize */
  scales = ScalesType( N );
  scales.Fill( 0.0 );

  /** Get fixed image region from registration. */
  const FixedImageRegionType & inputRegion = this->GetRegistration()->GetAsITKBaseType()->GetFixedImageRegion();
  SizeType size = inputRegion.GetSize();

  /** Set desired extraction region. */
  FixedImageIndexType start = inputRegion.GetIndex();
  start[ FixedImageDimension - 1 ] = size[ FixedImageDimension - 1 ] - 1;

 /** Set size of last dimension to 0. */
  size[ FixedImageDimension - 1 ] = 0;

  elxout << "start region for scales: " << start << std::endl;
  elxout << "size region for scales: " << size << std::endl;

  FixedImageRegionType desiredRegion;
  desiredRegion.SetSize( size );
  desiredRegion.SetIndex( start );

  /** Set up the grid sampler. */
  ImageSamplerPointer sampler = ImageSamplerType::New();
  sampler->SetInput( this->GetRegistration()->GetAsITKBaseType()->GetFixedImage() );
  sampler->SetInputImageRegion( desiredRegion );

  /** Compute the grid spacing. */
  unsigned long nrofsamples = 10000;
  sampler->SetNumberOfSamples( nrofsamples );

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  ImageSampleContainerPointer sampleContainer = sampler->GetOutput();
  nrofsamples = sampleContainer->Size();
  if( nrofsamples == 0 )
  {
    /** \todo: should we demand a minimum number (~100) of voxels? */
    itkExceptionMacro( << "No valid voxels found to estimate the scales." );
  }

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  /** Read fixed coordinates and get Jacobian. */
  JacobianType jacobian;
  NonZeroJacobianIndicesType nzji;
  for( iter = begin; iter != end; ++iter )
  {
    const InputPointType & point = (*iter).Value().m_ImageCoordinates;
    //const JacobianType & jacobian = thisITK->GetJacobian( point );
    thisITK->GetJacobian( point, jacobian, nzji );

    /** Square each element of the Jacobian and add each row to the new scales. */
    for( unsigned int d = 0; d < outdim; ++d )
    {
      ScalesType jacd( jacobian[ d ], N, false );
      scales += element_product( jacd, jacd );
    }
  }
  scales /= static_cast<double>( nrofsamples );

  const unsigned int numberOfScalesSubTransform = N / numberOfSubTransforms; //(FixedImageDimension)*(FixedImageDimension - 1);

  for( unsigned int i = 0; i < N; i += numberOfScalesSubTransform )
  {
    for( unsigned int j = 0; j < numberOfScalesSubTransform; ++j )
    {
      scales( i + j ) = scales( j );
    }
  }

} // end AutomaticScalesEstimationStackTransform()


} // end namespace elastix

#endif // end #ifndef __elxTransformBase_hxx
