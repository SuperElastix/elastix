/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxTransformBase_hxx
#define __elxTransformBase_hxx

#include "elxTransformBase.h"

#include "itkPointSet.h"
#include "itkDefaultStaticMeshTraits.h"
#include "itkTransformixInputPointFileReader.h"
#include "vnl/vnl_math.h"
#include <itksys/SystemTools.hxx>
#include "itkVector.h"
#include "itkTransformToDeformationFieldSource.h"
#include "itkImageFileWriter.h"
#include "itkImageGridSampler.h"
#include "itkContinuousIndex.h"

namespace elastix
{
  //using namespace itk; //Not here because the ITK also started to define a TransformBase class....


/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
TransformBase<TElastix>::TransformBase()
{
  /** Initialize. */
  this->m_TransformParametersPointer = 0;
  this->m_ConfigurationInitialTransform = 0;
  this->m_ReadWriteTransformParameters = true;

} // end Constructor()


/**
 * ********************** Destructor ****************************
 */

template <class TElastix>
TransformBase<TElastix>::~TransformBase()
{
  /** Delete. */
  if ( this->m_TransformParametersPointer )
  {
    delete this->m_TransformParametersPointer;
  }

} // end Destructor()


/**
 * ******************** BeforeAllBase ***************************
 */

template <class TElastix>
int TransformBase<TElastix>
::BeforeAllBase( void )
{
  /** Check Command line options and print them to the logfile. */
  elxout << "Command line options from TransformBase:" << std::endl;
  std::string check( "" );

  /** Check for appearance of "-t0". */
  check = this->m_Configuration->GetCommandLineArgument( "-t0" );
  if ( check.empty() )
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

template <class TElastix>
int TransformBase<TElastix>
::BeforeAllTransformix( void )
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Declare check. */
  std::string check = "";

  /** Check for appearance of "-ipp". */
  check = this->m_Configuration->GetCommandLineArgument( "-ipp" );
  if ( check == "" )
  {
    elxout << "-ipp      unspecified, so no inputpoints transformed" << std::endl;
  }
  else
  {
    elxout << "-ipp      " << check << std::endl;
  }

  /** Return a value. */
  return returndummy;

} // end BeforeAllTransformix()


/**
 * ******************* BeforeRegistrationBase *******************
 */

template <class TElastix>
void TransformBase<TElastix>
::BeforeRegistrationBase( void )
{ 
  /** Read from the configuration file how to combine the initial
   * transform with the current transform.
   */
  std::string howToCombineTransforms = "Add";
  this->m_Configuration->ReadParameter( howToCombineTransforms,
    "HowToCombineTransforms", 0, true );

  /** Check if this is a CombinationTransform. */
  CombinationTransformType * thisAsGrouper = 
    dynamic_cast< CombinationTransformType * >( this );
  if ( thisAsGrouper )
  {
    if ( howToCombineTransforms == "Compose" )
    {
      thisAsGrouper->SetUseComposition( true );
    }
    else
    {
      thisAsGrouper->SetUseComposition( false );
    }
  }

  /** Check if this is a CombinationTransform. */
  AdvancedCombinationTransformType * thisAsGrouper2 = 
    dynamic_cast< AdvancedCombinationTransformType * >( this );
  if ( thisAsGrouper2 )
  {
    if ( howToCombineTransforms == "Compose" )
    {
      thisAsGrouper2->SetUseComposition( true );
    }
    else
    {
      thisAsGrouper2->SetUseComposition( false );
    }
  }

  /** Set the initial transform. Elastix returns an itkObject, so 
   * try to cast it to an InitialTransformType, which is of type itk::Transform.
   * No need to cast to InitialAdvancedTransformType, since itk::AdvancedTransform
   * is a Superclass of itk::Transform.
   */
  if ( this->m_Elastix->GetInitialTransform() )
  {
    InitialTransformType * testPointer =
      dynamic_cast<InitialTransformType* >(
      this->m_Elastix->GetInitialTransform()  );
    if ( testPointer )
    {
      this->SetInitialTransform( testPointer );
    }
  }
  else
  {
    std::string fileName =  this->m_Configuration->GetCommandLineArgument( "-t0" );
    if ( !fileName.empty() )
    {
      if ( itksys::SystemTools::FileExists( fileName.c_str() ) )
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

template <class TElastix>
const typename TransformBase<TElastix>::InitialTransformType * 
TransformBase<TElastix>
::GetInitialTransform( void ) const
{
  /** Cast to a(n Advanced)CombinationTransform. */
  const CombinationTransformType * thisAsGrouper
    = dynamic_cast< const CombinationTransformType * >( this );
  const AdvancedCombinationTransformType * thisAsGrouper2
    = dynamic_cast< const AdvancedCombinationTransformType * >( this );

  /** Set the initial transform. */
  if ( thisAsGrouper )
  {
    return thisAsGrouper->GetInitialTransform();
  }
  else if ( thisAsGrouper2 )
  {
    const InitialTransformType * initialTransform
      = dynamic_cast< const InitialTransformType * >(
      thisAsGrouper2->GetInitialTransform() );
    if ( initialTransform )
    {
      return initialTransform;
    }
    return 0;
  }

  return 0;

} // end GetInitialTransform()


/**
 * ******************* SetInitialTransform **********************
 */

template <class TElastix>
void TransformBase<TElastix>
::SetInitialTransform( InitialTransformType * _arg )
{
  /** Cast to a(n Advanced)CombinationTransform. */
  CombinationTransformType * thisAsGrouper
    = dynamic_cast< CombinationTransformType * >( this );
  AdvancedCombinationTransformType * thisAsGrouper2
    = dynamic_cast< AdvancedCombinationTransformType * >( this );

  /** Set initial transform. */
  if ( thisAsGrouper )
  {
    thisAsGrouper->SetInitialTransform( _arg );
  }
  else if ( thisAsGrouper2 )
  {
    InitialAdvancedTransformType * _arg2
      = dynamic_cast< InitialAdvancedTransformType * >( _arg );
    if ( _arg2 )
    {
      thisAsGrouper2->SetInitialTransform( _arg2 );
    }
  }

  // \todo AdvancedCombinationTransformType

} // end SetInitialTransform()


/**
 * ******************* SetFinalParameters ********************
 */

template <class TElastix>
void TransformBase<TElastix>
::SetFinalParameters( void )
{
  /** Make a local copy, since some transforms do not do this,
   * like the BSpline-transform.
   */
  //m_FinalParameters = this->m_Registration->GetAsITKBaseType()->
  //GetLastTransformParameters();
  this->m_FinalParameters = this->GetElastix()->GetElxOptimizerBase()
    ->GetAsITKBaseType()->GetCurrentPosition();

  /** Set the final Parameters for the resampler. */
  this->GetAsITKBaseType()->SetParameters( this->m_FinalParameters );

} // end SetFinalParameters()


/**
 * ******************* AfterRegistrationBase ********************
 */

template <class TElastix>
void TransformBase<TElastix>
::AfterRegistrationBase( void )
{
  /** Set the final Parameters. */
  this->SetFinalParameters();

} // end AfterRegistrationBase()


/**
 * ******************* ReadFromFile *****************************
 */

template <class TElastix>
void TransformBase<TElastix>
::ReadFromFile( void )
{
  /**
   * This method assumes this->m_Configuration is initialized with a
   * transform parameter file, so not an elastix parameter file!!
   */

  /** Task 1 - Read the parameters from file. */

  /** Get the number of TransformParameters. */
  unsigned int NumberOfParameters = 0;
  this->m_Configuration->ReadParameter( NumberOfParameters, "NumberOfParameters", 0 );

  if ( this->m_ReadWriteTransformParameters )
  {
    /** Get the TransformParameters. */
    if ( this->m_TransformParametersPointer ) delete this->m_TransformParametersPointer;
    this->m_TransformParametersPointer = new ParametersType( NumberOfParameters );

    /** If NumberOfParameters < 20, we read in the normal way. */
    if ( NumberOfParameters < 20 )
    {     
      for ( unsigned int i = 0; i < NumberOfParameters; i++ )
      {
        this->m_Configuration->ReadParameter(
          (*(this->m_TransformParametersPointer))[ i ], "TransformParameters", i );
      }
    }
    /** Else, do the reading more 'manually'.
     * This is necessary, because the ReadParameter can not handle
     * many parameters.
     */
    else
    {
      std::ifstream input( this->GetConfiguration()->GetCommandLineArgument( "-tp" ) );
      if ( input.is_open() )
      {
        /** Search for the following pattern:
        *
        * // (TransformParameters)
        * // 1.0 435.0 etc... 
        *
        */
        bool found = false;
        std::string teststring;
        while ( !found && !input.eof() )
        {
          input >> teststring;
          if ( teststring == "//" )
          {
            input >> teststring;
            if ( teststring == "(TransformParameters)" )
            { 
              input >> teststring;
              if ( teststring == "//" )
              {
                found = true;
              }
            }
          }
        } // end while
        if ( found )
        {
          for ( unsigned int i = 0; i < NumberOfParameters; i++ )
          {
            input >> (*(this->m_TransformParametersPointer))[ i ];
          }
        }
        else
        {
          xl::xout["error"]
          << "Invalid transform parameter file! The parameters could not be found."
            << std::endl;
          itkGenericExceptionMacro( << "Error during reading the transform parameter file!" );
        }
        input.close();
      } // end if input-file is open
      else
      {
        xl::xout["error"] <<
          "The transform parameter file could not opened!" << std::endl;
        itkGenericExceptionMacro( << "Error during reading the transform parameter file!" );
      }
    } // end else

    /** Set the parameters into this transform. */
    this->GetAsITKBaseType()->SetParameters( *(this->m_TransformParametersPointer) );
  } // end if this->m_ReadWriteTransformParameters

  /** Task 2 - Get the InitialTransform. */

  /** Get the InitialTransformName. */
  std::string fileName = "NoInitialTransform";
  this->m_Configuration->ReadParameter( fileName,
    "InitialTransformParametersFileName", 0 );

  /** Call the function ReadInitialTransformFromFile. */
  if ( fileName != "NoInitialTransform" )
  {
    /** Check if the initial transform of this transform parameter file
     * is not the same as this transform parameter file. Otherwise,
     * we will have an infinite loop.
     */
    std::string fullFileName1 = itksys::SystemTools::CollapseFullPath( fileName.c_str() );
    std::string fullFileName2 = itksys::SystemTools::CollapseFullPath( 
      this->GetConfiguration()->GetCommandLineArgument( "-tp" ) );
    if ( fullFileName1 == fullFileName2 )
    {
      itkExceptionMacro( << "ERROR: The InitialTransformParametersFileName "
        << "is identical to the current TransformParameters filename! "
        << "An infinite loop is not allowed." );
    }

    /** We can safely read the initial transform. */
    this->ReadInitialTransformFromFile( fileName.c_str() );
  } 

  /** Task 3 - Read from the configuration file how to combine the
   * initial transform with the current transform.
   */
  std::string howToCombineTransforms = "Add"; // default
  this->m_Configuration->ReadParameter( howToCombineTransforms,
    "HowToCombineTransforms", 0, true );

  /** Convert 'this' to a pointer to a CombinationTransform and set how
   * to combine the current transform with the initial transform.
   */
  CombinationTransformType * thisAsGrouper = 
    dynamic_cast< CombinationTransformType * >( this );
  AdvancedCombinationTransformType * thisAsGrouper2 = 
    dynamic_cast< AdvancedCombinationTransformType * >( this );
  if ( thisAsGrouper )
  {
    if ( howToCombineTransforms == "Compose" )
    {
      thisAsGrouper->SetUseComposition( true );
    }
    else
    {
      thisAsGrouper->SetUseComposition( false );
    }
  }
  else if ( thisAsGrouper2 )
  {
    if ( howToCombineTransforms == "Compose" )
    {
      thisAsGrouper2->SetUseComposition( true );
    }
    else
    {
      thisAsGrouper2->SetUseComposition( false );
    }
  }

  /** Task 4 - Remember the name of the TransformParametersFileName.
   * This will be needed when another transform will use this transform
   * as an initial transform (see the WriteToFile method)
   */
  this->SetTransformParametersFileName(
    this->GetConfiguration()->GetCommandLineArgument( "-tp" ) );

} // end ReadFromFile()


/**
 * ******************* ReadInitialTransformFromFile *************
 */

template <class TElastix>
void TransformBase<TElastix>
::ReadInitialTransformFromFile( const char * transformParametersFileName )
{
  /** Create a new configuration, which will be initialized with
   * the transformParameterFileName.
   */
  if ( !(this->m_ConfigurationInitialTransform) )
  {
    this->m_ConfigurationInitialTransform = ConfigurationType::New();
  }

  /** Create argmapInitialTransform. */
  ArgumentMapType argmapInitialTransform;
  argmapInitialTransform.insert( ArgumentMapEntryType(
    "-tp", transformParametersFileName ) );

  int dummy = this->m_ConfigurationInitialTransform->Initialize(
    argmapInitialTransform );

  /** Read the InitialTransform name. */
  ComponentDescriptionType InitialTransformName = "AffineTransform";
  this->m_ConfigurationInitialTransform->ReadParameter(
    InitialTransformName, "Transform", 0 );

  /** Create an InitialTransform. */
  ObjectType::Pointer initialTransform;

  PtrToCreator testcreator = 0;
  testcreator = this->GetElastix()->GetComponentDatabase()
    ->GetCreator( InitialTransformName, this->m_Elastix->GetDBIndex() );
  initialTransform = testcreator ? testcreator() : NULL;

  Self * elx_initialTransform = dynamic_cast< Self * >(
    initialTransform.GetPointer() );      

  /** Call the ReadFromFile method of the initialTransform. */
  if ( elx_initialTransform )
  {
    //elx_initialTransform->SetTransformParametersFileName(transformParametersFileName);
    elx_initialTransform->SetElastix( this->GetElastix() );
    elx_initialTransform->SetConfiguration( this->m_ConfigurationInitialTransform );      
    elx_initialTransform->ReadFromFile();

    /** Set initial transform. */
    InitialTransformType * testPointer =
      dynamic_cast<InitialTransformType* >( initialTransform.GetPointer() );
    if ( testPointer )
    {
      this->SetInitialTransform( testPointer );
    }

  } // end if

} // end ReadInitialTransformFromFile()


/**
 * ******************* WriteToFile ******************************
 */

template <class TElastix>
void TransformBase<TElastix>
::WriteToFile( void ) const
{
  /** Write the current set parameters to file. */
  this->WriteToFile( this->GetAsITKBaseType()->GetParameters() );

} // end WriteToFile()


/**
 * ******************* WriteToFile ******************************
 */

template <class TElastix>
void TransformBase<TElastix>
::WriteToFile( const ParametersType & param ) const
{
  using namespace xl;

  /** Write the name of this transform. */
  xout["transpar"] << "(Transform \""
    << this->elxGetClassName() << "\")" << std::endl;

  /** Get the number of parameters of this transform. */
  unsigned int nrP = param.GetSize();

  /** Write the number of parameters of this transform. */
  xout["transpar"] << "(NumberOfParameters "
    << nrP << ")" << std::endl;

  /** Write the parameters of this transform. */
  if ( this->m_ReadWriteTransformParameters )
  {
    if ( nrP < 20 )
    {
      /** In this case, write in a normal way to the parameter file. */
      xout["transpar"] << "(TransformParameters ";
      for ( unsigned int i = 0; i < nrP - 1; i++ )
      {
        xout["transpar"] << param[ i ] << " ";
      }
      xout["transpar"] << param[ nrP - 1 ] << ")" << std::endl;
    }
    else
    {
      /** Otherwise, write to parameter file with "// " in front of it.
       * This is necessary, because the ReadParameter can not handle
       * many parameters.
       */
      xout["transpar"] << "// (TransformParameters)" << std::endl << "// ";
      for ( unsigned int i = 0; i < nrP - 1; i++ )
      {
        xout["transpar"] << param[ i ] << " ";
      }
      xout["transpar"] << param[ nrP -1 ] << std::endl;
    }
  } // end if this->m_WriteTransformParameters

  /** Write the name of the parameters-file of the initial transform. */
  if ( this->GetInitialTransform() )
  {
    xout["transpar"] << "(InitialTransformParametersFileName \""
      << (dynamic_cast<const Self *>( this->GetInitialTransform() ))
      ->GetTransformParametersFileName() << "\")" << std::endl;
  }
  else
  {
    xout["transpar"]
      << "(InitialTransformParametersFileName \"NoInitialTransform\")"
      << std::endl;
  }

  /** Write the way Transforms are combined. */
  std::string combinationMethod = "Add";
  const CombinationTransformType * dummyComboTransform
    = dynamic_cast< const CombinationTransformType * >( this );
  if ( dummyComboTransform )
  {
    if ( dummyComboTransform->GetUseComposition() )
    {
      combinationMethod = "Compose";
    }
  }
  const AdvancedCombinationTransformType * dummyComboTransform2
    = dynamic_cast< const AdvancedCombinationTransformType * >( this );
  if ( dummyComboTransform2 )
  {
    if ( dummyComboTransform2->GetUseComposition() )
    {
      combinationMethod = "Compose";
    }
  }

  xout["transpar"] << "(HowToCombineTransforms \""
    << combinationMethod << "\")" << std::endl;

  /** Write image specific things. */
  xout["transpar"] << std::endl << "// Image specific" << std::endl;

  /** Write image dimensions. */
  unsigned int FixDim = FixedImageDimension;
  unsigned int MovDim = MovingImageDimension;
  xout["transpar"] << "(FixedImageDimension "
    << FixDim << ")" << std::endl;
  xout["transpar"] << "(MovingImageDimension "
    << MovDim << ")" << std::endl;

  /** Write image pixel types. */
  std::string fixpix = "float";
  std::string movpix = "float";
  this->m_Configuration->ReadParameter( fixpix, "FixedInternalImagePixelType", 0 );
  this->m_Configuration->ReadParameter( movpix, "MovingInternalImagePixelType", 0 );
  xout["transpar"] << "(FixedInternalImagePixelType \""
    << fixpix << "\")" << std::endl;
  xout["transpar"] << "(MovingInternalImagePixelType \""
    << movpix << "\")" << std::endl;

  /** Get the Size, Spacing and Origin of the fixed image. */
  typedef typename FixedImageType::SizeType                 FixedImageSizeType;
  typedef typename FixedImageType::IndexType                FixedImageIndexType;
  typedef typename FixedImageType::SpacingType              FixedImageSpacingType;
  typedef typename FixedImageType::PointType                FixedImageOriginType;
  typedef typename FixedImageType::DirectionType            FixedImageDirectionType;
  FixedImageSizeType size = 
    this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  FixedImageIndexType index = 
    this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetIndex();
  FixedImageSpacingType spacing =
    this->m_Elastix->GetFixedImage()->GetSpacing();
  FixedImageOriginType origin = 
    this->m_Elastix->GetFixedImage()->GetOrigin();
  FixedImageDirectionType direction = 
    this->m_Elastix->GetFixedImage()->GetDirection();

  /** Write image Size. */
  xout["transpar"] << "(Size ";
  for ( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
  {
    xout["transpar"] << size[ i ] << " ";
  }
  xout["transpar"] << size[ FixedImageDimension - 1 ] << ")" << std::endl;

  /** Write image Index. */
  xout["transpar"] << "(Index ";
  for ( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
  {
    xout["transpar"] << index[ i ] << " ";
  }
  xout["transpar"] << index[ FixedImageDimension - 1 ] << ")" << std::endl;

  /** Set the precision of cout to 2, because Spacing and
   * Origin must have at least one digit precision.
   */
  xout["transpar"] << std::setprecision(10);

  /** Write image Spacing. */
  xout["transpar"] << "(Spacing ";
  for ( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
  {
    xout["transpar"] << spacing[ i ] << " ";
  }
  xout["transpar"] << spacing[ FixedImageDimension - 1 ] << ")" << std::endl;

  /** Write image Origin. */
  xout["transpar"] << "(Origin ";
  for ( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
  {
    xout["transpar"] << origin[ i ] << " ";
  }
  xout["transpar"] << origin[ FixedImageDimension - 1 ] << ")" << std::endl;

  /** Write direction cosines. */
  xout["transpar"] << "(Direction";
  for ( unsigned int i = 0; i < FixedImageDimension; i++ )
  {
    for ( unsigned int j = 0; j < FixedImageDimension; j++ )
    {
      xout["transpar"] << " " << direction( j, i );
    }
  }
  xout["transpar"] << ")" << std::endl;

  /** Set the precision back to default value. */
  xout["transpar"] << std::setprecision( this->m_Elastix->GetDefaultOutputPrecision() );

} // end WriteToFile()


/**
 * ******************* TransformPoints **************************
 *
 * This function reads points from a file (but only if requested)
 * and transforms these fixed-image coordinates to moving-image
 * coordinates.
 */

template <class TElastix>
void TransformBase<TElastix>
::TransformPoints( void ) const
{
  /** If the optional command "-ipp" is given in the command
   * line arguments, then and only then we continue.
   */
  std::string ipp = this->GetConfiguration()->GetCommandLineArgument( "-ipp" );

  /** If there is an inputpoint-file? */
  if ( ipp != "" && ipp != "all" )
  {
    elxout << "  The transform is evaluated on some points, "
      << "specified in the input point file." << std::endl;
    this->TransformPointsSomePoints( ipp );
  }
  else if ( ipp == "all" )
  {
    elxout << "  The transform is evaluated on all points. "
      << "The result is a deformation field." << std::endl;
    this->TransformPointsAllPoints();
  }
  else
  {
    // just a message
    elxout << "  The command-line option \"-ipp\" is not used, "
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

template <class TElastix>
void TransformBase<TElastix>
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
    itk::ContinuousIndex<double, FixedImageDimension>   FixedImageContinuousIndexType;
  typedef 
    itk::ContinuousIndex<double, MovingImageDimension>  MovingImageContinuousIndexType;

  typedef bool                                          DummyIPPPixelType;
  typedef itk::DefaultStaticMeshTraits<
    DummyIPPPixelType, FixedImageDimension,
    FixedImageDimension, CoordRepType>                  MeshTraitsType;
  typedef itk::PointSet< DummyIPPPixelType, 
    FixedImageDimension, MeshTraitsType>                PointSetType;
  typedef itk::TransformixInputPointFileReader<
    PointSetType >                                      IPPReaderType;
  typedef itk::Vector< float, FixedImageDimension >     DeformationVectorType;

  /** Construct an ipp-file reader. */
  typename IPPReaderType::Pointer ippReader = IPPReaderType::New();
  ippReader->SetFileName( filename.c_str() );

  /** Read the input points. */
  elxout << "  Reading input point file: " << filename << std::endl;
  try
  {
    ippReader->Update();
  }
  catch (itk::ExceptionObject & err)
  { 
    xl::xout["error"] << "  Error while opening input point file." << std::endl;
    xl::xout["error"] << err << std::endl;      
  }

  /** Some user-feedback. */
  if ( ippReader->GetPointsAreIndices() )
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
  std::vector< FixedImageIndexType >    inputindexvec(  nrofpoints ); 
  std::vector< InputPointType >         inputpointvec(  nrofpoints );
  std::vector< OutputPointType >        outputpointvec( nrofpoints );
  std::vector< FixedImageIndexType >    outputindexfixedvec( nrofpoints );
  std::vector< MovingImageIndexType >   outputindexmovingvec( nrofpoints );
  std::vector< DeformationVectorType >  deformationvec( nrofpoints );

  /** Make a temporary image with the right region info,
   * which we can use to convert between points and indices.
   */
  FixedImageRegionType region;
  FixedImageOriginType origin = 
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputOrigin();
  FixedImageSpacingType spacing =
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputSpacing();
  region.SetIndex(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputStartIndex() );
  region.SetSize(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetSize() );

  typename FixedImageType::Pointer dummyImage = FixedImageType::New();
  dummyImage->SetRegions( region );
  dummyImage->SetOrigin( origin );
  dummyImage->SetSpacing( spacing );

  /** Temp vars */
  FixedImageContinuousIndexType fixedcindex;
  MovingImageContinuousIndexType movingcindex;

  /** Also output moving image indices if a moving image was supplied. */
  bool alsoMovingIndices = false;
  typename MovingImageType::Pointer movingImage = this->GetElastix()->GetMovingImage();
  if ( movingImage.IsNotNull() )
  {
    alsoMovingIndices = true;
  }

  /** Read the input points, as index or as point. */
  if ( !(ippReader->GetPointsAreIndices()) )
  {
    for ( unsigned int j = 0; j < nrofpoints; j++ )
    {
      /** Compute index of nearest voxel in fixed image. */
      InputPointType point;
      inputPointSet->GetPoint( j, &point );
      inputpointvec[ j ] = point;
      dummyImage->TransformPhysicalPointToContinuousIndex( 
        point, fixedcindex );
      for ( unsigned int i = 0; i < FixedImageDimension; i++ )
      {
        inputindexvec[ j ][ i ] = static_cast<FixedImageIndexValueType>(
          vnl_math_rnd( fixedcindex[ i ] ) );
      }
    } 
  } 
  else //so: inputasindex
  {
    for ( unsigned int j = 0; j < nrofpoints; j++ )
    {
      /** The read point from the inutPointSet is actually an index
       * Cast to the proper type.
       */
      InputPointType point;
      inputPointSet->GetPoint( j, &point );
      for ( unsigned int i = 0; i < FixedImageDimension; i++ )
      {
        inputindexvec[ j ][ i ] = static_cast<FixedImageIndexValueType>(
          vnl_math_rnd( point[ i ] ) );
      }
      /** Compute the input point in physical coordinates. */
      dummyImage->TransformIndexToPhysicalPoint(
        inputindexvec[ j ], inputpointvec[ j ] );
    }
  }

  /** Apply the transform. */
  elxout << "  The input points are transformed." << std::endl;
  for ( unsigned int j = 0; j < nrofpoints; j++ )
  {
    /** Call TransformPoint. */
    outputpointvec[ j ] = this->GetAsITKBaseType()->TransformPoint( inputpointvec[ j ] );

    /** Transform back to index in fixed image domain. */
    dummyImage->TransformPhysicalPointToContinuousIndex( 
      outputpointvec[ j ], fixedcindex );
    for ( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      outputindexfixedvec[ j ][ i ] = static_cast<FixedImageIndexValueType>(
        vnl_math_rnd( fixedcindex[ i ] ) );
    }

    if ( alsoMovingIndices )
    {
      /** Transform back to index in moving image domain. */
      movingImage->TransformPhysicalPointToContinuousIndex( 
        outputpointvec[ j ], movingcindex );
      for ( unsigned int i = 0; i < MovingImageDimension; i++ )
      {
        outputindexmovingvec[ j ][ i ] = static_cast<MovingImageIndexValueType>(
          vnl_math_rnd( movingcindex[ i ] ) );
      }
    }

    /** Compute displacement. */
    deformationvec[ j ].CastFrom( outputpointvec[ j ] - inputpointvec[ j ] );
  }

  /** Create filename and filestream. */
  std::string outputPointsFileName = this->m_Configuration->GetCommandLineArgument( "-out" );
  outputPointsFileName += "outputpoints.txt";
  std::ofstream outputPointsFile( outputPointsFileName.c_str() );
  outputPointsFile << std::showpoint << std::fixed;
  elxout << "  The transformed points are saved in: " <<  outputPointsFileName << std::endl;
  //\todo do not write opp to log file, but only to outputPointsFile.
  elxout.AddOutput( "opp", &outputPointsFile );

  /** Print the results. */
  for ( unsigned int j = 0; j < nrofpoints; j++ )
  {
    /** The input index. */
    elxout << "Point\t" << j << "\t; InputIndex = [ "; 
    for ( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      elxout << inputindexvec[ j ][ i ] << " ";
    }

    /** The input point. */
    elxout << "]\t; InputPoint = [ "; 
    for ( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      elxout << inputpointvec[ j ][ i ] << " ";
    }

    /** The output index in fixed image. */
    elxout << "]\t; OutputIndexFixed = [ "; 
    for ( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      elxout << outputindexfixedvec[ j ][ i ] << " ";
    }

    /** The output point. */
    elxout << "]\t; OutputPoint = [ "; 
    for ( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      elxout << outputpointvec[ j ][ i ] << " ";
    }

    /** The output point minus the input point. */
    elxout << "]\t; Deformation = [ "; 
    for ( unsigned int i = 0; i < MovingImageDimension; i++ )
    {
      elxout << deformationvec[ j ][ i ] << " ";
    }

    if ( alsoMovingIndices )
    {
      /** The output index in moving image. */
      elxout << "]\t; OutputIndexMoving = [ "; 
      for ( unsigned int i = 0; i < MovingImageDimension; i++ )
      {
        elxout << outputindexmovingvec[ j ][ i ] << " ";
      }
    }

    elxout << "]" << std::endl;
  } // end for nrofpoints 

  /** Stop writing to the output file. */
  elxout.RemoveOutput( "opp" );

} // end TransformPointsSomePoints()


/**
 * ************** TransformPointsAllPoints **********************
 *
 * This function transforms all indexes to a physical point.
 * The difference vector (= the deformation at that index) is
 * stored in an image of vectors (of floats).
 */

template <class TElastix>
void TransformBase<TElastix>
::TransformPointsAllPoints( void ) const
{
  /** Typedef's. */
  typedef typename FixedImageType::RegionType         FixedImageRegionType;
  typedef itk::Vector<
    float, FixedImageDimension >                      VectorPixelType;
  typedef itk::Image<
    VectorPixelType, FixedImageDimension >            DeformationFieldImageType;
  typedef typename DeformationFieldImageType::Pointer DeformationFieldImagePointer;
  typedef itk::TransformToDeformationFieldSource<
    DeformationFieldImageType, CoordRepType >         DeformationFieldGeneratorType;
  typedef itk::ImageFileWriter<
    DeformationFieldImageType >                       DeformationFieldWriterType;

  /** Create an setup deformation field generator. */
  typename DeformationFieldGeneratorType::Pointer defGenerator
    = DeformationFieldGeneratorType::New();
  defGenerator->SetOutputSize(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetSize() );
  defGenerator->SetOutputSpacing(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputSpacing() );
  defGenerator->SetOutputOrigin(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputOrigin() );
  defGenerator->SetOutputIndex(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputStartIndex() );
  defGenerator->SetOutputDirection(
    this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputDirection() );
  defGenerator->SetTransform( const_cast<const ITKBaseType *>( this->GetAsITKBaseType() ) );

  /** Track the progress of the generation of the deformation field. */
  typename ProgressCommandType::Pointer progressObserver = ProgressCommandType::New();
  progressObserver->ConnectObserver( defGenerator );
  progressObserver->SetStartString( "  Progress: " );
  progressObserver->SetEndString( "%" );

  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter( resultImageFormat, "ResultImageFormat", 0, true );
  std::ostringstream makeFileName( "" );
  makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
    << "deformationField." << resultImageFormat;

  /** Write outputImage to disk. */
  typename DeformationFieldWriterType::Pointer defWriter
    = DeformationFieldWriterType::New();
  defWriter->SetInput( defGenerator->GetOutput() );
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
 * ************** SetTransformParametersFileName ****************
 */

template <class TElastix>
void TransformBase<TElastix>
::SetTransformParametersFileName( const char * filename )
{
  /** Copied from itkSetStringMacro. */
  if ( filename && ( filename == this->m_TransformParametersFileName )  )
  {
    return;
  }
  if ( filename )
  {
    this->m_TransformParametersFileName = filename;
  }
  else
  {
    this->m_TransformParametersFileName = "";
  }
  ObjectType * thisAsObject = dynamic_cast<ObjectType *>(this);
  thisAsObject->Modified();

} // end SetTransformParametersFileName()


/**
 * ************** SetReadWriteTransformParameters ***************
 */

template <class TElastix>
void TransformBase<TElastix>
::SetReadWriteTransformParameters( const bool _arg )
{
  /** Copied from itkSetMacro. */
  if ( this->m_ReadWriteTransformParameters != _arg  )
  {
    this->m_ReadWriteTransformParameters = _arg;
    ObjectType * thisAsObject = dynamic_cast<ObjectType *>( this );
    thisAsObject->Modified();
  }

} // end SetReadWriteTransformParameters()


/**
 * ************** AutomaticScalesEstimation ***************
 */

template <class TElastix>
void TransformBase<TElastix>
::AutomaticScalesEstimation( ScalesType & scales ) const
{
  typedef itk::ImageGridSampler< FixedImageType >     ImageSamplerType;
  typedef typename ImageSamplerType::Pointer          ImageSamplerPointer;
  typedef typename 
    ImageSamplerType::ImageSampleContainerType        ImageSampleContainerType;
  typedef typename ImageSampleContainerType::Pointer  ImageSampleContainerPointer;
  typedef typename ITKBaseType::JacobianType          JacobianType;

  const ITKBaseType * const thisITK = this->GetAsITKBaseType();
  const unsigned int outdim = MovingImageDimension;
  const unsigned int N = thisITK->GetNumberOfParameters();
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
  if ( nrofsamples == 0 )
  {
    /** \todo: should we demand a minimum number (~100) of voxels? */
    itkExceptionMacro( << "No valid voxels found to estimate the scales." );
  }

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  /** initialize */
  scales.Fill( 0.0 );

  /** Read fixed coordinates and get Jacobian. */
  for ( iter = begin; iter != end; ++iter )
  {      
    const InputPointType & point = (*iter).Value().m_ImageCoordinates;
    const JacobianType & jacobian = thisITK->GetJacobian( point );   

    /** Square each element of the jacobian and add each row
     * to the newscales.
     */
    for( unsigned int d = 0; d < outdim; ++d )
    {
      ScalesType jacd(jacobian[d], N, false);
      scales += element_product( jacd, jacd );
    }
  }
  scales /= static_cast<double>( nrofsamples ); 

} // end AutomaticScalesEstimation()


} // end namespace elastix


#endif // end #ifndef __elxTransformBase_hxx

