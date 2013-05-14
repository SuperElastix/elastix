/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "elxElastixBase.h"
#include <sstream>
#include "itkMersenneTwisterRandomVariateGenerator.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

ElastixBase::ElastixBase()
{
  /** Initialize. */
  this->m_Configuration = 0;
  this->m_ComponentDatabase = 0;
  this->m_DBIndex = 0;

  /** The default output precision of elxout is set to 6. */
  this->m_DefaultOutputPrecision = 6;

  /** Create the component containers. */
  this->m_FixedImagePyramidContainer = ObjectContainerType::New();
  this->m_MovingImagePyramidContainer = ObjectContainerType::New();
  this->m_InterpolatorContainer = ObjectContainerType::New();
  this->m_ImageSamplerContainer = ObjectContainerType::New();
  this->m_MetricContainer = ObjectContainerType::New();
  this->m_OptimizerContainer = ObjectContainerType::New();
  this->m_RegistrationContainer = ObjectContainerType::New();
  this->m_ResamplerContainer = ObjectContainerType::New();
  this->m_ResampleInterpolatorContainer = ObjectContainerType::New();
  this->m_TransformContainer = ObjectContainerType::New();

  /** Create image and mask containers. */
  this->m_FixedImageContainer = DataObjectContainerType::New();
  this->m_MovingImageContainer = DataObjectContainerType::New();
  this->m_FixedImageFileNameContainer = FileNameContainerType::New();
  this->m_MovingImageFileNameContainer = FileNameContainerType::New();

  this->m_FixedMaskContainer = DataObjectContainerType::New();
  this->m_MovingMaskContainer = DataObjectContainerType::New();
  this->m_FixedMaskFileNameContainer = FileNameContainerType::New();
  this->m_MovingMaskFileNameContainer = FileNameContainerType::New();

  this->m_ResultImageContainer = DataObjectContainerType::New();

  /** Initialize initialTransform and final transform. */
  this->m_InitialTransform = 0;
  this->m_FinalTransform = 0;

  /** Ignore direction cosines by default, for backward compatability. */
  this->m_UseDirectionCosines = false;

} // end Constructor


/**
 * ********************* SetDBIndex ***********************
 */

void ElastixBase::SetDBIndex( DBIndexType _arg )
{
  /** If m_DBIndex is not set, set it. */
  if ( this->m_DBIndex != _arg )
  {
    this->m_DBIndex = _arg;

    itk::Object * thisasobject = dynamic_cast<itk::Object *>( this );
    if ( thisasobject )
    {
      thisasobject->Modified();
    }
  }

} // end SetDBIndex()


/**
 * ************************ BeforeAllBase ***************************
 */

int ElastixBase::BeforeAllBase( void )
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Set the default precision of floating values in the output. */
  this->m_Configuration->ReadParameter(
    this->m_DefaultOutputPrecision, "DefaultOutputPrecision", 0, false );
  elxout << std::setprecision( this->m_DefaultOutputPrecision );

  /** Print to log file. */
  elxout << std::fixed;
  elxout << std::showpoint;
  elxout << std::setprecision( 3 );
  elxout << "ELASTIX version: " << __ELASTIX_VERSION << std::endl;
  elxout << std::setprecision( this->GetDefaultOutputPrecision() );

  /** Check Command line options and print them to the logfile. */
  elxout << "Command line options from ElastixBase:" << std::endl;
  std::string check = "";

  /** Read the fixed and moving image filenames. These are obliged options,
   * so print an error if they are not present.
   * Print also some info (second boolean = true).
   */
#ifndef _ELASTIX_BUILD_LIBRARY
  this->m_FixedImageFileNameContainer = this->GenerateFileNameContainer(
    "-f", returndummy, true, true );
  this->m_MovingImageFileNameContainer = this->GenerateFileNameContainer(
    "-m", returndummy, true, true );
#endif
  /** Read the fixed and moving mask filenames. These are not obliged options,
   * so do not print any errors if they are not present.
   * Do print some info (second boolean = true).
   */
  int maskreturndummy = 0;
  this->m_FixedMaskFileNameContainer = this->GenerateFileNameContainer(
    "-fMask", maskreturndummy, false, true );
  if ( maskreturndummy != 0 )
  {
    elxout << "-fMask    unspecified, so no fixed mask used" << std::endl;
  }
  maskreturndummy = 0;
  this->m_MovingMaskFileNameContainer = this->GenerateFileNameContainer(
    "-mMask", maskreturndummy, false, true );
  if ( maskreturndummy != 0 )
  {
    elxout << "-mMask    unspecified, so no moving mask used" << std::endl;
  }

  /** Check for appearance of "-out".
   * This check has already been performed in elastix.cxx,
   * Here we do it again. MS: WHY?
   */
  check = "";
  check = this->GetConfiguration()->GetCommandLineArgument( "-out" );
  if ( check == "" )
  {
    xl::xout["error"] << "ERROR: No CommandLine option \"-out\" given!" << std::endl;
    returndummy |= 1;
  }
  else
  {
    /** Make sure that last character of the output folder equals a '/' or '\'. */
    std::string folder( check );
    const char last = folder[ folder.size() - 1 ];
    if( last != '/' && last != '\\' )
    {
      folder.append( "/" );
      folder = itksys::SystemTools::ConvertToOutputPath( folder.c_str() );

      /** Note that on Windows, in case the output folder contains a space,
       * the path name is double quoted by ConvertToOutputPath, which is undesirable.
       * So, we remove these quotes again.
       */
      if(  itksys::SystemTools::StringStartsWith( folder.c_str(), "\"" )
        && itksys::SystemTools::StringEndsWith(   folder.c_str(), "\"" ) )
      {
        folder = folder.substr( 1, folder.length() - 2 );
      }

      this->GetConfiguration()->SetCommandLineArgument( "-out", folder.c_str() );
    }
    elxout << "-out      " << check << std::endl;
  }

  /** Print all "-p". */
  unsigned int i = 1;
  bool loop = true;
  while ( loop )
  {
    check = "";
    std::ostringstream tempPname("");
    tempPname << "-p(" << i << ")";
    check = this->GetConfiguration()->GetCommandLineArgument( tempPname.str().c_str() );
    if ( check == "" ) loop = false;
    else elxout << "-p        " << check << std::endl;
    ++i;
  }

  /** Check for appearance of "-priority", if this is a Windows station. */
#ifdef _WIN32
  check = "";
  check = this->GetConfiguration()->GetCommandLineArgument( "-priority" );
  if ( check == "" )
  {
    elxout << "-priority unspecified, so NORMAL process priority" << std::endl;
  }
  else
  {
    elxout << "-priority " << check << std::endl;
  }
#endif

  /** Check for appearance of -threads, which specifies the maximum number of threads. */
  check = "";
  check = this->GetConfiguration()->GetCommandLineArgument( "-threads" );
  if ( check == "" )
  {
    elxout << "-threads  unspecified, so all available threads are used" << std::endl;
  }
  else
  {
    elxout << "-threads  " << check << std::endl;
  }

  /** Check the very important UseDirectionCosines parameter. */
  this->m_UseDirectionCosines = false;
  bool retudc = this->GetConfiguration()->ReadParameter( this->m_UseDirectionCosines,
    "UseDirectionCosines", 0 );
  if ( !retudc )
  {
    xl::xout["warning"]
      << "\nWARNING: From elastix 4.3 it is highly recommended to add\n"
      << "the UseDirectionCosines option to your parameter file! See\n"
      << "http://elastix.isi.uu.nl/whatsnew_04_3.php for more information.\n"
      << std::endl;
  }

  /** Set the random seed. Use 121212 as a default, which is the same as
   * the default in the MersenneTwister code.
   * Use silent parameter file readout, to avoid annoying warning when
   * starting elastix */
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;
  typedef RandomGeneratorType::IntegerType SeedType;
  unsigned int randomSeed = 121212;
  this->GetConfiguration()->ReadParameter( randomSeed, "RandomSeed", 0, false );
  RandomGeneratorType::Pointer randomGenerator = RandomGeneratorType::GetInstance();
  randomGenerator->SetSeed( static_cast<SeedType>( randomSeed ) );

  /** Return a value. */
  return returndummy;

} // end BeforeAllBase()


/**
 * ************************ BeforeAllTransformixBase ***************************
 */

int ElastixBase::BeforeAllTransformixBase( void )
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Print to log file. */
  elxout << std::fixed;
  elxout << std::showpoint;
  elxout << std::setprecision( 3 );
  elxout << "ELASTIX version: " << __ELASTIX_VERSION << std::endl;
  elxout << std::setprecision( this->GetDefaultOutputPrecision() );

  /** Check Command line options and print them to the logfile. */
  elxout << "Command line options from ElastixBase:" << std::endl;
  std::string check = "";
#ifndef _ELASTIX_BUILD_LIBRARY
  /** Read the input image filenames. These are not obliged options,
   * so do not print an error if they are not present.
   * Print also some info (second boolean = true)
   * Save the result in the moving image file name container.
   */
  int inreturndummy = 0;
  this->m_MovingImageFileNameContainer = this->GenerateFileNameContainer(
    "-in", inreturndummy, false, true );
  if ( inreturndummy != 0 )
  {
    elxout << "-in       unspecified, so no input image specified" << std::endl;
  }
#endif
  /** Check for appearance of "-out". */
  check = this->GetConfiguration()->GetCommandLineArgument( "-out" );
  if ( check == "" )
  {
    xl::xout["error"] << "ERROR: No CommandLine option \"-out\" given!" << std::endl;
    returndummy |= 1;
  }
  else
  {
    /** Make sure that last character of -out equals a '/'. */
    std::string folder( check );
    if ( folder.find_last_of( "/" ) != folder.size() - 1 )
    {
      folder.append( "/" );
      this->GetConfiguration()->SetCommandLineArgument( "-out", folder.c_str() );
    }
    elxout << "-out      " << check << std::endl;
  }

  /** Check for appearance of -threads, which specifies the maximum number of threads. */
  check = "";
  check = this->GetConfiguration()->GetCommandLineArgument( "-threads" );
  if ( check == "" )
  {
    elxout << "-threads  unspecified, so all available threads are used" << std::endl;
  }
  else
  {
    elxout << "-threads  " << check << std::endl;
  }
#ifndef _ELASTIX_BUILD_LIBRARY
  /** Print "-tp". */
  check = this->GetConfiguration()->GetCommandLineArgument( "-tp" );
  elxout << "-tp       " << check << std::endl;
#endif
  /** Check the very important UseDirectionCosines parameter. */
  this->m_UseDirectionCosines = false;
  bool retudc = this->GetConfiguration()->ReadParameter( this->m_UseDirectionCosines,
    "UseDirectionCosines", 0 );
  if ( !retudc )
  {
    xl::xout["warning"]
      << "\nWARNING: From elastix 4.3 it is highly recommended to add\n"
      << "the UseDirectionCosines option to your parameter file! See\n"
      << "http://elastix.isi.uu.nl/whatsnew_04_3.php for more information.\n"
      << std::endl;
  }

  return returndummy;

} // end BeforeAllTransformixBase()


/**
 * ************************ BeforeRegistrationBase ******************
 */

void ElastixBase::BeforeRegistrationBase( void )
{
  using namespace xl;

  /** Set up the "iteration" writing field. */
  this->m_IterationInfo.SetOutputs( xout.GetCOutputs() );
  this->m_IterationInfo.SetOutputs( xout.GetXOutputs() );

  xout.AddTargetCell( "iteration", &this->m_IterationInfo );

} // end BeforeRegistrationBase()


/**
 * **************** AfterRegistrationBase ***********************
 */

void ElastixBase::AfterRegistrationBase( void )
{
  /** Remove the "iteration" writing field. */
  xl::xout.RemoveTargetCell( "iteration" );

} // end AfterRegistrationBase()


/**
 * ********************* GenerateFileNameContainer ******************
 */

ElastixBase::FileNameContainerPointer
ElastixBase::GenerateFileNameContainer(
  const std::string & optionkey, int & errorcode,
  bool printerrors, bool printinfo ) const
{
  FileNameContainerPointer fileNameContainer = FileNameContainerType::New();
  std::string check = "";
  std::string argused( "" );

  /** Try optionkey0. */
  std::ostringstream argusedss( "" );
  argusedss << optionkey << 0;
  argused = argusedss.str();
  check = this->GetConfiguration()->GetCommandLineArgument( argused.c_str() );
  if ( check == "" )
  {
    /** Try optionkey. */
    std::ostringstream argusedss2( "" );
    argusedss2 << optionkey;
    argused = argusedss2.str();
    check = this->GetConfiguration()->GetCommandLineArgument( argused.c_str() );
    if ( check == "" )
    {
      /** Both failed; return an error message, if desired. */
      if ( printerrors )
      {
        xl::xout["error"]
        << "ERROR: No CommandLine option \""
          << optionkey << "\" or \""
          << optionkey << 0 << "\" given!" << std::endl;
      }
      errorcode |= 1;

      return fileNameContainer;
    }
  }

  /** Optionkey or optionkey0 is found. */
  if ( check != "" )
  {
    /** Print info, if desired. */
    if ( printinfo )
    {
      /** Print the option, with some spaces, followed by the value. */
      int nrSpaces0 = 10 - argused.length();
      unsigned int nrSpaces = nrSpaces0 > 1 ? nrSpaces0 : 1;
      std::string spaces = "";
      spaces.resize( nrSpaces, ' ' );
      elxout << argused << spaces << check << std::endl;
    }
    fileNameContainer->CreateElementAt( 0 ) = check;

    /** Loop over all optionkey<i> options given with i > 0. */
    unsigned int i = 1;
    bool readsuccess = true;
    while ( readsuccess )
    {
      std::ostringstream argusedss2( "" );
      argusedss2 << optionkey << i;
      argused = argusedss2.str();
      check = this->GetConfiguration()->GetCommandLineArgument( argused.c_str() );
      if ( check == "" )
      {
        readsuccess = false;
      }
      else
      {
        if ( printinfo )
        {
          /** Print the option, with some spaces, followed by the value. */
          int nrSpaces0 = 10 - argused.length();
          unsigned int nrSpaces = nrSpaces0 > 1 ? nrSpaces0 : 1;
          std::string spaces = "";
          spaces.resize( nrSpaces, ' ' );
          elxout << argused << spaces << check << std::endl;
        }
        fileNameContainer->CreateElementAt(i) = check;
        ++i;
      }
    } // end while
  } // end if

  return fileNameContainer;

} // end GenerateFileNameContainer()


/**
 * ******************** GetUseDirectionCosines ********************
 */

bool ElastixBase::GetUseDirectionCosines( void ) const
{
  return this->m_UseDirectionCosines;
}


/**
 * ******************** SetOriginalFixedImageDirectionFlat ********************
 */

void ElastixBase::SetOriginalFixedImageDirectionFlat(
  const FlatDirectionCosinesType & arg )
{
  this->m_OriginalFixedImageDirection = arg;
}


/**
 * ******************** GetOriginalFixedImageDirectionFlat ********************
 */

const ElastixBase::FlatDirectionCosinesType &
ElastixBase::GetOriginalFixedImageDirectionFlat( void ) const
{
  return this->m_OriginalFixedImageDirection;
}


} // end namespace elastix

