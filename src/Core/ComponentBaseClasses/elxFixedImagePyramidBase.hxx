/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxFixedImagePyramidBase_hxx
#define __elxFixedImagePyramidBase_hxx

#include "elxFixedImagePyramidBase.h"

namespace elastix
{
using namespace itk;

/**
 * ******************* BeforeRegistrationBase *******************
 */

template <class TElastix>
void
FixedImagePyramidBase<TElastix>
::BeforeRegistrationBase( void )
{
  /** Call SetFixedSchedule.*/
  this->SetFixedSchedule();

} // end BeforeRegistrationBase()


/**
 * ******************* AfterEachResolutionBase *******************
 */

template <class TElastix>
void
FixedImagePyramidBase<TElastix>
::AfterEachResolutionBase( void )
{
  /** What is the current resolution level? */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Decide whether or not to write the pyramid image this resolution. */
  bool writePyramidImage = false;
  this->m_Configuration->ReadParameter( writePyramidImage,
    "WritePyramidImagesAfterEachResolution", "", level, 0, false );

  /** Writing result image. */
  if ( writePyramidImage )
  {
    /** Create a name for the final result. */
    std::string resultImageFormat = "mhd";
    this->m_Configuration->ReadParameter( resultImageFormat,
      "ResultImageFormat", 0, false );
    std::ostringstream makeFileName( "" );
    makeFileName
      << this->m_Configuration->GetCommandLineArgument( "-out" )
      << "pyramid_fixed." << this->m_Configuration->GetElastixLevel()
      << ".R" << level
      << "." << resultImageFormat;

//     /** Time the writing. */
//     typedef tmr::Timer TimerType;
//     TimerType::Pointer timer = TimerType::New();
//     timer->StartTimer();

    /** Save the fixed pyramid image. */
    elxout << "Writing fixed pyramid image this resolution ..." << std::endl;
    try
    {
      this->WritePyramidImage( makeFileName.str(), level );
    }
    catch( itk::ExceptionObject & excp )
    {
      xl::xout["error"] << "Exception caught: " << std::endl;
      xl::xout["error"] << excp
        << "Resuming elastix." << std::endl;
    }

//     /** Print the elapsed time for the writing. */
//     timer->StopTimer();
//     elxout << "  Applying transform took "
//       << static_cast<long>( timer->GetElapsedClockSec() )
//       << " s." << std::endl;

  } // end if

} // end AfterEachResolutionBase()


/**
 * ********************** SetFixedSchedule **********************
 */

template <class TElastix>
void
FixedImagePyramidBase<TElastix>
::SetFixedSchedule( void )
{
  /** Get the ImageDimension. */
  const unsigned int FixedImageDimension = InputImageType::ImageDimension;

  /** Read numberOfResolutions. */
  unsigned int numberOfResolutions = 0;
  this->m_Configuration->ReadParameter( numberOfResolutions,
    "NumberOfResolutions", 0, true );
  if ( numberOfResolutions == 0 )
  {
    xl::xout["error"] << "ERROR: NumberOfResolutions not specified!" << std::endl;
  }
  /** \todo quit program? Actually this check should be in the ::BeforeAll() method. */

  /** Create a default fixedSchedule. Set the numberOfLevels first. */
  this->GetAsITKBaseType()->SetNumberOfLevels( numberOfResolutions );
  ScheduleType fixedSchedule = this->GetAsITKBaseType()->GetSchedule();

  /** Set the fixedPyramidSchedule to the FixedImagePyramidSchedule given 
   * in the parameter-file. The following parameter file fields can be used:
   * ImagePyramidSchedule
   * FixedImagePyramidSchedule
   * FixedImagePyramid<i>Schedule, for the i-th fixed image pyramid used. 
   */
  bool found = true;
  for ( unsigned int i = 0; i < numberOfResolutions; i++ )
  {
    for ( unsigned int j = 0; j < FixedImageDimension; j++ )
    {
      bool ijfound = false;
      const unsigned int entrynr = i * FixedImageDimension + j;
      ijfound |= this->m_Configuration->ReadParameter( fixedSchedule[ i ][ j ],
        "ImagePyramidSchedule", entrynr, false );
      ijfound |= this->m_Configuration->ReadParameter( fixedSchedule[ i ][ j ],
        "FixedImagePyramidSchedule", entrynr, false );
      ijfound |= this->m_Configuration->ReadParameter( fixedSchedule[ i ][ j ],
        "Schedule", this->GetComponentLabel(), entrynr, -1, false );

      /** Remember if for at least one schedule element no value could be found. */
      found &= ijfound;

    } // end for FixedImageDimension
  } // end for numberOfResolutions

  if ( !found && this->GetConfiguration()->GetPrintErrorMessages() )
  {
    xl::xout["warning"] << "WARNING: the fixed pyramid schedule is not fully specified!\n";
    xl::xout["warning"] << "  A default pyramid schedule is used." << std::endl;
  }
  else
  {
    /** Set the schedule into this class. */
    this->GetAsITKBaseType()->SetSchedule( fixedSchedule );
  }

} // end SetFixedSchedule()


/*
 * ******************* WritePyramidImage ********************
 */

template<class TElastix>
void
FixedImagePyramidBase<TElastix>
::WritePyramidImage( const std::string & filename, const unsigned int & level )
{
//   /** Make sure the resampler is updated. */
//   this->GetAsITKBaseType()->Modified();
// 
//   /** Add a progress observer to the resampler. */
//   typename ProgressCommandType::Pointer progressObserver = ProgressCommandType::New();
//   progressObserver->ConnectObserver( this->GetAsITKBaseType() );
//   progressObserver->SetStartString( "  Progress: " );
//   progressObserver->SetEndString( "%" );
//
//   /** Do the resampling. */
//   try
//   {
//     this->GetAsITKBaseType()->Update();
//   }
//   catch( itk::ExceptionObject & excp )
//   {
//     /** Add information to the exception. */
//     excp.SetLocation( "ResamplerBase - WriteResultImage()" );
//     std::string err_str = excp.GetDescription();
//     err_str += "\nError occurred while resampling the image.\n";
//     excp.SetDescription( err_str );
// 
//     /** Pass the exception to an higher level. */
//     throw excp;
//   }

  /** Read output pixeltype from parameter the file. Replace possible " " with "_". */
  std::string resultImagePixelType = "short";
  this->m_Configuration->ReadParameter( resultImagePixelType,
    "ResultImagePixelType", 0, false );
  std::basic_string<char>::size_type pos = resultImagePixelType.find( " " );
  const std::basic_string<char>::size_type npos = std::basic_string<char>::npos;
  if ( pos != npos ) resultImagePixelType.replace( pos, 1, "_" );

  /** Read from the parameter file if compression is desired. */
  bool doCompression = false;
  this->m_Configuration->ReadParameter(
    doCompression, "CompressResultImage", 0, false );

  /** Typedef's for writing the output image. */
  typedef ImageFileCastWriter< OutputImageType >  WriterType;

  /** Create writer. */
  typename WriterType::Pointer writer = WriterType::New();

  /** Setup the pipeline. */
  writer->SetInput( this->GetAsITKBaseType()->GetOutput( level ) );
  writer->SetFileName( filename.c_str() );
  writer->SetOutputComponentType( resultImagePixelType.c_str() );
  writer->SetUseCompression( doCompression );

  /** Do the writing. */
  xl::xout["coutonly"] << std::flush;
  xl::xout["coutonly"] << "  Writing image ..." << std::endl;
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "FixedImagePyramidBase - AfterEachResolutionBase()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing pyramid image.\n";
    excp.SetDescription( err_str );

    /** Pass the exception to an higher level. */
    throw excp;
  }

  /** Disconnect from the resampler. */
  //progressObserver->DisconnectObserver( this->GetAsITKBaseType() );

} // end WritePyramidImage()


} // end namespace elastix

#endif // end #ifndef __elxFixedImagePyramidBase_hxx

