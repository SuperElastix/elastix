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

#ifndef __elxMovingImagePyramidBase_hxx
#define __elxMovingImagePyramidBase_hxx

#include "elxMovingImagePyramidBase.h"
#include "itkImageFileCastWriter.h"

namespace elastix
{

/**
 * ******************* BeforeRegistrationBase *******************
 */

template< class TElastix >
void
MovingImagePyramidBase< TElastix >
::BeforeRegistrationBase( void )
{
  /** Call SetMovingSchedule.*/
  this->SetMovingSchedule();

} // end BeforeRegistrationBase()


/**
 * ******************* BeforeEachResolutionBase *******************
 */

template< class TElastix >
void
MovingImagePyramidBase< TElastix >
::BeforeEachResolutionBase( void )
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Decide whether or not to write the pyramid images this resolution. */
  bool writePyramidImage = false;
  this->m_Configuration->ReadParameter( writePyramidImage,
    "WritePyramidImagesAfterEachResolution", "", level, 0, false );

  /** Get the desired extension / file format. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter( resultImageFormat,
    "ResultImageFormat", 0, false );

  /** Writing result image. */
  if( writePyramidImage )
  {
    /** Create a name for the final result. */
    std::ostringstream makeFileName( "" );
    makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" );
    makeFileName
      << this->GetComponentLabel() << "."
      << this->m_Configuration->GetElastixLevel()
      << ".R" << level
      << "." << resultImageFormat;

    /** Save the fixed pyramid image. */
    elxout << "Writing moving pyramid image "
           << this->GetComponentLabel()
           << " from resolution " << level << "..." << std::endl;
    try
    {
      this->WritePyramidImage( makeFileName.str(), level );
    }
    catch( itk::ExceptionObject & excp )
    {
      xl::xout[ "error" ] << "Exception caught: " << std::endl;
      xl::xout[ "error" ] << excp << "Resuming elastix." << std::endl;
    }
  } // end if

} // end BeforeEachResolutionBase()


/**
 * ********************** SetMovingSchedule **********************
 */

template< class TElastix >
void
MovingImagePyramidBase< TElastix >
::SetMovingSchedule( void )
{
  /** Get the ImageDimension. */
  const unsigned int MovingImageDimension = InputImageType::ImageDimension;

  /** Read numberOfResolutions. */
  unsigned int numberOfResolutions = 0;
  this->m_Configuration->ReadParameter( numberOfResolutions,
    "NumberOfResolutions", 0, true );
  if( numberOfResolutions == 0 )
  {
    xl::xout[ "error" ] << "ERROR: NumberOfResolutions not specified!" << std::endl;
  }
  /** \todo quit program? Actually this check should be in the ::BeforeAll() method. */

  /** Create a default movingSchedule. Set the numberOfLevels first. */
  this->GetAsITKBaseType()->SetNumberOfLevels( numberOfResolutions );
  ScheduleType movingSchedule = this->GetAsITKBaseType()->GetSchedule();

  /** Set the movingPyramidSchedule to the MovingImagePyramidSchedule given
   * in the parameter-file. The following parameter file fields can be used:
   * ImagePyramidSchedule
   * MovingImagePyramidSchedule
   * MovingImagePyramid<i>Schedule, for the i-th moving image pyramid used.
   */
  bool found = true;
  for( unsigned int i = 0; i < numberOfResolutions; i++ )
  {
    for( unsigned int j = 0; j < MovingImageDimension; j++ )
    {
      bool               ijfound = false;
      const unsigned int entrynr = i * MovingImageDimension + j;
      ijfound |= this->m_Configuration->ReadParameter( movingSchedule[ i ][ j ],
        "ImagePyramidSchedule", entrynr, false );
      ijfound |= this->m_Configuration->ReadParameter( movingSchedule[ i ][ j ],
        "MovingImagePyramidSchedule", entrynr, false );
      ijfound |= this->m_Configuration->ReadParameter( movingSchedule[ i ][ j ],
        "Schedule", this->GetComponentLabel(), entrynr, -1, false );

      /** Remember if for at least one schedule element no value could be found. */
      found &= ijfound;

    } // end for MovingImageDimension
  }   // end for numberOfResolutions

  if( !found && this->GetConfiguration()->GetPrintErrorMessages() )
  {
    xl::xout[ "warning" ] << "WARNING: the moving pyramid schedule is not fully specified!\n";
    xl::xout[ "warning" ] << "  A default pyramid schedule is used." << std::endl;
  }
  else
  {
    /** Set the schedule into this class. */
    this->GetAsITKBaseType()->SetSchedule( movingSchedule );
  }

} // end SetMovingSchedule()


/*
 * ******************* WritePyramidImage ********************
 */

template< class TElastix >
void
MovingImagePyramidBase< TElastix >
::WritePyramidImage( const std::string & filename,
  const unsigned int & level ) //const
{
  /** Read output pixeltype from parameter the file. Replace possible " " with "_". */
  std::string resultImagePixelType = "short";
  this->m_Configuration->ReadParameter( resultImagePixelType,
    "ResultImagePixelType", 0, false );
  std::basic_string< char >::size_type       pos  = resultImagePixelType.find( " " );
  const std::basic_string< char >::size_type npos = std::basic_string< char >::npos;
  if( pos != npos ) { resultImagePixelType.replace( pos, 1, "_" ); }

  /** Read from the parameter file if compression is desired. */
  bool doCompression = false;
  this->m_Configuration->ReadParameter(
    doCompression, "CompressResultImage", 0, false );

  /** Create writer. */
  typedef itk::ImageFileCastWriter< OutputImageType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();

  /** Setup the pipeline. */
  writer->SetInput( this->GetAsITKBaseType()->GetOutput( level ) );
  writer->SetFileName( filename.c_str() );
  writer->SetOutputComponentType( resultImagePixelType.c_str() );
  writer->SetUseCompression( doCompression );

  /** Do the writing. */
  xl::xout[ "coutonly" ] << std::flush;
  xl::xout[ "coutonly" ] << "  Writing image ..." << std::endl;
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "MovingImagePyramidBase - BeforeEachResolutionBase()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing pyramid image.\n";
    excp.SetDescription( err_str );

    /** Pass the exception to an higher level. */
    throw excp;
  }

} // end WritePyramidImage()


} // end namespace elastix

#endif // end #ifndef __elxMovingImagePyramidBase_hxx
