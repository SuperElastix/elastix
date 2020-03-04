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
#ifndef __elxResamplerBase_hxx
#define __elxResamplerBase_hxx

#include "elxResamplerBase.h"

#include "itkImageFileCastWriter.h"
#include "itkChangeInformationImageFilter.h"
#include "itkAdvancedRayCastInterpolateImageFunction.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* Constructor *******************
 */

template< class TElastix >
ResamplerBase< TElastix >
::ResamplerBase()
{
  this->m_ShowProgress = true;
} // end Constructor


/**
 * ******************* BeforeRegistrationBase *******************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::BeforeRegistrationBase( void )
{
  /** Connect the components. */
  this->SetComponents();

  /** Set the size of the image to be produced by the resampler. */

  /** Get a pointer to the fixedImage.
   * \todo make it a cast to the fixed image type
   */
  typedef typename ElastixType::FixedImageType FixedImageType;
  FixedImageType * fixedImage = this->m_Elastix->GetFixedImage();

  /** Set the region info to the same values as in the fixedImage. */
  this->GetAsITKBaseType()->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  this->GetAsITKBaseType()->SetOutputStartIndex( fixedImage->GetLargestPossibleRegion().GetIndex() );
  this->GetAsITKBaseType()->SetOutputOrigin( fixedImage->GetOrigin() );
  this->GetAsITKBaseType()->SetOutputSpacing( fixedImage->GetSpacing() );
  this->GetAsITKBaseType()->SetOutputDirection( fixedImage->GetDirection() );

  /** Set the DefaultPixelValue (for pixels in the resampled image
   * that come from outside the original (moving) image.
   */
  double defaultPixelValue = itk::NumericTraits< double >::Zero;
  this->m_Configuration->ReadParameter(
    defaultPixelValue, "DefaultPixelValue", 0, false );

  /** Set the defaultPixelValue. */
  this->GetAsITKBaseType()->SetDefaultPixelValue(
    static_cast< OutputPixelType >( defaultPixelValue ) );

} // end BeforeRegistrationBase()


/**
 * ******************* AfterEachResolutionBase ********************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::AfterEachResolutionBase( void )
{
  /** Set the final transform parameters. */
  this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Decide whether or not to write the result image this resolution. */
  bool writeResultImageThisResolution = false;
  this->m_Configuration->ReadParameter( writeResultImageThisResolution,
    "WriteResultImageAfterEachResolution", "", level, 0, false );

  /** Writing result image. */
  if( writeResultImageThisResolution )
  {
    /** Create a name for the final result. */
    std::string resultImageFormat = "mhd";
    this->m_Configuration->ReadParameter( resultImageFormat,
      "ResultImageFormat", 0, false );
    std::ostringstream makeFileName( "" );
    makeFileName
      << this->m_Configuration->GetCommandLineArgument( "-out" )
      << "result." << this->m_Configuration->GetElastixLevel()
      << ".R" << level
      << "." << resultImageFormat;

    /** Time the resampling. */
    itk::TimeProbe timer;
    timer.Start();

    /** Apply the final transform, and save the result. */
    elxout << "Applying transform this resolution ..." << std::endl;
    try
    {
      this->ResampleAndWriteResultImage( makeFileName.str().c_str() );
    }
    catch( itk::ExceptionObject & excp )
    {
      xl::xout[ "error" ] << "Exception caught: " << std::endl;
      xl::xout[ "error" ] << excp
                          << "Resuming elastix." << std::endl;
    }

    /** Print the elapsed time for the resampling. */
    timer.Stop();
    elxout << "  Applying transform took "
           << this->ConvertSecondsToDHMS( timer.GetMean(), 2 ) << std::endl;

  } // end if

} // end AfterEachResolutionBase()


/**
 * ******************* AfterEachIterationBase ********************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::AfterEachIterationBase( void )
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** What is the current iteration number? */
  const unsigned int iter = this->m_Elastix->GetIterationCounter();

  /** Decide whether or not to write the result image this iteration. */
  bool writeResultImageThisIteration = false;
  this->m_Configuration->ReadParameter( writeResultImageThisIteration,
    "WriteResultImageAfterEachIteration", "", level, 0, false );

  /** Writing result image. */
  if( writeResultImageThisIteration )
  {
    /** Set the final transform parameters. */
    this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

    /** Create a name for the final result. */
    std::string resultImageFormat = "mhd";
    this->m_Configuration->ReadParameter( resultImageFormat,
      "ResultImageFormat", 0, false );
    std::ostringstream makeFileName( "" );
    makeFileName
      << this->m_Configuration->GetCommandLineArgument( "-out" )
      << "result." << this->m_Configuration->GetElastixLevel()
      << ".R" << level
      << ".It" << std::setfill( '0' ) << std::setw( 7 ) << iter
      << "." << resultImageFormat;

    /** Apply the final transform, and save the result. */
    try
    {
      this->ResampleAndWriteResultImage( makeFileName.str().c_str(), false );
    }
    catch( itk::ExceptionObject & excp )
    {
      xl::xout[ "error" ] << "Exception caught: " << std::endl;
      xl::xout[ "error" ] << excp
                          << "Resuming elastix." << std::endl;
    }

  } // end if

} // end AfterEachIterationBase()


/**
 * ******************* AfterRegistrationBase ********************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::AfterRegistrationBase( void )
{
  /** Set the final transform parameters. */
  this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

  /** Decide whether or not to write the result image. */
  std::string writeResultImage = "true";
  this->m_Configuration->ReadParameter(
    writeResultImage, "WriteResultImage", 0 );

  const auto isElastixLibrary = BaseComponent::IsElastixLibrary();

  /** The library interface may executed multiple times in
   * a session in which case the images should not be released
   * However, if this is not the library interface:
   * Release memory to be able to resample in case a limited
   * amount of memory is available.
   */
  bool releaseMemoryBeforeResampling{ !isElastixLibrary };

  this->m_Configuration->ReadParameter( releaseMemoryBeforeResampling, "ReleaseMemoryBeforeResampling", 0, false );
  if( releaseMemoryBeforeResampling )
  {
    this->ReleaseMemory();
  }

  /**
   * Create the result image and put it in ResultImageContainer
   * Only necessary when compiling elastix as a library!
   */
  if (isElastixLibrary)
  {
    if( writeResultImage == "true" )
    {
      this->CreateItkResultImage();
    }
  }
  else
  {
    /** Writing result image. */
    if( writeResultImage == "true" )
    {
      /** Create a name for the final result. */
      std::string resultImageFormat = "mhd";
      this->m_Configuration->ReadParameter(
        resultImageFormat, "ResultImageFormat", 0 );
      std::ostringstream makeFileName( "" );
      makeFileName
        << this->m_Configuration->GetCommandLineArgument( "-out" )
        << "result." << this->m_Configuration->GetElastixLevel()
        << "." << resultImageFormat;

      /** Time the resampling. */
      itk::TimeProbe timer;
      timer.Start();

      /** Apply the final transform, and save the result,
       * by calling ResampleAndWriteResultImage.
       */
      elxout << "\nApplying final transform ..." << std::endl;
      try
      {
        this->ResampleAndWriteResultImage( makeFileName.str().c_str(), this->m_ShowProgress );
      }
      catch( itk::ExceptionObject & excp )
      {
        xl::xout[ "error" ] << "Exception caught: " << std::endl;
        xl::xout[ "error" ] << excp << "Resuming elastix." << std::endl;
      }

      /** Print the elapsed time for the resampling. */
      timer.Stop();
      elxout << "  Applying final transform took "
             << this->ConvertSecondsToDHMS( timer.GetMean(), 2 ) << std::endl;
    }
    else
    {
      /** Do not apply the final transform. */
      elxout << std::endl
             << "Skipping applying final transform, no resulting output image generated."
             << std::endl;
    } // end if
  }

} // end AfterRegistrationBase()


/**
 * *********************** SetComponents ************************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::SetComponents( void )
{
  /** Set the transform, the interpolator and the inputImage
   * (which is the moving image).
   */
  this->GetAsITKBaseType()->SetTransform( dynamic_cast< TransformType * >(
      this->m_Elastix->GetElxTransformBase() ) );

  this->GetAsITKBaseType()->SetInterpolator( dynamic_cast< InterpolatorType * >(
      this->m_Elastix->GetElxResampleInterpolatorBase() ) );

  this->GetAsITKBaseType()->SetInput( dynamic_cast< InputImageType * >(
      this->m_Elastix->GetMovingImage() ) );

} // end SetComponents()


/**
 * ******************* ResampleAndWriteResultImage ********************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::ResampleAndWriteResultImage( const char * filename, const bool & showProgress )
{
  /** Make sure the resampler is updated. */
  this->GetAsITKBaseType()->Modified();

  /** Add a progress observer to the resampler. */
  const auto progressObserver = BaseComponent::IsElastixLibrary() ?
    nullptr : ProgressCommandType::New();
  if( showProgress && (progressObserver != nullptr) )
  {
    progressObserver->ConnectObserver( this->GetAsITKBaseType() );
    progressObserver->SetStartString( "  Progress: " );
    progressObserver->SetEndString( "%" );
  }

  /** Do the resampling. */
  try
  {
    this->GetAsITKBaseType()->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "ResamplerBase - WriteResultImage()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while resampling the image.\n";
    excp.SetDescription( err_str );

    /** Pass the exception to an higher level. */
    throw excp;
  }

  /** Perform the writing. */
  this->WriteResultImage( this->GetAsITKBaseType()->GetOutput(), filename, showProgress );

  /** Disconnect from the resampler. */
  if( showProgress && (progressObserver != nullptr) )
  {
    progressObserver->DisconnectObserver( this->GetAsITKBaseType() );
  }

} // end ResampleAndWriteResultImage()


/**
 * ******************* WriteResultImage ********************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::WriteResultImage( OutputImageType * image,
  const char * filename, const bool & showProgress )
{
  /** Check if ResampleInterpolator is the RayCastResampleInterpolator  */
  typedef itk::AdvancedRayCastInterpolateImageFunction<  InputImageType,
    CoordRepType > RayCastInterpolatorType;
  const RayCastInterpolatorType * testptr = dynamic_cast< const
    RayCastInterpolatorType * >( this->GetAsITKBaseType()->GetInterpolator() );

  /** If RayCastResampleInterpolator is used reset the Transform to
   * overrule default Resampler settings.
   */

  if( testptr )
  {
    this->GetAsITKBaseType()->SetTransform(
      ( const_cast< RayCastInterpolatorType * >( testptr ) )->GetTransform() );
  }

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

  /** Typedef's for writing the output image. */
  typedef itk::ImageFileCastWriter< OutputImageType > WriterType;
  typedef typename WriterType::Pointer                WriterPointer;
  typedef itk::ChangeInformationImageFilter<
    OutputImageType >                                 ChangeInfoFilterType;

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false.
   */
  typename ChangeInfoFilterType::Pointer infoChanger = ChangeInfoFilterType::New();
  DirectionType originalDirection;
  bool          retdc = this->GetElastix()->GetOriginalFixedImageDirection( originalDirection );
  infoChanger->SetOutputDirection( originalDirection );
  infoChanger->SetChangeDirection( retdc & !this->GetElastix()->GetUseDirectionCosines() );
  infoChanger->SetInput( image );

  /** Create writer. */
  WriterPointer writer = WriterType::New();

  /** Setup the pipeline. */
  writer->SetInput( infoChanger->GetOutput() );
  writer->SetFileName( filename );
  writer->SetOutputComponentType( resultImagePixelType.c_str() );
  writer->SetUseCompression( doCompression );

  /** Do the writing. */
  if( showProgress )
  {
    xl::xout[ "coutonly" ] << std::flush;
    xl::xout[ "coutonly" ] << "\n  Writing image ..." << std::endl;
  }
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "ResamplerBase - AfterRegistrationBase()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing resampled image.\n";
    excp.SetDescription( err_str );

    /** Pass the exception to an higher level. */
    throw excp;
  }
} // end WriteResultImage()


/*
 * ******************* CreateItkResultImage ********************
 * \todo: avoid code duplication with WriteResultImage function
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::CreateItkResultImage( void )
{
  itk::DataObject::Pointer resultImage;

  /** Make sure the resampler is updated. */
  this->GetAsITKBaseType()->Modified();

  const auto progressObserver = BaseComponent::IsElastixLibrary() ?
    nullptr : ProgressCommandType::CreateAndConnect(*(this->GetAsITKBaseType()));

  /** Do the resampling. */
  try
  {
    this->GetAsITKBaseType()->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "ResamplerBase - WriteResultImage()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while resampling the image.\n";
    excp.SetDescription( err_str );

    /** Pass the exception to an higher level. */
    throw excp;
  }

  /** Check if ResampleInterpolator is the RayCastResampleInterpolator */
  typedef itk::AdvancedRayCastInterpolateImageFunction<  InputImageType,
    CoordRepType > RayCastInterpolatorType;

  const RayCastInterpolatorType * testptr = dynamic_cast< const
    RayCastInterpolatorType * >( this->GetAsITKBaseType()->GetInterpolator() );

  /** If RayCastResampleInterpolator is used reset the Transform to
   * overrule default Resampler settings */

  if( testptr )
  {
    this->GetAsITKBaseType()->SetTransform(
      ( const_cast< RayCastInterpolatorType * >( testptr ) )->GetTransform() );
  }

  /** Read output pixeltype from parameter the file. Replace possible " " with "_". */
  std::string resultImagePixelType = "short";
  this->m_Configuration->ReadParameter( resultImagePixelType,
    "ResultImagePixelType", 0, false );

  /** Typedef's for writing the output image. */
  typedef itk::ChangeInformationImageFilter<
    OutputImageType >                             ChangeInfoFilterType;

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false.
   */
  typename ChangeInfoFilterType::Pointer infoChanger = ChangeInfoFilterType::New();
  DirectionType originalDirection;
  bool          retdc = this->GetElastix()->GetOriginalFixedImageDirection( originalDirection );
  infoChanger->SetOutputDirection( originalDirection );
  infoChanger->SetChangeDirection( retdc & !this->GetElastix()->GetUseDirectionCosines() );
  infoChanger->SetInput( this->GetAsITKBaseType()->GetOutput() );

  typedef itk::CastImageFilter< InputImageType,
    itk::Image< char, InputImageType::ImageDimension > >            CastFilterChar;
  typedef itk::CastImageFilter< InputImageType,
    itk::Image< unsigned char, InputImageType::ImageDimension > >   CastFilterUChar;
  typedef itk::CastImageFilter< InputImageType,
    itk::Image< short, InputImageType::ImageDimension > >           CastFilterShort;
  typedef itk::CastImageFilter< InputImageType,
    itk::Image< unsigned short, InputImageType::ImageDimension > >  CastFilterUShort;
  typedef itk::CastImageFilter< InputImageType,
    itk::Image< int, InputImageType::ImageDimension > >             CastFilterInt;
  typedef itk::CastImageFilter< InputImageType,
    itk::Image< unsigned int, InputImageType::ImageDimension > >    CastFilterUInt;
  typedef itk::CastImageFilter< InputImageType,
    itk::Image< long, InputImageType::ImageDimension > >            CastFilterLong;
  typedef itk::CastImageFilter< InputImageType,
    itk::Image< unsigned long, InputImageType::ImageDimension > >  CastFilterULong;
  typedef itk::CastImageFilter< InputImageType,
    itk::Image< float, InputImageType::ImageDimension > >           CastFilterFloat;
  typedef itk::CastImageFilter< InputImageType,
    itk::Image< double, InputImageType::ImageDimension > >          CastFilterDouble;

  /** cast the image to the correct output image Type */
  if( resultImagePixelType.compare( "char" ) == 0 )
  {
    typename CastFilterChar::Pointer castFilter = CastFilterChar::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }
  if( resultImagePixelType.compare( "unsigned char" ) == 0 )
  {
    typename CastFilterUChar::Pointer castFilter = CastFilterUChar::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }
  else if( resultImagePixelType.compare( "short" ) == 0 )
  {
    typename CastFilterShort::Pointer castFilter = CastFilterShort::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }
  else if( resultImagePixelType.compare( "ushort" ) == 0 || resultImagePixelType.compare( "unsigned short" ) == 0 ) // <-- ushort for backwards compatibility
  {
    typename CastFilterUShort::Pointer castFilter = CastFilterUShort::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }
  else if( resultImagePixelType.compare( "int" ) == 0 )
  {
    typename CastFilterInt::Pointer castFilter = CastFilterInt::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }
  else if( resultImagePixelType.compare( "unsigned int" ) == 0 )
  {
    typename CastFilterUInt::Pointer castFilter = CastFilterUInt::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }
  else if( resultImagePixelType.compare( "long" ) == 0 )
  {
    typename CastFilterLong::Pointer castFilter = CastFilterLong::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }
  else if( resultImagePixelType.compare( "unsigned long" ) == 0 )
  {
    typename CastFilterULong::Pointer castFilter = CastFilterULong::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }
  else if( resultImagePixelType.compare( "float" ) == 0 )
  {
    typename CastFilterFloat::Pointer castFilter = CastFilterFloat::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }
  else if( resultImagePixelType.compare( "double" ) == 0 )
  {
    typename CastFilterDouble::Pointer castFilter = CastFilterDouble::New();
    castFilter->SetInput( infoChanger->GetOutput() );
    castFilter->Update();
    resultImage = castFilter->GetOutput();
  }

  if( resultImage.IsNull() )
  {
    itkExceptionMacro(
      << "Unable to cast result image: ResultImagePixelType must be one of \"char\", \"unsigned char\", \"short\", \"ushort\", \"unsigned short\", \"int\", \"unsigned int\", \"long\", \"unsigned long\", \"float\" or \"double\" but was \""
      << resultImagePixelType
      << "\"." );
  }

  //put image in container
  this->m_Elastix->SetResultImage( resultImage );

  if ( progressObserver != nullptr )
  {
    /** Disconnect from the resampler. */
    progressObserver->DisconnectObserver( this->GetAsITKBaseType() );
  }
} // end CreateItkResultImage()


/*
 * ************************* ReadFromFile ***********************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::ReadFromFile( void )
{
  /** Connect the components. */
  this->SetComponents();

  /** Get spacing, origin and size of the image to be produced by the resampler. */
  SpacingType     spacing;
  IndexType       index;
  OriginPointType origin;
  SizeType        size;
  DirectionType   direction;
  direction.SetIdentity();
  for( unsigned int i = 0; i < ImageDimension; i++ )
  {
    /** No default size. Read size from the parameter file. */
    this->m_Configuration->ReadParameter( size[ i ], "Size", i );

    /** Default index. Read index from the parameter file. */
    index[ i ] = 0;
    this->m_Configuration->ReadParameter( index[ i ], "Index", i );

    /** Default spacing. Read spacing from the parameter file. */
    spacing[ i ] = 1.0;
    this->m_Configuration->ReadParameter( spacing[ i ], "Spacing", i );

    /** Default origin. Read origin from the parameter file. */
    origin[ i ] = 0.0;
    this->m_Configuration->ReadParameter( origin[ i ], "Origin", i );

    /** Read direction cosines. Default identity */
    for( unsigned int j = 0; j < ImageDimension; j++ )
    {
      this->m_Configuration->ReadParameter( direction( j, i ),
        "Direction", i * ImageDimension + j );
    }
  }

  /** Check for image size. */
  unsigned int sum = 0;
  for( unsigned int i = 0; i < ImageDimension; i++ )
  {
    if( size[ i ] == 0 ) { sum++; }
  }
  if( sum > 0 )
  {
    xl::xout[ "error" ] << "ERROR: One or more image sizes are 0!" << std::endl;
    /** \todo quit program nicely. */
  }

  /** Set the region info to the same values as in the fixedImage. */
  this->GetAsITKBaseType()->SetSize( size );
  this->GetAsITKBaseType()->SetOutputStartIndex( index );
  this->GetAsITKBaseType()->SetOutputOrigin( origin );
  this->GetAsITKBaseType()->SetOutputSpacing( spacing );

  /** Set the direction cosines. If no direction cosines
   * should be used, set identity cosines, to simulate the
   * old ITK behavior.
   */
  if( !this->GetElastix()->GetUseDirectionCosines() )
  {
    direction.SetIdentity();
  }
  this->GetAsITKBaseType()->SetOutputDirection( direction );

  /** Set the DefaultPixelValue (for pixels in the resampled image
   * that come from outside the original (moving) image.
   */
  double defaultPixelValue = itk::NumericTraits< double >::Zero;
  bool   found             = this->m_Configuration->ReadParameter( defaultPixelValue,
    "DefaultPixelValue", 0, false );

  if( found )
  {
    this->GetAsITKBaseType()->SetDefaultPixelValue(
      static_cast< OutputPixelType >( defaultPixelValue ) );
  }

} // end ReadFromFile()


/**
 * ******************* WriteToFile ******************************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::WriteToFile( void ) const
{
  /** Write resampler specific things. */
  xl::xout[ "transpar" ] << std::endl << "// Resampler specific" << std::endl;

  /** Write the name of the resampler. */
  xl::xout[ "transpar" ] << "(Resampler \""
                         << this->elxGetClassName() << "\")" << std::endl;

  /** Write the DefaultPixelValue. */
  xl::xout[ "transpar" ] << "(DefaultPixelValue "
                         << static_cast< double >( this->GetAsITKBaseType()->GetDefaultPixelValue() )
                         << ")" << std::endl;

  /** Write the output image format. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter(
    resultImageFormat, "ResultImageFormat", 0, false );
  xl::xout[ "transpar" ] << "(ResultImageFormat \""
                         << resultImageFormat << "\")" << std::endl;

  /** Write output pixel type. */
  std::string resultImagePixelType = "short";
  this->m_Configuration->ReadParameter(
    resultImagePixelType, "ResultImagePixelType", 0, false );
  xl::xout[ "transpar" ] << "(ResultImagePixelType \""
                         << resultImagePixelType << "\")" << std::endl;

  /** Write compression flag. */
  std::string doCompression = "false";
  this->m_Configuration->ReadParameter(
    doCompression, "CompressResultImage", 0, false );
  xl::xout[ "transpar" ] << "(CompressResultImage \""
                         << doCompression << "\")" << std::endl;

} // end WriteToFile()


/**
 * ******************* CreateTransformParametersMap ******************************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::CreateTransformParametersMap( ParameterMapType * paramsMap ) const
{
  std::string                parameterName;
  std::vector< std::string > parameterValues;

  /** Write the name of this transform. */
  parameterName = "Resampler";
  parameterValues.push_back( this->elxGetClassName() );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write the DefaultPixelValue. */
  parameterName = "DefaultPixelValue";
  std::ostringstream strDefaultPixelValue;
  strDefaultPixelValue << this->GetAsITKBaseType()->GetDefaultPixelValue();
  parameterValues.push_back( strDefaultPixelValue.str() );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write the output image format. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter(
    resultImageFormat, "ResultImageFormat", 0, false );
  parameterName = "ResultImageFormat";
  parameterValues.push_back( resultImageFormat );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write output pixel type. */
  std::string resultImagePixelType = "short";
  this->m_Configuration->ReadParameter(
    resultImagePixelType, "ResultImagePixelType", 0, false );
  parameterName = "ResultImagePixelType";
  parameterValues.push_back( resultImagePixelType );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

  /** Write compression flag. */
  std::string doCompression = "false";
  this->m_Configuration->ReadParameter(
    doCompression, "CompressResultImage", 0, false );
  parameterName = "CompressResultImage";
  parameterValues.push_back( doCompression );
  paramsMap->insert( make_pair( parameterName, parameterValues ) );
  parameterValues.clear();

} // end CreateTransformParametersMap()


/**
 * ******************* ReleaseMemory ********************
 */

template< class TElastix >
void
ResamplerBase< TElastix >
::ReleaseMemory( void )
{
  /** Release some memory. Sometimes it is not possible to
   * resample and write an image, because too much memory is consumed by
   * elastix. Releasing some memory at this point helps a lot.
   */

  /** Release more memory, but only if this is the final elastix level. */
  if( this->GetConfiguration()->GetElastixLevel() + 1
    == this->GetConfiguration()->GetTotalNumberOfElastixLevels() )
  {
    /** Release fixed image memory. */
    const unsigned int nofi = this->GetElastix()->GetNumberOfFixedImages();
    for( unsigned int i = 0; i < nofi; ++i )
    {
      this->GetElastix()->GetFixedImage( i )->ReleaseData();
    }

    /** Release fixed mask image memory. */
    const unsigned int nofm = this->GetElastix()->GetNumberOfFixedMasks();
    for( unsigned int i = 0; i < nofm; ++i )
    {
      if( this->GetElastix()->GetFixedMask( i ) != 0 )
      {
        this->GetElastix()->GetFixedMask( i )->ReleaseData();
      }
    }

    /** Release moving mask image memory. */
    const unsigned int nomm = this->GetElastix()->GetNumberOfMovingMasks();
    for( unsigned int i = 0; i < nomm; ++i )
    {
      if( this->GetElastix()->GetMovingMask( i ) != 0 )
      {
        this->GetElastix()->GetMovingMask( i )->ReleaseData();
      }
    }

  } // end if final elastix level

  /** The B-spline interpolator stores a coefficient image of doubles the
   * size of the moving image. We clear it by setting the input image to
   * zero. The interpolator is not needed anymore, since we have the
   * resampler interpolator.
   */
  this->GetElastix()->GetElxInterpolatorBase()->GetAsITKBaseType()->SetInputImage( 0 );

  // Clear ImageSampler, metric, optimizer, interpolator, registration, internal images?

} // end ReleaseMemory()


} // end namespace elastix

#endif // end #ifndef __elxResamplerBase_hxx
