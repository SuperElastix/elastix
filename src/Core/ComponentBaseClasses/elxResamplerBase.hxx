/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxResamplerBase_hxx
#define __elxResamplerBase_hxx

#include "elxResamplerBase.h"
#include "itkImageFileCastWriter.h"
#include "elxTimer.h"

namespace elastix
{
  using namespace itk;


  /*
   * ******************* BeforeRegistrationBase *******************
   */
  
  template<class TElastix>
    void ResamplerBase<TElastix>
    ::BeforeRegistrationBase( void )
  {
    /** Connect the components. */
    this->SetComponents();
    
    /** Set the size of the image to be produced by the resampler. */
    
    /** Get a pointer to the fixedImage. 
     * \todo make it a cast to the fixed image type
     */
    typedef typename ElastixType::FixedImageType FixedImageType;
    FixedImageType * fixedImage =	this->m_Elastix->GetFixedImage();
    
    /** Set the region info to the same values as in the fixedImage. */
    this->GetAsITKBaseType()->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
    this->GetAsITKBaseType()->SetOutputStartIndex( fixedImage->GetLargestPossibleRegion().GetIndex() );
    this->GetAsITKBaseType()->SetOutputOrigin( fixedImage->GetOrigin() );
    this->GetAsITKBaseType()->SetOutputSpacing( fixedImage->GetSpacing() );
    this->GetAsITKBaseType()->SetOutputDirection( fixedImage->GetDirection() );
    
    /** Set the DefaultPixelValue (for pixels in the resampled image
     * that come from outside the original (moving) image.
     */
    double defaultPixelValueDouble = NumericTraits<double>::Zero;
    int defaultPixelValueInt = NumericTraits<int>::Zero;
    int retd = this->m_Configuration->ReadParameter(
      defaultPixelValueDouble, "DefaultPixelValue", 0, true );
    int reti = this->m_Configuration->ReadParameter(
      defaultPixelValueInt, "DefaultPixelValue", 0, true );
    
    /** Set the defaultPixelValue. int values overrule double values. */
    if ( retd == 0 )
    {
      this->GetAsITKBaseType()->SetDefaultPixelValue(
        static_cast<OutputPixelType>( defaultPixelValueDouble ) );
    }
    if ( reti == 0 )
    {
      this->GetAsITKBaseType()->SetDefaultPixelValue(
        static_cast<OutputPixelType>( defaultPixelValueInt ) );
    }
    if ( reti != 0 && retd != 0 )
    {
      this->GetAsITKBaseType()->SetDefaultPixelValue(
        static_cast<OutputPixelType>( defaultPixelValueInt ) );
    }

  } // end BeforeRegistrationBase


  /*
   * ******************* AfterEachResolutionBase ********************
   */
  
  template<class TElastix>
    void ResamplerBase<TElastix>
    ::AfterEachResolutionBase( void )
  {
    /** Set the final transform parameters. */
    this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

    /** What is the current resolution level? */
    unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

    /** Decide whether or not to write the result image this resolution. */
    bool writeResultImageThisResolution = false;
    this->m_Configuration->ReadParameter(	writeResultImageThisResolution,
      "WriteResultImageAfterEachResolution", "", level, 0, true );

    /** Writing result image. */
    if ( writeResultImageThisResolution )
    {
      /** Create a name for the final result. */
      std::string resultImageFormat = "mhd";
      this->m_Configuration->ReadParameter(	resultImageFormat, "ResultImageFormat", 0, true);
      std::ostringstream makeFileName( "" );
      makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
        << "result." << this->m_Configuration->GetElastixLevel()
        << ".R" << level
        << "." << resultImageFormat;

      /** Time the resampling. */
      typedef tmr::Timer TimerType;
      TimerType::Pointer timer = TimerType::New();
      timer->StartTimer();

      /** Apply the final transform, and save the result. */
      elxout << "Applying transform this resolution ..." << std::endl;
      try
      {
        this->WriteResultImage( makeFileName.str().c_str() );
      }
      catch( itk::ExceptionObject & excp )
      {
        xl::xout["error"] << "Exception caught: " << std::endl;
        xl::xout["error"] << excp
          << "Resuming elastix." << std::endl;
      }

      /** Print the elapsed time for the resampling. */
      timer->StopTimer();
      elxout << "  Applying transform took "
        << static_cast<long>( timer->GetElapsedClockSec() )
        << " s." << std::endl;

    } // end if

  } // end AfterEachResolutionBase()


  /*
   * ******************* AfterRegistrationBase ********************
   */
  
  template<class TElastix>
    void ResamplerBase<TElastix>
    ::AfterRegistrationBase(void)
  {
    /** Set the final transform parameters. */
    this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

    /** Decide whether or not to write the result image. */
    std::string writeResultImage = "true";
    this->m_Configuration->ReadParameter(	writeResultImage, "WriteResultImage", 0 );

    /** Release some memory, here already. Sometimes it is not possible to
     * resample and write an image, because too much memory is consumed by
     * elastix. Releasing the memory of the pyramids and the fixed image at
     * this point helps a lot.
     */
    unsigned int numberOfOutputs = this->GetElastix()
      ->GetElxFixedImagePyramidBase()->GetAsITKBaseType()->GetNumberOfOutputs();
    for ( unsigned int i = 0; i < numberOfOutputs; ++i )
    {
      this->GetElastix()->GetElxFixedImagePyramidBase()->GetAsITKBaseType()
        ->GetOutput( i )->ReleaseData();
    }
    numberOfOutputs = this->GetElastix()
      ->GetElxMovingImagePyramidBase()->GetAsITKBaseType()->GetNumberOfOutputs();
    for ( unsigned int i = 0; i < numberOfOutputs; ++i )
    {
      this->GetElastix()->GetElxMovingImagePyramidBase()->GetAsITKBaseType()
        ->GetOutput( i )->ReleaseData();
    }

    /** Only release fixed image memory if this is the final elastix level. */
    if ( this->GetConfiguration()->GetElastixLevel() + 1
      == this->GetConfiguration()->GetTotalNumberOfElastixLevels() )
    {
      this->GetElastix()->GetFixedImage()->ReleaseData();
    }
 
    /** Writing result image. */
    if ( writeResultImage == "true" )
    {
      /** Create a name for the final result. */
      std::string resultImageFormat = "mhd";
      this->m_Configuration->ReadParameter(	resultImageFormat, "ResultImageFormat", 0);
      std::ostringstream makeFileName( "" );
      makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
        << "result." << this->m_Configuration->GetElastixLevel()
        << "." << resultImageFormat;

      /** Time the resampling. */
      typedef tmr::Timer TimerType;
      TimerType::Pointer timer = TimerType::New();
      timer->StartTimer();

      /** Apply the final transform, and save the result,
       * by calling WriteResultImage.
       */
      elxout << "\nApplying final transform ..." << std::endl;
      try
      {
        this->WriteResultImage( makeFileName.str().c_str() );
      }
      catch( itk::ExceptionObject & excp )
      {
        xl::xout["error"] << "Exception caught: " << std::endl;
        xl::xout["error"] << excp
          << "Resuming elastix." << std::endl;
      }

      /** Print the elapsed time for the resampling. */
      timer->StopTimer();
      elxout << "  Applying final transform took "
        << static_cast<long>( timer->GetElapsedClockSec() )
        << " s." << std::endl;
    }
    else
    {
      /** Do not apply the final transform. */
      elxout << std::endl
        << "Skipping applying final transform, no resulting output image generated."
        << std::endl;
    } // end if

  } // end AfterRegistrationBase()


  /*
   * *********************** SetComponents ************************
   */
  
  template <class TElastix>
    void ResamplerBase<TElastix>
    ::SetComponents(void)
  {
    /** Set the transform, the interpolator and the inputImage
     * (which is the moving image).
     */
    this->GetAsITKBaseType()->SetTransform( dynamic_cast<TransformType *>(
      this->m_Elastix->GetElxTransformBase() ) );
    
    this->GetAsITKBaseType()->SetInterpolator( dynamic_cast<InterpolatorType *>(
      this->m_Elastix->GetElxResampleInterpolatorBase() ) );
    
    this->GetAsITKBaseType()->SetInput( dynamic_cast<InputImageType *>(
      this->m_Elastix->GetMovingImage() ) );
    
  } // end SetComponents()


  /*
   * ******************* WriteResultImage ********************
   */
  
  template<class TElastix>
    void ResamplerBase<TElastix>
    ::WriteResultImage( const char * filename )
  {
    /** Make sure the resampler is updated. */
    this->GetAsITKBaseType()->Modified();

    /** Add a progress observer to the resampler. */
    typename ProgressCommandType::Pointer progressObserver = ProgressCommandType::New();
    progressObserver->ConnectObserver( this->GetAsITKBaseType() );
    progressObserver->SetStartString( "  Progress: " );
    progressObserver->SetEndString( "%" );

    /** Read output pixeltype from parameter the file. Replace possible " " with "_". */
    std::string resultImagePixelType = "short";
    this->m_Configuration->ReadParameter(	resultImagePixelType, "ResultImagePixelType", 0, true );
    std::basic_string<char>::size_type pos = resultImagePixelType.find( " " );
    const std::basic_string<char>::size_type npos = std::basic_string<char>::npos;
    if ( pos != npos ) resultImagePixelType.replace( pos, 1, "_" );

    /** Read from the parameter file if compression is desired. */
    bool doCompression = false;
    this->m_Configuration->ReadParameter(
      doCompression, "CompressResultImage", 0, true );
    
    /** Typedef's for writing the output image. */
    typedef ImageFileCastWriter< OutputImageType >	WriterType;
    typedef typename WriterType::Pointer					  WriterPointer;

    /** Create writer. */
    WriterPointer writer = WriterType::New();

    /** Setup the pipeline. */
    writer->SetInput( this->GetAsITKBaseType()->GetOutput() );
    writer->SetFileName( filename );
    writer->SetOutputComponentType( resultImagePixelType.c_str() );
    writer->SetUseCompression( doCompression );

    /** Do the writing. */
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

    /** Disconnect from the resampler. */
    progressObserver->DisconnectObserver( this->GetAsITKBaseType() );

  } // end WriteResultImage()


  /*
   * ************************* ReadFromFile ***********************
   */
  
  template<class TElastix>
    void ResamplerBase<TElastix>
    ::ReadFromFile( void )
  {
    /** Connect the components. */
    this->SetComponents();
    
    /** Get spacing, origin and size of the image to be produced by the resampler. */
    SpacingType			spacing;
    IndexType				index;
    OriginPointType	origin;
    SizeType				size;
    DirectionType   direction;
    direction.SetIdentity();
    for ( unsigned int i = 0; i < ImageDimension; i++ )
    {
      /** No default size. Read size from the parameter file. */
      this->m_Configuration->ReadParameter(	size[ i ], "Size", i );

      /** Default index. Read index from the parameter file. */
      index[ i ] = 0;
      this->m_Configuration->ReadParameter(	index[ i ], "Index", i );

      /** Default spacing. Read spacing from the parameter file. */
      spacing[ i ] = 1.0;
      this->m_Configuration->ReadParameter(	spacing[ i ], "Spacing", i );

      /** Default origin. Read origin from the parameter file. */
      origin[ i ] = 0.0;
      this->m_Configuration->ReadParameter(	origin[ i ], "Origin", i );

      /** Read direction cosines. Default identity */
      for ( unsigned int j = 0; j < ImageDimension; j++ )
      {
        this->m_Configuration->ReadParameter(	direction( j, i ), "Direction", i * ImageDimension + j );        
      }
    }

    /** Check for image size. */
    unsigned int sum = 0;
    for ( unsigned int i = 0; i < ImageDimension; i++ )
    {
      if ( size[ i ] == 0 ) sum++;
    }
    if ( sum > 0 )
    {
      xl::xout["error"] << "ERROR: One or more image sizes are 0!" << std::endl;
      /** \todo quit program nicely. */
    }
    
    /** Set the region info to the same values as in the fixedImage. */
    this->GetAsITKBaseType()->SetSize( size );
    this->GetAsITKBaseType()->SetOutputStartIndex( index );
    this->GetAsITKBaseType()->SetOutputOrigin( origin );
    this->GetAsITKBaseType()->SetOutputSpacing( spacing );
    this->GetAsITKBaseType()->SetOutputDirection( direction );
    
    /** Set the DefaultPixelValue (for pixels in the resampled image
     * that come from outside the original (moving) image.
     */
    double defaultPixelValueDouble = NumericTraits<double>::Zero;
    int defaultPixelValueInt = NumericTraits<int>::Zero;
    int retd = this->m_Configuration->ReadParameter( defaultPixelValueDouble, "DefaultPixelValue", 0, true );
    int reti = this->m_Configuration->ReadParameter( defaultPixelValueInt, "DefaultPixelValue", 0, true );
    
    /** Set the defaultPixelValue. int values overrule double values in case
     * both have been supplied.
     */
    if ( retd == 0 )
    {
      this->GetAsITKBaseType()->SetDefaultPixelValue(
        static_cast<OutputPixelType>( defaultPixelValueDouble ) );
    }
    if ( reti == 0 )
    {
      this->GetAsITKBaseType()->SetDefaultPixelValue(
        static_cast<OutputPixelType>( defaultPixelValueInt ) );
    }
    if ( reti != 0 && retd != 0 )
    {
      this->GetAsITKBaseType()->SetDefaultPixelValue(
        static_cast<OutputPixelType>( defaultPixelValueInt ) );
    }
    
  } // end ReadFromFile()


  /**
   * ******************* WriteToFile ******************************
   */

  template <class TElastix>
    void ResamplerBase<TElastix>
    ::WriteToFile( void )
  {
    /** Write Resampler specific things. */
    xl::xout["transpar"] << std::endl << "// Resampler specific" << std::endl;

    /** Write the name of the Resampler. */
    xl::xout["transpar"] << "(Resampler \""
      << this->elxGetClassName() << "\")" << std::endl;

    /** Write the DefaultPixelValue. */
    xl::xout["transpar"] << "(DefaultPixelValue "
      << this->GetAsITKBaseType()->GetDefaultPixelValue() << ")" << std::endl;

    /** Write the output image format. */
    std::string resultImageFormat = "mhd";
    this->m_Configuration->ReadParameter(
      resultImageFormat, "ResultImageFormat", 0, true );
    xl::xout["transpar"] << "(ResultImageFormat \""
      << resultImageFormat << "\")" << std::endl;

    /** Write output pixeltype. */
    std::string resultImagePixelType = "short";
    this->m_Configuration->ReadParameter(
      resultImagePixelType, "ResultImagePixelType", 0, true );
    xl::xout["transpar"] << "(ResultImagePixelType \""
      << resultImagePixelType << "\")" << std::endl;

    /** Write compression flag. */
    std::string doCompression = "false";
    this->m_Configuration->ReadParameter(
      doCompression, "CompressResultImage", 0, true );
    xl::xout["transpar"] << "(CompressResultImage \""
      << doCompression << "\")" << std::endl;

  } // end WriteToFile()


} // end namespace elastix


#endif // end #ifndef __elxResamplerBase_hxx

