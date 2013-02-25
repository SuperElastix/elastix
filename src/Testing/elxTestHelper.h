/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxTestHelper_h
#define __elxTestHelper_h

#include <string>
#include <vector>
#include <iomanip>
#include <time.h>

#include "itkImage.h"
#include "itkTimeProbe.h"
#include "itkStdStreamLogOutput.h"
#include "itkLogger.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageToImageFilter.h"

#include "elxTestOutputWindow.h"

namespace elastix
{
//------------------------------------------------------------------------------
void SetupForDebugging()
{
  itk::TestOutputWindow::Pointer tow = itk::TestOutputWindow::New();
  itk::OutputWindow::SetInstance( tow );

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  itk::Object::SetGlobalWarningDisplay( true );
  std::cout << "INFO: test called itk::Object::SetGlobalWarningDisplay(true)\n";
#endif
}

//------------------------------------------------------------------------------
void ITKObjectEnableWarnings( itk::Object *object )
{
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  object->SetDebug( true );
  std::cout << "INFO: " << object->GetNameOfClass() << " called SetDebug(true);\n";
#endif
}

//------------------------------------------------------------------------------
// Get current date, format is m-d-y
const std::string GetCurrentDate()
{
  time_t    now = time( 0 );
  struct tm tstruct;
  char      buf[80];

#if !defined( _WIN32 ) || defined( __CYGWIN__ )
  tstruct = *localtime( &now );
#else
  localtime_s( &tstruct, &now );
#endif

  // Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
  // for more information about date/time format
  strftime( buf, sizeof( buf ), "%m-%d-%y", &tstruct );

  return buf;
}

//------------------------------------------------------------------------------
// Helper function to compute RMSE
template< class TScalarType, class CPUImageType, class GPUImageType >
TScalarType ComputeRMSE( const CPUImageType *cpuImage, const GPUImageType *gpuImage )
{
  itk::ImageRegionConstIterator< CPUImageType > cit(
    cpuImage, cpuImage->GetLargestPossibleRegion() );
  itk::ImageRegionConstIterator< GPUImageType > git(
    gpuImage, gpuImage->GetLargestPossibleRegion() );

  TScalarType rmse = 0.0;

  for ( cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git )
  {
    TScalarType err = static_cast< TScalarType >( cit.Get() ) - static_cast< TScalarType >( git.Get() );
    rmse += err * err;
  }
  rmse = vcl_sqrt( rmse / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );
  return rmse;
} // end ComputeRMSE()

//------------------------------------------------------------------------------
// Helper function to get test result from output images
template< class TScalarType, class CPUImageType, class GPUImageType >
void GetTestOutputResult(
  const CPUImageType *cpuImage, const GPUImageType *gpuImage,
  const float RMSError, TScalarType & rmsError, bool & testPassed,
  const bool skipCPU, const bool skipGPU,
  const bool updateExceptionCPU, const bool updateExceptionGPU )
{
  rmsError = 0.0;
  testPassed = true;
  if ( updateExceptionCPU || updateExceptionGPU )
  {
    testPassed = false;
  }
  else
  {
    if ( !skipCPU && !skipGPU && cpuImage && gpuImage )
    {
      rmsError = ComputeRMSE< TScalarType, CPUImageType, GPUImageType >
                   ( cpuImage, gpuImage );
      std::cout << "RMS Error: " << std::fixed << std::setprecision( 8 ) << rmsError << std::endl;
      testPassed = ( rmsError <= RMSError );
    }
  }
}

//------------------------------------------------------------------------------
// Helper function to get test result from filters
template< class TScalarType,
          class ImageToImageFilterType, class OutputImage >
void GetTestFilterResult(
  typename ImageToImageFilterType::Pointer & cpuFilter,
  typename ImageToImageFilterType::Pointer & gpuFilter,
  const float RMSError, TScalarType & rmsError, bool & testPassed,
  const bool skipCPU, const bool skipGPU,
  const bool updateExceptionCPU, const bool updateExceptionGPU,
  const unsigned int outputindex = 0 )
{
  rmsError = 0.0;
  testPassed = true;
  if ( updateExceptionCPU || updateExceptionGPU )
  {
    testPassed = false;
  }
  else
  {
    if ( !skipCPU && !skipGPU && cpuFilter.IsNotNull() && gpuFilter.IsNotNull() )
    {
      if ( outputindex == 0 )
      {
        rmsError = ComputeRMSE< TScalarType, OutputImage, OutputImage >
                     ( cpuFilter->GetOutput(), gpuFilter->GetOutput() );
      }
      else
      {
        rmsError = ComputeRMSE< TScalarType, OutputImage, OutputImage >
                     ( cpuFilter->GetOutput( outputindex ), gpuFilter->GetOutput( outputindex ) );
      }

      std::cout << "RMS Error: " << std::fixed << std::setprecision( 8 ) << rmsError << std::endl;
      testPassed = ( rmsError <= RMSError );
    }
  }
}

//------------------------------------------------------------------------------
// Helper function to compute RMSE with masks
template< class TScalarType, class CPUImageType, class GPUImageType, class MaskImageType >
TScalarType ComputeRMSE( const CPUImageType *cpuImage, const GPUImageType *gpuImage,
                         const MaskImageType *cpuImageMask, const MaskImageType *gpuImageMask )
{
  itk::ImageRegionConstIterator< CPUImageType > cit(
    cpuImage, cpuImage->GetLargestPossibleRegion() );
  itk::ImageRegionConstIterator< GPUImageType > git(
    gpuImage, gpuImage->GetLargestPossibleRegion() );

  itk::ImageRegionConstIterator< MaskImageType > mcit(
    cpuImageMask, cpuImageMask->GetLargestPossibleRegion() );
  itk::ImageRegionConstIterator< MaskImageType > mgit(
    gpuImageMask, gpuImageMask->GetLargestPossibleRegion() );

  TScalarType rmse = 0.0;

  for ( cit.GoToBegin(), git.GoToBegin(), mcit.GoToBegin(), mgit.GoToBegin();
        !cit.IsAtEnd(); ++cit, ++git, ++mcit, ++mgit )
  {
    if ( ( mcit.Get() == itk::NumericTraits< typename MaskImageType::PixelType >::One )
         && ( mgit.Get() == itk::NumericTraits< typename MaskImageType::PixelType >::One ) )
    {
      TScalarType err = static_cast< TScalarType >( cit.Get() ) - static_cast< TScalarType >( git.Get() );
      rmse += err * err;
    }
  }
  rmse = vcl_sqrt( rmse / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );
  return rmse;
} // end ComputeRMSE()

//----------------------------------------------------------------------------
// Write log file in Microsoft Excel semicolon separated format.
template< class ImageType >
void WriteLog(
  const std::string & filename,
  const unsigned int dim,
  const typename ImageType::SizeType & imagesize,
  const double rmsError,
  const bool testPassed,
  const bool exceptionGPU,
  const unsigned int numThreads,
  const unsigned int runTimes,
  const std::string filterName,
  const itk::TimeProbe::TimeStampType cpuTime,
  const itk::TimeProbe::TimeStampType gpuTime,
  const std::string & comments = "" )
{
  // Create an ITK StdStreamLogOutputs
  itk::StdStreamLogOutput::Pointer foutput = itk::StdStreamLogOutput::New();
  std::ofstream                    fout( filename.c_str(), std::ios_base::app );

  foutput->SetStream( fout );

  // Create an ITK Logger
  itk::Logger::Pointer                 logger = itk::Logger::New();
  itk::LoggerBase::TimeStampFormatType timeStampFormat = itk::LoggerBase::HUMANREADABLE;
  logger->SetTimeStampFormat( timeStampFormat );
  std::string humanReadableFormat = "%b %d %Y %H:%M:%S";
  logger->SetHumanReadableFormat( humanReadableFormat );

  // Setting the logger
  logger->SetName( "CPUGPULogger" );
  logger->SetPriorityLevel( itk::LoggerBase::INFO );
  logger->SetLevelForFlushing( itk::LoggerBase::CRITICAL );
  logger->AddLogOutput( foutput );

  std::ostringstream logoss;
  logoss << " ; ";

  logoss << filterName << " CPU vs GPU ;";

  logoss << " Dimension ; " << dim << " ;";
  logoss << " ImageSize ; ";

  for ( unsigned int i = 0; i < dim; i++ )
  {
    logoss << imagesize.GetSize()[i];
    if ( i < dim - 1 ) { logoss << "x"; }
  }

  logoss << " ;";
  logoss << " CPU(" << numThreads << ") ; " << cpuTime << " ;";

  if ( !exceptionGPU )
  {
    logoss << " GPU ; " << gpuTime << " ;";
  }
  else
  {
    logoss << " GPU ; na ;";
  }

  if ( !exceptionGPU )
  {
    if ( gpuTime != 0.0 )
    {
      logoss << " CPU/GPU SpeedRatio ; " << ( cpuTime / gpuTime ) << " ;";
    }
    else
    {
      logoss << " CPU/GPU SpeedRatio ; 0 ;";
    }
  }
  else
  {
    logoss << " CPU/GPU SpeedRatio ; na ;";
  }

  logoss << " RMSError ; "   << rmsError << " ;";
  logoss << " TestPassed ; " << ( testPassed ? "Yes" : "No" ) << " ;";
  logoss << " RunTimes ; "   << runTimes << " ;";

  if ( comments.size() > 0 )
  {
    logoss << " Comments ; " << comments << " ;";
  }
  else
  {
    logoss << " Comments ; none ;";
  }

  logoss << std::endl;

  // Writing by the logger
  logger->Write( itk::LoggerBase::INFO, logoss.str() );
  logger->Flush();
}
} // end namespace elastix

#endif // end #ifndef __elxTestHelper_h
