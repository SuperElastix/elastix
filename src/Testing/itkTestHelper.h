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
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//
#ifndef __itkTestHelper_h
#define __itkTestHelper_h

#include <string>
#include <vector>
#include <iomanip>
#include <time.h>
#include <fstream>
#include <itksys/SystemTools.hxx>

#if defined( _WIN32 )
#include <io.h>
#endif

// ITK includes
#include "itkImage.h"
#include "itkTimeProbe.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageToImageFilter.h"

// OpenCL includes
#include "itkOpenCLContext.h"
#include "itkOpenCLDevice.h"
#include "itkOpenCLLogger.h"
#include "itkOpenCLKernels.h"

#include "itkTestOutputWindow.h"

//------------------------------------------------------------------------------
// Definition of the OCLImageDims
struct OCLImageDims
{
  itkStaticConstMacro( Support1D, bool, false );
  itkStaticConstMacro( Support2D, bool, false );
  itkStaticConstMacro( Support3D, bool, true );
};

#define ITK_OPENCL_COMPARE( actual, expected )                                    \
  if( !itk::Compare( actual, expected, #actual, #expected, __FILE__, __LINE__ ) ) \
    itkGenericExceptionMacro( << "Compared values are not the same" )             \

namespace itk
{
//------------------------------------------------------------------------------
template< typename T >
inline bool
Compare( T const & t1, T const & t2, const char * actual, const char * expected,
  const char * file, int line )
{
  return ( t1 == t2 ) ? true : false;
}


//------------------------------------------------------------------------------
bool
CreateContext()
{
  // Create and check OpenCL context
  OpenCLContext::Pointer context = OpenCLContext::GetInstance();

#if defined( OPENCL_USE_INTEL_CPU ) || defined( OPENCL_USE_AMD_CPU )
  context->Create( OpenCLContext::DevelopmentSingleMaximumFlopsDevice );
#else
  context->Create( OpenCLContext::SingleMaximumFlopsDevice );
#endif

  if( !context->IsCreated() )
  {
    std::cerr << "OpenCL-enabled device is not present." << std::endl;
    return false;
  }

  return true;
}


//------------------------------------------------------------------------------
void
ReleaseContext()
{
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  if( context->IsCreated() )
  {
    context->Release();
  }
}


//------------------------------------------------------------------------------
void
CreateOpenCLLogger( const std::string & prefixFileName )
{
  /** Create the OpenCL logger */
  OpenCLLogger::Pointer logger = OpenCLLogger::GetInstance();
  logger->SetLogFileNamePrefix( prefixFileName );
  logger->SetOutputDirectory( OpenCLKernelsDebugDirectory );
}


//------------------------------------------------------------------------------
void
SetupForDebugging()
{
  TestOutputWindow::Pointer tow = TestOutputWindow::New();
  OutputWindow::SetInstance( tow );

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  Object::SetGlobalWarningDisplay( true );
  std::cout << "INFO: test called Object::SetGlobalWarningDisplay(true)\n";
#endif
}


//------------------------------------------------------------------------------
void
ITKObjectEnableWarnings( Object * object )
{
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  object->SetDebug( true );
  std::cout << "INFO: " << object->GetNameOfClass() << " called SetDebug(true);\n";
#endif
}


//------------------------------------------------------------------------------
// Get current date, format is m-d-y
const std::string
GetCurrentDate()
{
  time_t    now = time( 0 );
  struct tm tstruct;
  char      buf[ 80 ];

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
// Get the name of the log file
const std::string
GetLogFileName()
{
  OpenCLContext::Pointer context    = OpenCLContext::GetInstance();
  std::string            fileName   = "CPUGPULog-" + itk::GetCurrentDate() + "-";
  std::string            deviceName = context->GetDefaultDevice().GetName();
  std::replace( deviceName.begin(), deviceName.end(), ' ', '-' ); // replace spaces
  deviceName.erase( deviceName.end() - 1, deviceName.end() );     // remove end of line

  switch( context->GetDefaultDevice().GetDeviceType() )
  {
    case OpenCLDevice::Default:
      fileName.append( "Default" ); break;
    case OpenCLDevice::CPU:
      fileName.append( "CPU" ); break;
    case OpenCLDevice::GPU:
      fileName.append( "GPU" ); break;
    case OpenCLDevice::Accelerator:
      fileName.append( "Accelerator" ); break;
    case OpenCLDevice::All:
      fileName.append( "All" ); break;
    default:
      fileName.append( "Unknown" ); break;
  }

  fileName.append( "-" );
  fileName.append( deviceName );
  fileName.append( ".txt" );

  return fileName;
}


//------------------------------------------------------------------------------
// Helper function to compute RMSE
template< class TScalarType, class CPUImageType, class GPUImageType >
TScalarType
ComputeRMSE( const CPUImageType * cpuImage, const GPUImageType * gpuImage,
  TScalarType & rmsRelative )
{
  ImageRegionConstIterator< CPUImageType > cit(
    cpuImage, cpuImage->GetLargestPossibleRegion() );
  ImageRegionConstIterator< GPUImageType > git(
    gpuImage, gpuImage->GetLargestPossibleRegion() );

  TScalarType rmse          = 0.0;
  TScalarType sumCPUSquared = 0.0;

  for( cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git )
  {
    TScalarType cpu = static_cast< TScalarType >( cit.Get() );
    TScalarType err = cpu - static_cast< TScalarType >( git.Get() );
    rmse          += err * err;
    sumCPUSquared += cpu * cpu;
  }

  rmse        = vcl_sqrt( rmse / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );
  rmsRelative = rmse / vcl_sqrt( sumCPUSquared / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );

  return rmse;
} // end ComputeRMSE()


//------------------------------------------------------------------------------
// Helper function to compute RMSE
template< class TScalarType, class CPUImageType, class GPUImageType >
TScalarType
ComputeRMSE2( const CPUImageType * cpuImage, const GPUImageType * gpuImage,
  const float & threshold )
{
  ImageRegionConstIterator< CPUImageType > cit(
    cpuImage, cpuImage->GetLargestPossibleRegion() );
  ImageRegionConstIterator< GPUImageType > git(
    gpuImage, gpuImage->GetLargestPossibleRegion() );

  TScalarType rmse = 0.0;

  for( cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git )
  {
    TScalarType err = static_cast< TScalarType >( cit.Get() ) - static_cast< TScalarType >( git.Get() );
    if( err > threshold )
    {
      rmse += err * err;
    }
  }
  rmse = vcl_sqrt( rmse / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );
  return rmse;
} // end ComputeRMSE2()


//------------------------------------------------------------------------------
// Helper function to get test result from output images
template< class TScalarType, class CPUImageType, class GPUImageType >
void
GetTestOutputResult(
  const CPUImageType * cpuImage, const GPUImageType * gpuImage,
  const float allowedRMSerror, TScalarType & rmsError, TScalarType & rmsRelative,
  bool & testPassed,
  const bool skipCPU, const bool skipGPU,
  const TimeProbe::TimeStampType cpuTime,
  const TimeProbe::TimeStampType gpuTime,
  const bool updateExceptionCPU, const bool updateExceptionGPU )
{
  rmsError    = 0.0;
  rmsRelative = 0.0;
  testPassed  = true;
  if( updateExceptionCPU || updateExceptionGPU )
  {
    testPassed = false;
  }
  else
  {
    if( !skipCPU && !skipGPU && cpuImage && gpuImage )
    {
      rmsError = ComputeRMSE< TScalarType, CPUImageType, GPUImageType >
          ( cpuImage, gpuImage, rmsRelative );

      std::cout << ", speed up " << ( cpuTime / gpuTime ) << std::endl;
      std::cout << std::fixed << std::setprecision( 8 );
      std::cout << "Maximum allowed RMS Error: " << allowedRMSerror << std::endl;
      std::cout << "Computed real   RMS Error: " << rmsError << std::endl;
      std::cout << "Computed real  nRMS Error: " << rmsRelative << std::endl;

      testPassed = ( rmsError <= allowedRMSerror );
    }
  }
}


//------------------------------------------------------------------------------
// Helper function to get test result from filters
template< class TScalarType,
class ImageToImageFilterType, class OutputImage >
void
GetTestFilterResult(
  typename ImageToImageFilterType::Pointer & cpuFilter,
  typename ImageToImageFilterType::Pointer & gpuFilter,
  const float allowedRMSerror, TScalarType & rmsError, TScalarType & rmsRelative,
  bool & testPassed, const bool skipCPU, const bool skipGPU,
  const TimeProbe::TimeStampType cpuTime,
  const TimeProbe::TimeStampType gpuTime,
  const bool updateExceptionCPU, const bool updateExceptionGPU,
  const unsigned int outputindex = 0 )
{
  rmsError   = 0.0;
  testPassed = true;
  if( updateExceptionCPU || updateExceptionGPU )
  {
    testPassed = false;
  }
  else
  {
    if( !skipCPU && !skipGPU && cpuFilter.IsNotNull() && gpuFilter.IsNotNull() )
    {
      if( outputindex == 0 )
      {
        rmsError = ComputeRMSE< TScalarType, OutputImage, OutputImage >
            ( cpuFilter->GetOutput(), gpuFilter->GetOutput(), rmsRelative );
      }
      else
      {
        rmsError = ComputeRMSE< TScalarType, OutputImage, OutputImage >
            ( cpuFilter->GetOutput( outputindex ), gpuFilter->GetOutput( outputindex ), rmsRelative );
      }

      std::cout << ", speed up " << ( cpuTime / gpuTime ) << std::endl;
      std::cout << std::fixed << std::setprecision( 8 );
      std::cout << "Maximum allowed RMS Error: " << allowedRMSerror << std::endl;
      std::cout << "Computed real   RMS Error: " << rmsError << std::endl;
      std::cout << "Computed real  nRMS Error: " << rmsRelative << std::endl;

      testPassed = ( rmsError <= allowedRMSerror );
    }
  }
}


//------------------------------------------------------------------------------
// Helper function to compute RMSE with masks
template< class TScalarType, class CPUImageType, class GPUImageType, class MaskImageType >
TScalarType
ComputeRMSE( const CPUImageType * cpuImage, const GPUImageType * gpuImage,
  const MaskImageType * cpuImageMask, const MaskImageType * gpuImageMask,
  TScalarType & rmsRelative )
{
  ImageRegionConstIterator< CPUImageType > cit(
    cpuImage, cpuImage->GetLargestPossibleRegion() );
  ImageRegionConstIterator< GPUImageType > git(
    gpuImage, gpuImage->GetLargestPossibleRegion() );

  ImageRegionConstIterator< MaskImageType > mcit(
    cpuImageMask, cpuImageMask->GetLargestPossibleRegion() );
  ImageRegionConstIterator< MaskImageType > mgit(
    gpuImageMask, gpuImageMask->GetLargestPossibleRegion() );

  TScalarType rmse          = 0.0;
  TScalarType sumCPUSquared = 0.0;
  std::size_t count         = 0;
  for( cit.GoToBegin(), git.GoToBegin(), mcit.GoToBegin(), mgit.GoToBegin();
    !cit.IsAtEnd(); ++cit, ++git, ++mcit, ++mgit )
  {
    if( ( mcit.Get() == NumericTraits< typename MaskImageType::PixelType >::OneValue() )
      && ( mgit.Get() == NumericTraits< typename MaskImageType::PixelType >::OneValue() ) )
    {
      TScalarType cpu = static_cast< TScalarType >( cit.Get() );
      TScalarType err = cpu - static_cast< TScalarType >( git.Get() );
      rmse          += err * err;
      sumCPUSquared += cpu * cpu;
      count++;
    }
  }

  if( count == 0 )
  {
    rmsRelative = 0.0;
    return 0.0;
  }

  rmse        = vcl_sqrt( rmse / count );
  rmsRelative = rmse / vcl_sqrt( sumCPUSquared / count );
  return rmse;
} // end ComputeRMSE()


//------------------------------------------------------------------------------
// Helper function to compute RMSE with masks and threshold
template< class TScalarType, class CPUImageType, class GPUImageType, class MaskImageType >
TScalarType
ComputeRMSE2( const CPUImageType * cpuImage, const GPUImageType * gpuImage,
  const MaskImageType * cpuImageMask, const MaskImageType * gpuImageMask,
  const float threshold, TScalarType & rmsRelative )
{
  ImageRegionConstIterator< CPUImageType > cit(
    cpuImage, cpuImage->GetLargestPossibleRegion() );
  ImageRegionConstIterator< GPUImageType > git(
    gpuImage, gpuImage->GetLargestPossibleRegion() );

  ImageRegionConstIterator< MaskImageType > mcit(
    cpuImageMask, cpuImageMask->GetLargestPossibleRegion() );
  ImageRegionConstIterator< MaskImageType > mgit(
    gpuImageMask, gpuImageMask->GetLargestPossibleRegion() );

  TScalarType rmse          = 0.0;
  TScalarType sumCPUSquared = 0.0;
  std::size_t count         = 0;
  for( cit.GoToBegin(), git.GoToBegin(), mcit.GoToBegin(), mgit.GoToBegin();
    !cit.IsAtEnd(); ++cit, ++git, ++mcit, ++mgit )
  {
    if( mcit.Get() == NumericTraits< typename MaskImageType::PixelType >::One
      && mgit.Get() == NumericTraits< typename MaskImageType::PixelType >::OneValue() )
    {
      ++count;
      TScalarType cpu = static_cast< TScalarType >( cit.Get() );
      TScalarType err = cpu - static_cast< TScalarType >( git.Get() );
      if( vnl_math_abs( err ) > threshold )
      {
        rmse += err * err;
      }
      sumCPUSquared += cpu * cpu;
    }
  }

  if( count == 0 )
  {
    rmsRelative = 0.0;
    return 0.0;
  }

  rmse        = vcl_sqrt( rmse / count );
  rmsRelative = rmse / vcl_sqrt( sumCPUSquared / count );
  return rmse;
} // end ComputeRMSE()


//----------------------------------------------------------------------------
// Write log file in Microsoft Excel semicolon separated format.
template< class ImageType >
void
WriteLog(
  const std::string & filename,
  const unsigned int dim,
  const typename ImageType::SizeType & imagesize,
  const double rmsError,
  const double rmsRelative,
  const bool testPassed,
  const bool exceptionGPU,
  const unsigned int numThreads,
  const unsigned int runTimes,
  const std::string & filterName,
  const TimeProbe::TimeStampType cpuTime,
  const TimeProbe::TimeStampType gpuTime,
  const std::string & comments = "" )
{
  const std::string s( " ; " );   // separator
  std::ofstream     fout;
  const bool        fileExists = itksys::SystemTools::FileExists( filename.c_str() );

  fout.open( filename.c_str(), std::ios_base::app );
  // If file does not exist, then print table header
  if( !fileExists )
  {
    fout << "Filter Name" << s << "Dimension" << s << "Image Size"
         << s << "CPU(" << numThreads << ") (s)" << s << "GPU (s)" << s << "CPU/GPU Speed Ratio"
         << s << "RMS Error" << s << "RMS Relative" << s << "Test Passed" << s << "Run Times" << s << "Comments" << std::endl;
  }

  fout << filterName << s << dim << s;
  for( unsigned int i = 0; i < dim; i++ )
  {
    fout << imagesize.GetSize()[ i ];
    if( i < dim - 1 ) { fout << "x"; }
  }

  fout << s << cpuTime << s;

  if( !exceptionGPU )
  {
    fout << gpuTime << s;
  }
  else
  {
    fout << "na" << s;
  }

  if( !exceptionGPU )
  {
    if( gpuTime != 0.0 )
    {
      fout << ( cpuTime / gpuTime ) << s;
    }
    else
    {
      fout << "0" << s;
    }
  }
  else
  {
    fout << "na" << s;
  }

  fout << rmsError << s;
  fout << rmsRelative << s;
  fout << ( testPassed ? "Yes" : "No" ) << s;
  fout << runTimes << s;

  if( comments.size() > 0 )
  {
    fout << comments;
  }
  else
  {
    fout << "none";
  }

  fout << std::endl;

  fout.close();
  return;
}


}

#endif // end #ifndef __itkTestHelper_h
