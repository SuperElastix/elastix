/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#pragma warning(push)
// warning C4996: 'std::copy': Function call with parameters that may be unsafe
// - this call relies on the caller to check that the passed values are correct.
// To disable this warning, use -D_SCL_SECURE_NO_WARNINGS. See documentation on
// how to use Visual C++ 'Checked Iterators'
#pragma warning(disable:4996)
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#pragma warning(pop)

#include "itkTimeProbe.h"
#include "itkGPURecursiveGaussianImageFilter.h"

#include "itkOpenCLUtil.h" // IsGPUAvailable()

/**
* Testing GPU Recursive Gaussian Image Filter
*/
#define ImageDimension 3 // 2

// 3D:
// D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512x512x107-3D.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512x512x107-3D-out.mha
// 3D:
// D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-48x62x42-3D.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-48x62x42-3D-out.mha
int main(int argc, char *argv[])
{
  // register object factory for GPU image and filter
  //itk::ObjectFactoryBase::RegisterFactory( itk::GPUImageFactory::New() );
  //itk::ObjectFactoryBase::RegisterFactory( itk::GPURecursiveGaussianImageFilterFactory::New() );

  typedef float InputPixelType;
  typedef float OutputPixelType;

  typedef itk::GPUImage<InputPixelType,  ImageDimension>  InputImageType;
  typedef itk::GPUImage<OutputPixelType, ImageDimension>  OutputImageType;

  typedef itk::RecursiveGaussianImageFilter<InputImageType, OutputImageType> CPUFilterType;
  typedef itk::GPURecursiveGaussianImageFilter<InputImageType, OutputImageType> GPUFilterType;

  typedef itk::ImageFileReader<InputImageType>   ReaderType;
  typedef itk::ImageFileWriter<OutputImageType>  WriterType;

  if( argc <  3 )
  {
    std::cerr << "Error: missing arguments" << std::endl;
    std::cerr << "inputfile outputfile " << std::endl;
    return EXIT_FAILURE;
  }

  if(!itk::IsGPUAvailable())
  {
    std::cerr << "OpenCL-enabled GPU is not present." << std::endl;
    return EXIT_FAILURE;
  }

  const double sigma = 3.0;
  unsigned int direction = 0;
  const double epsilon = 0.01;
  const unsigned int maximumNumberOfThreads = itk::MultiThreader::GetGlobalDefaultNumberOfThreads();
  const bool testspeed      = true;
  const bool testdirections = true; // Works for 48x62x42 image, Does not work for 512x512x107 ! Why?
  const bool test1D         = true;

  // Reader
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( argv[1] );
  try
  {
    reader->Update();
  }
  catch( itk::ExceptionObject & itkNotUsed( excp ) )
  {
    //ML_PRINT_ERROR( "Execute()", ML_UNKNOWN_EXCEPTION, excp.GetDescription() );
    //return;
  }

  // Test 1~n threads for CPU
  // Speed CPU vs GPU
  if(testspeed)
  {
    std::cout << "Testing 1~"<< maximumNumberOfThreads <<" threads for CPU vs GPU:\n";

    for(unsigned int nThreads = 1; nThreads <= maximumNumberOfThreads; nThreads++)
    {
      // Test CPU
      CPUFilterType::Pointer CPUFilter = CPUFilterType::New();

      itk::TimeProbe cputimer;
      cputimer.Start();

      CPUFilter->SetNumberOfThreads( nThreads );

      CPUFilter->SetInput( reader->GetOutput() );
      CPUFilter->SetSigma( sigma );
      CPUFilter->SetDirection( direction );
      CPUFilter->Update();

      cputimer.Stop();
      std::cout << "CPU Recursive Gaussian filter took " << cputimer.GetMean() << " seconds with "
        << CPUFilter->GetNumberOfThreads() << " threads."
        << " For direction: "<< direction <<", sigma: "<< sigma << std::endl;

      // Test GPU
      if( nThreads == maximumNumberOfThreads )
      {
        GPUFilterType::Pointer GPUFilter = GPUFilterType::New();

        itk::TimeProbe gputimer;
        gputimer.Start();

        GPUFilter->SetInput( reader->GetOutput() );
        GPUFilter->SetSigma( sigma );
        GPUFilter->SetDirection( direction );
        GPUFilter->Update();

        // Commented out, not in the GPU-Alpha branch
        //GPUFilter->GetOutput()->UpdateBuffers(); // synchronization point (GPU->CPU memcpy)

        gputimer.Stop();
        std::cout << "GPU Recursive Gaussian filter took " << gputimer.GetMean() << " seconds."
          << " For direction: "<< direction <<", sigma: "<< sigma << std::endl;

        // RMS Error check
        double diff = 0.0;
        unsigned int nPix = 0;
        itk::ImageRegionIterator<OutputImageType> cit(CPUFilter->GetOutput(), CPUFilter->GetOutput()->GetLargestPossibleRegion());
        itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(), GPUFilter->GetOutput()->GetLargestPossibleRegion());

        for(cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git)
        {
          double err = vnl_math_abs((double)(cit.Get()) - (double)(git.Get()));
          if(err > epsilon)
            std::cout << "CPU : " << (double)(cit.Get()) << ", GPU : " << (double)(git.Get()) << std::endl;
          diff += err*err;
          nPix++;
        }

        std::cout << "RMS Error: " << sqrt( diff / (double)nPix ) << std::endl;

        // Write output image
        //WriterType::Pointer writer = WriterType::New();
        //writer->SetInput(GPUFilter->GetOutput());
        //writer->SetFileName( argv[2] );
        //writer->Update();
      }
    }
  }

  // Works for 48x62x42 image, Does not work for 512x512x107 ! Why?
  if(testdirections)
  {
    std::cout << "Testing directions switch CPU vs GPU:\n";

    // Check directions
    for(direction=0; direction<ImageDimension; direction++)
    {
      // Test CPU
      CPUFilterType::Pointer CPUFilter = CPUFilterType::New();

      itk::TimeProbe cputimer;
      cputimer.Start();

      CPUFilter->SetNumberOfThreads( maximumNumberOfThreads );
      CPUFilter->SetInput( reader->GetOutput() );
      CPUFilter->SetSigma( sigma );
      CPUFilter->SetDirection( direction );
      CPUFilter->Modified();
      CPUFilter->Update();

      cputimer.Stop();
      std::cout << "CPU Recursive Gaussian filter took " << cputimer.GetMean() << " seconds with "
        << CPUFilter->GetNumberOfThreads() << " threads."
        << " For direction: "<< direction <<", sigma: "<< sigma << std::endl;

      // Test GPU
      GPUFilterType::Pointer GPUFilter = GPUFilterType::New();

      itk::TimeProbe gputimer;
      gputimer.Start();

      GPUFilter->SetInput( reader->GetOutput() );
      GPUFilter->SetSigma( sigma );
      GPUFilter->SetDirection( direction );
      GPUFilter->Modified();
      GPUFilter->Update();

      // Commented out, not in the GPU-Alpha branch
      //GPUFilter->GetOutput()->UpdateBuffers(); // synchronization point (GPU->CPU memcpy)

      gputimer.Stop();
      std::cout << "GPU Recursive Gaussian filter took " << gputimer.GetMean() << " seconds."
        << " For direction: "<< direction <<", sigma: "<< sigma << std::endl;

      // RMS Error check
      double diff = 0.0;
      unsigned int nPix = 0;
      itk::ImageRegionIterator<OutputImageType> cit(CPUFilter->GetOutput(), CPUFilter->GetOutput()->GetLargestPossibleRegion());
      itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(), GPUFilter->GetOutput()->GetLargestPossibleRegion());

      for(cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git)
      {
        double err = vnl_math_abs((double)(cit.Get()) - (double)(git.Get()));
        if(err > epsilon)
          std::cout << "CPU : " << (double)(cit.Get()) << ", GPU : " << (double)(git.Get()) << std::endl;
        diff += err*err;
        nPix++;
      }

      std::cout << "RMS Error for direction" << direction << " : " << sqrt( diff / (double)nPix ) << std::endl;
    }
  }

  // Test 1D
  if(test1D)
  {
    std::cout << "Testing 1D CPU vs GPU:\n";

    typedef itk::Image<InputPixelType, 1> ImageType1D;
    typedef ImageType1D::SizeType         SizeType;
    typedef ImageType1D::IndexType        IndexType;
    typedef ImageType1D::RegionType       RegionType;
    typedef ImageType1D::SpacingType      SpacingType;

    typedef itk::NumericTraits<InputPixelType>::RealType PixelRealType;

    SizeType size;
    size[0] = 513;

    IndexType start;
    start[0] = 0;

    RegionType region;
    region.SetIndex( start );
    region.SetSize( size );

    SpacingType spacing;
    spacing[0] = 1.0;

    // CPU Image
    ImageType1D::Pointer CPUInputImage = ImageType1D::New();
    CPUInputImage->SetRegions( region );
    CPUInputImage->Allocate();
    CPUInputImage->SetSpacing( spacing );
    CPUInputImage->FillBuffer( itk::NumericTraits<InputPixelType>::Zero );

    IndexType index0;
    index0[0] = ( size[0] - 1 ) / 2; // the middle pixel
    CPUInputImage->SetPixel( index0, static_cast<InputPixelType>( 1000.0 ) );

    // CPU filter
    typedef itk::RecursiveGaussianImageFilter<ImageType1D, ImageType1D> CPUFilterType1D;
    CPUFilterType1D::Pointer CPUFilter = CPUFilterType1D::New();
    CPUFilter->SetInput( CPUInputImage );
    CPUFilter->SetNormalizeAcrossScale( true );
    CPUFilter->SetSigma( sigma );
    CPUFilter->Update();

    // GPU Image
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUImageFactory::New() );

    ImageType1D::Pointer GPUInputImage = ImageType1D::New();
    GPUInputImage->SetRegions( region );
    GPUInputImage->Allocate();
    GPUInputImage->SetSpacing( spacing );
    GPUInputImage->FillBuffer( itk::NumericTraits<InputPixelType>::Zero );

    IndexType index1;
    index1[0] = ( size[0] - 1 ) / 2;  // the middle pixel
    GPUInputImage->SetPixel( index1, static_cast<InputPixelType>( 1000.0 ) );

    // Check input images
    double diff = 0.0;
    unsigned int nPix = 0;
    itk::ImageRegionIterator<ImageType1D> cinIt(CPUInputImage, CPUInputImage->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType1D> ginIt(GPUInputImage, GPUInputImage->GetLargestPossibleRegion());

    std::cout <<"Checking input images...";

    for(cinIt.GoToBegin(), ginIt.GoToBegin(); !cinIt.IsAtEnd(); ++cinIt, ++ginIt)
    {
      double err = vnl_math_abs((double)(cinIt.Get()) - (double)(ginIt.Get()));
      if(err > epsilon)
      {
        const IndexType cindex = cinIt.GetIndex();
        std::cout <<"Index : "<< cindex << " CPU : " << (double)(cinIt.Get()) << ", GPU : " << (double)(ginIt.Get()) << std::endl;
      }
      diff += err*err;
      nPix++;
    }
    if(diff == 0)
      std::cout <<" Ok." << std::endl;
    else
      std::cout <<" Fail." << std::endl;

    // GPU filter
    typedef itk::GPURecursiveGaussianImageFilter<ImageType1D, ImageType1D> GPUFilterType1D;
    GPUFilterType1D::Pointer GPUFilter = GPUFilterType1D::New();
    GPUFilter->SetInput( GPUInputImage );
    GPUFilter->SetNormalizeAcrossScale( true );
    GPUFilter->SetSigma( sigma );
    GPUFilter->Update();

    // RMS Error check
    diff = 0.0;
    nPix = 0;
    itk::ImageRegionIterator<ImageType1D> cit(CPUFilter->GetOutput(), CPUFilter->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType1D> git(GPUFilter->GetOutput(), GPUFilter->GetOutput()->GetLargestPossibleRegion());

    for(cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git)
    {
      double err = vnl_math_abs((double)(cit.Get()) - (double)(git.Get()));
      if(err > epsilon)
      {
        const IndexType cindex = cit.GetIndex();
        std::cout <<"Index : "<< cindex << " CPU : " << (double)(cit.Get()) << ", GPU : " << (double)(git.Get()) << std::endl;
      }
      diff += err*err;
      nPix++;
    }

    std::cout << "RMS Error for 1D test: " << sqrt( diff / (double)nPix ) << std::endl;
  }

  return EXIT_SUCCESS;
}
