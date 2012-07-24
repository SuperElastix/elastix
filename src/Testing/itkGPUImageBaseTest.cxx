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
/**
* Test program for GPUImageBase class.
*/
#include "itkGPUKernelManagerHelperFunctions.h"
#include <iomanip>

int main(int, char *[])
{
  const unsigned int dimension = 2;

  unsigned int width, height;
  width  = 256;
  height = 256;
  const unsigned int nElem = width*height;

  typedef itk::Image<float, dimension>    CPUImage2DType;
  typedef itk::GPUImage<float, dimension> GPUImage2DType;

  GPUImage2DType::SpacingType spacing;
  spacing[0] = 0.5;
  spacing[1] = 2.7;

  GPUImage2DType::PointType origin;
  origin[0] = 1.5;
  origin[1] = 5.8;

  GPUImage2DType::SizeType size;
  size[0] = width;
  size[1] = height;

  GPUImage2DType::DirectionType direction;
  direction.Fill(0.0);
  direction[0][1] = -1;
  direction[1][0] = 1;

  GPUImage2DType::IndexType start;
  start[0] = 0;
  start[1] = 0;

  GPUImage2DType::RegionType region;
  region.SetSize( size );
  region.SetIndex( start );

  // Create CPU 1D image
  CPUImage2DType::Pointer CPUImage2D = CPUImage2DType::New();
  CPUImage2D->SetRegions( region );
  CPUImage2D->SetSpacing( spacing );
  CPUImage2D->SetOrigin( origin );
  CPUImage2D->SetDirection( direction );
  CPUImage2D->Allocate();
  CPUImage2D->FillBuffer( 1.0f );

  // Create GPU 1D image
  GPUImage2DType::Pointer inGPUImage2D = GPUImage2DType::New();
  inGPUImage2D->SetRegions( region );
  inGPUImage2D->SetSpacing( spacing );
  inGPUImage2D->SetOrigin( origin );
  inGPUImage2D->SetDirection( direction );
  inGPUImage2D->Allocate();
  inGPUImage2D->FillBuffer( 1.0f );

  GPUImage2DType::Pointer outGPUImage2D = GPUImage2DType::New();
  outGPUImage2D->SetRegions( region );
  //inGPUImage2D->SetSpacing( spacing );
  //inGPUImage2D->SetDirection( direction );
  outGPUImage2D->Allocate();
  outGPUImage2D->FillBuffer( 1.0f );

  std::ostringstream defines;

  defines << "#define DIM_" << int(GPUImage2DType::ImageDimension) << std::endl;
  defines << "#define INPIXELTYPE float" << std::endl;
  defines << "#define OUTPIXELTYPE float" << std::endl;

  // Load GPUMath
  const std::string oclGPUMathPath = "D:\\work\\elastix-ext\\ITK4OpenCL\\common\\kernels\\GPUMath.cl";
  std::string oclGPUMathSource;
  if(!itk::LoadProgramFromFile(oclGPUMathPath, oclGPUMathSource))
  {
    itkGenericExceptionMacro( << "GPUMath has not been loaded from: " << oclGPUMathPath );
  }
  else
  {
    defines << oclGPUMathSource << std::endl;
  }

  // Load GPUImageBase
  const std::string oclGPUImageBasePath = "D:\\work\\elastix-ext\\ITK4OpenCL\\common\\kernels\\GPUImageBase.cl";
  std::string oclGPUImageBaseSource;
  if(!itk::LoadProgramFromFile(oclGPUImageBasePath, oclGPUImageBaseSource))
  {
    itkGenericExceptionMacro( << "GPUImageBase has not been loaded from: " << oclGPUImageBasePath );
  }
  else
  {
    defines << oclGPUImageBaseSource << std::endl;
  }

  // Create GPU program object
  itk::GPUKernelManager::Pointer kernelManager = itk::GPUKernelManager::New();

  // Load program and compile test kernel
  const std::string oclTestSourcePath = "D:\\work\\elastix-ext\\ITK4OpenCL\\common\\kernels\\GPUImageBaseTest.cl";
  kernelManager->LoadProgramFromFile( oclTestSourcePath.c_str(), defines.str().c_str() );

  // create addition kernel
  cl_uint argidx = 0;
  int kernel_add = kernelManager->CreateKernel("GPUImageBaseTest2D");
  itk::SetKernelWithITKImage<GPUImage2DType>(kernelManager, kernel_add, argidx, inGPUImage2D);
  kernelManager->SetKernelArgWithImage(kernel_add, argidx++, outGPUImage2D->GetGPUDataManager());
  kernelManager->SetKernelArg(kernel_add, argidx++, sizeof(unsigned int), &nElem);

  inGPUImage2D->SetCurrentCommandQueue(0);
  outGPUImage2D->SetCurrentCommandQueue(0);
  kernelManager->SetCurrentCommandQueue(0);

  // check pixel value
  std::cout << "\n--- inGPUImage2D -------------------------------------------\n";
  inGPUImage2D->Print( std::cout );

  //std::cout << "Current Command Queue ID : 0 " << std::endl;
  //std::cout << "======================" << std::endl;
  //std::cout << "Kernel : SetImageBaseInfo2D" << std::endl;
  //std::cout << "------------------" << std::endl;
  //std::cout << "Before GPU kernel execution" << std::endl;
  //std::cout << "inGPUImage2D : " << inGPUImage2D->GetSpacing() << std::endl;
  //std::cout << "outGPUImage2D : " << outGPUImage2D->GetSpacing() << std::endl;

  kernelManager->LaunchKernel2D(kernel_add, width, height, 16, 16);

  // Check Spacing
  unsigned int index = 0;
  GPUImage2DType::IndexType idx;
  idx[0] = 0;
  for(unsigned int i=0; i<dimension; i++)
  {
    idx[1] = index;
    float spacingIn  = static_cast<float>(inGPUImage2D->GetSpacing()[i]);
    float spacingOut = static_cast<float>(outGPUImage2D->GetPixel(idx));
    if(spacingIn != spacingOut)
    {
      std::cout << "ERROR: Image spacing " << " " << spacingIn << " != " << spacingOut << std::endl;
      return EXIT_FAILURE;
    }
    index++;
  }

  // Check Origin
  for(unsigned int i=0; i<dimension; i++)
  {
    idx[1] = index;
    float originIn  = static_cast<float>(inGPUImage2D->GetOrigin()[i]);
    float originOut = static_cast<float>(outGPUImage2D->GetPixel(idx));
    if(originIn != originOut)
    {
      std::cout << "ERROR: Image origin " << " " << originIn << " != " << originOut << std::endl;
      return EXIT_FAILURE;
    }
    index++;
  }

  // Check Size
  for(unsigned int i=0; i<dimension; i++)
  {
    idx[1] = index;
    float sizeIn = static_cast<float>(inGPUImage2D->GetLargestPossibleRegion().GetSize()[i]);
    float sizeOut = static_cast<float>(outGPUImage2D->GetPixel(idx));
    if(sizeIn != sizeOut)
    {
      std::cout << "ERROR: Image size " << " " << sizeIn << " != " << sizeOut << std::endl;
      return EXIT_FAILURE;
    }
    index++;
  }

  // Check Direction
  for(unsigned int i=0; i<dimension; i++)
  {
    for(unsigned int j=0; j<dimension; j++)
    {
      idx[1] = index;
      float directionIn = static_cast<float>(inGPUImage2D->GetDirection()[i][j]);
      float directionOut = static_cast<float>(outGPUImage2D->GetPixel(idx));
      if(directionIn != directionOut)
      {
        std::cout << "ERROR: Image direction " << " " << directionIn << " != " << directionOut << std::endl;
        return EXIT_FAILURE;
      }
      index++;
    }
  }

  // Check IndexToPhysicalPoint only for itkGPUImage
  for(unsigned int i=0; i<dimension; i++)
  {
    for(unsigned int j=0; j<dimension; j++)
    {
      idx[1] = index;
      float directionIn = static_cast<float>(inGPUImage2D->GetIndexToPhysicalPoint()[i][j]);
      float directionOut = static_cast<float>(outGPUImage2D->GetPixel(idx));
      if(directionIn != directionOut)
      {
        std::cout << "ERROR: Image IndexToPhysicalPoint " << " " << directionIn << " != " << directionOut << std::endl;
        return EXIT_FAILURE;
      }
      index++;
    }
  }

  // Check PhysicalPointToIndex only for itkGPUImage
  for(unsigned int i=0; i<dimension; i++)
  {
    for(unsigned int j=0; j<dimension; j++)
    {
      idx[1] = index;
      float directionIn = static_cast<float>(inGPUImage2D->GetPhysicalPointToIndex()[i][j]);
      float directionOut = static_cast<float>(outGPUImage2D->GetPixel(idx));
      if(directionIn != directionOut)
      {
        std::cout << "ERROR: Image IndexToPhysicalPoint " << " " << directionIn << " != " << directionOut << std::endl;
        return EXIT_FAILURE;
      }
      index++;
    }
  }

  // Check for image->TransformIndexToPhysicalPoint(index, point);
  typedef itk::Point<float, dimension> PointType;
  GPUImage2DType::IndexType index2d;
  index2d[0] = 10;
  index2d[1] = 25;
  PointType outputPoint;
  inGPUImage2D->TransformIndexToPhysicalPoint(index2d, outputPoint);

  // Check OpenCL implementation of transform_index_to_physical_point_2d()
  for(unsigned int i=0; i<dimension; i++)
  {
    idx[1] = index;
    float outputPointGPU = static_cast<float>(outGPUImage2D->GetPixel(idx));
    if(outputPoint[i] != outputPointGPU)
    {
      std::cout << "ERROR: physical point " << " " << outputPoint[i] << " != " << outputPointGPU << std::endl;
      return EXIT_FAILURE;
    }
    index++;
  }

  // Check for image->TransformPhysicalPointToContinuousIndex(point, continuousindex);
  const float RMSEpsilon = 0.00001;
  typedef itk::ContinuousIndex<float, dimension> ContinuousInputIndexType;
  PointType point2d_1;

  // Invalid case
  point2d_1[0] = 62.0;
  point2d_1[1] = 11.0;
  ContinuousInputIndexType outputContinuousInput1;
  const bool insideCPU1 = inGPUImage2D->TransformPhysicalPointToContinuousIndex(point2d_1, outputContinuousInput1);

  // Check OpenCL implementation of transform_physical_point_to_continuous_index_2d()
  for(unsigned int i=0; i<dimension; i++)
  {
    idx[1] = index;
    float outputPointGPU = static_cast<float>(outGPUImage2D->GetPixel(idx));
    double err = vnl_math_abs(outputPointGPU - outputContinuousInput1[i]);
    if(err > RMSEpsilon)
    {
      std::cout << "ERROR: continuous index point " << " "
        << std::setprecision(16) << outputContinuousInput1[i] << " != " << outputPointGPU << std::endl;
      return EXIT_FAILURE;
    }
    index++;
  }

  // Check for valid status
  idx[1] = index;
  const bool insideGPU1 = static_cast<bool>(outGPUImage2D->GetPixel(idx));
  if(insideCPU1 != insideGPU1)
  {
    std::cout << "ERROR: inside " << " " << insideCPU1 << " != " << insideGPU1 << std::endl;
    return EXIT_FAILURE;
  }

  index++;

  // Valid case
  PointType point2d_2;
  point2d_2[0] = -67.4;
  point2d_2[1] = 13.2;
  ContinuousInputIndexType outputContinuousInput2;
  const bool insideCPU2 = inGPUImage2D->TransformPhysicalPointToContinuousIndex(point2d_2, outputContinuousInput2);

  // Check OpenCL implementation of transform_physical_point_to_continuous_index_2d()
  for(unsigned int i=0; i<dimension; i++)
  {
    idx[1] = index;
    float outputPointGPU = static_cast<float>(outGPUImage2D->GetPixel(idx));
    double err = vnl_math_abs(outputPointGPU - outputContinuousInput2[i]);
    if(err > RMSEpsilon)
    {
      std::cout << "ERROR: continuous index point " << " "
        << std::setprecision(16) << outputContinuousInput2[i] << " != " << outputPointGPU << std::endl;
      return EXIT_FAILURE;
    }
    index++;
  }

  // Check for valid status
  idx[1] = index;
  const bool insideGPU2 = static_cast<bool>(outGPUImage2D->GetPixel(idx));
  if(insideCPU2 != insideGPU2)
  {
    std::cout << "ERROR: inside " << " " << insideCPU2 << " != " << insideGPU2 << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

  /*
  unsigned int width, height;

  width  = 256;
  height = 256;

  //
  // create GPUImage
  //

  // set size & region
  ItkImage1f::Pointer srcA, srcB, dest;

  ItkImage1f::IndexType start;
  start[0] = 0;
  start[1] = 0;
  ItkImage1f::SizeType size;
  size[0] = width;
  size[1] = height;
  ItkImage1f::RegionType region;
  region.SetSize( size );
  region.SetIndex( start );

  // create
  srcA = ItkImage1f::New();
  srcA->SetRegions( region );
  srcA->Allocate();
  srcA->FillBuffer( 1.0f );

  srcB = ItkImage1f::New();
  srcB->SetRegions( region );
  srcB->Allocate();
  srcB->FillBuffer( 3.0f );

  dest = ItkImage1f::New();
  dest->SetRegions( region );
  dest->Allocate();
  dest->FillBuffer( 0.0f );

  // check pixel value
  ItkImage1f::IndexType idx;
  idx[0] = 0;
  idx[1] = 0;


  unsigned int nElem = width*height;

  //
  // create GPU program object
  //
  itk::GPUKernelManager::Pointer kernelManager = itk::GPUKernelManager::New();

  // load program and compile
  //std::string oclSrcPath = itk_root_path;
  //oclSrcPath += "/Modules/GPU/Common/ImageOps.cl";
  std::string oclSrcPath = "D:\\work\\elastix-ext\\ITK4OpenCL\\common\\kernels\\ImageOps.cl";
  kernelManager->LoadProgramFromFile( oclSrcPath.c_str(), "#define PIXELTYPE float\n" );

  //
  // create addition kernel
  //
  int kernel_add = kernelManager->CreateKernel("ImageAdd");

  srcA->SetCurrentCommandQueue( 0 );
  srcB->SetCurrentCommandQueue( 0 );
  dest->SetCurrentCommandQueue( 0 );
  kernelManager->SetCurrentCommandQueue( 0 );

  std::cout << "Current Command Queue ID : 0 " << std::endl;

  std::cout << "======================" << std::endl;
  std::cout << "Kernel : Addition" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "Before GPU kernel execution" << std::endl;
  std::cout << "SrcA : " << srcA->GetPixel( idx ) << std::endl;
  std::cout << "SrcB : " << srcB->GetPixel( idx ) << std::endl;
  std::cout << "Dest : " << dest->GetPixel( idx ) << std::endl;

  kernelManager->SetKernelArgWithImage(kernel_add, 0, srcA->GetGPUDataManager());
  kernelManager->SetKernelArgWithImage(kernel_add, 1, srcB->GetGPUDataManager());
  kernelManager->SetKernelArgWithImage(kernel_add, 2, dest->GetGPUDataManager());
  kernelManager->SetKernelArg(kernel_add, 3, sizeof(unsigned int), &nElem);
  kernelManager->LaunchKernel2D(kernel_add, width, height, 16, 16);

  std::cout << "------------------" << std::endl;
  std::cout << "After GPU kernel execution" << std::endl;
  std::cout << "SrcA : " << srcA->GetPixel( idx ) << std::endl;
  std::cout << "SrcB : " << srcB->GetPixel( idx ) << std::endl;
  std::cout << "Des  : " << dest->GetPixel( idx ) << std::endl;
  std::cout << "======================" << std::endl;

  //
  // create multiplication kernel
  //
  int kernel_mult = kernelManager->CreateKernel("ImageMult");

  std::cout << "======================" << std::endl;
  std::cout << "Kernel : Multiplication" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "Before GPU kernel execution" << std::endl;
  std::cout << "SrcA : " << srcA->GetPixel( idx ) << std::endl;
  std::cout << "SrcB : " << srcB->GetPixel( idx ) << std::endl;
  std::cout << "Dest : " << dest->GetPixel( idx ) << std::endl;

  kernelManager->SetKernelArgWithImage(kernel_mult, 0, srcA->GetGPUDataManager());
  kernelManager->SetKernelArgWithImage(kernel_mult, 1, srcB->GetGPUDataManager());
  kernelManager->SetKernelArgWithImage(kernel_mult, 2, dest->GetGPUDataManager());
  kernelManager->SetKernelArg(kernel_mult, 3, sizeof(unsigned int), &nElem);
  kernelManager->LaunchKernel2D(kernel_mult, width, height, 16, 16);

  std::cout << "------------------" << std::endl;
  std::cout << "After GPU kernel execution" << std::endl;
  std::cout << "SrcA : " << srcA->GetPixel( idx ) << std::endl;
  std::cout << "SrcB : " << srcB->GetPixel( idx ) << std::endl;
  std::cout << "Des  : " << dest->GetPixel( idx ) << std::endl;
  std::cout << "======================" << std::endl;

  //
  // Change Command Queue
  //
  itk::GPUContextManager *contextManager = itk::GPUContextManager::GetInstance();
  if(contextManager->GetNumCommandQueue() < 2) return 1;

  std::cout << "Current Command Queue ID : 1 " << std::endl;

  //
  // create subtraction kernel
  //
  int kernel_sub = kernelManager->CreateKernel("ImageSub");

  srcA->FillBuffer( 2.0f );
  srcB->FillBuffer( 4.0f );
  dest->FillBuffer( 1.0f );

  // default queue id was 0
  srcA->SetCurrentCommandQueue( 1 );
  srcB->SetCurrentCommandQueue( 1 );
  dest->SetCurrentCommandQueue( 1 );
  kernelManager->SetCurrentCommandQueue( 1 );

  std::cout << "======================" << std::endl;
  std::cout << "Kernel : Subtraction" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "Before GPU kernel execution" << std::endl;
  std::cout << "SrcA : " << srcA->GetPixel( idx ) << std::endl;
  std::cout << "SrcB : " << srcB->GetPixel( idx ) << std::endl;
  std::cout << "Dest : " << dest->GetPixel( idx ) << std::endl;

  kernelManager->SetKernelArgWithImage(kernel_sub, 0, srcA->GetGPUDataManager());
  kernelManager->SetKernelArgWithImage(kernel_sub, 1, srcB->GetGPUDataManager());
  kernelManager->SetKernelArgWithImage(kernel_sub, 2, dest->GetGPUDataManager());
  kernelManager->SetKernelArg(kernel_sub, 3, sizeof(unsigned int), &nElem);
  kernelManager->LaunchKernel2D(kernel_sub, width, height, 16, 16);

  std::cout << "------------------" << std::endl;
  std::cout << "After GPU kernel execution" << std::endl;
  std::cout << "SrcA : " << srcA->GetPixel( idx ) << std::endl;
  std::cout << "SrcB : " << srcB->GetPixel( idx ) << std::endl;
  std::cout << "Des  : " << dest->GetPixel( idx ) << std::endl;
  std::cout << "======================" << std::endl;

  return EXIT_SUCCESS;
  */
}
