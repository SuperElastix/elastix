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
* Test program for GPUMath class.
*/
#include "itkGPUKernelManagerHelperFunctions.h"

#include <iomanip>
#include <math.h>
#include <string>
#include <iostream>
#include "itkIndex.h"

bool math_test_helper(std::string str, bool test)
{
  if (!test)
    {
    std::cout<<"test ("<<str<<") failed"<<std::endl;
    }
  return test;
}

int main( int, char *[] )
{
  const unsigned int dimension = 2;

  unsigned int width, height;
  width  = 256;
  height = 256;
  const unsigned int nElem = width*height;
  typedef itk::GPUImage<int, dimension> GPUImage2DType;

  GPUImage2DType::SizeType size;
  size[0] = width;
  size[1] = height;

  GPUImage2DType::IndexType start;
  start[0] = 0;
  start[1] = 0;

  GPUImage2DType::RegionType region;
  region.SetSize( size );
  region.SetIndex( start );

  // Create GPU 2D in/out image
  GPUImage2DType::Pointer inGPUImage2D = GPUImage2DType::New();
  inGPUImage2D->SetRegions( region );
  inGPUImage2D->Allocate();
  inGPUImage2D->FillBuffer( 0 );

  GPUImage2DType::Pointer outGPUImage2D = GPUImage2DType::New();
  outGPUImage2D->SetRegions( region );
  outGPUImage2D->Allocate();
  outGPUImage2D->FillBuffer( 0 );

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

  // Create GPU program object
  itk::GPUKernelManager::Pointer kernelManager = itk::GPUKernelManager::New();

  // Load program and compile test kernel
  const std::string oclTestSourcePath = "D:\\work\\elastix-ext\\ITK4OpenCL\\common\\kernels\\GPUMathRoundTest.cl";
  kernelManager->LoadProgramFromFile( oclTestSourcePath.c_str(), defines.str().c_str() );

  // create addition kernel
  cl_uint argidx = 0;
  int kernel_add = kernelManager->CreateKernel("MathRoundTest");
  itk::SetKernelWithITKImage<GPUImage2DType>(kernelManager, kernel_add, argidx, inGPUImage2D);
  kernelManager->SetKernelArgWithImage(kernel_add, argidx++, outGPUImage2D->GetGPUDataManager());
  kernelManager->SetKernelArg(kernel_add, argidx++, sizeof(unsigned int), &nElem);

  inGPUImage2D->SetCurrentCommandQueue(0);
  outGPUImage2D->SetCurrentCommandQueue(0);
  kernelManager->SetCurrentCommandQueue(0);

  // check pixel value
  std::cout << "\n--- inGPUImage2D -------------------------------------------\n";
  inGPUImage2D->Print( std::cout );

  kernelManager->LaunchKernel2D(kernel_add, width, height, 16, 16);

  bool ok = true;

  typedef itk::Index<3>::IndexValueType IndexValueType;

  ok &= math_test_helper("rnd(-8.4999)  == -8", itk::Math::Round<IndexValueType>(-8.4999)  == -8);
  ok &= math_test_helper("rnd(-8.4999f) == -8", itk::Math::Round<IndexValueType>(-8.4999f) == -8);
  ok &= math_test_helper("rnd(-8.50)    == -8", itk::Math::Round<IndexValueType>(-8.50)    == -8);
  ok &= math_test_helper("rnd(-8.50f)   == -8", itk::Math::Round<IndexValueType>(-8.50f)   == -8);
  ok &= math_test_helper("rnd(-8.5001)  == -9", itk::Math::Round<IndexValueType>(-8.5001)  == -9);
  ok &= math_test_helper("rnd(-8.5001f) == -9", itk::Math::Round<IndexValueType>(-8.5001f) == -9);
  ok &= math_test_helper("rnd(8.4999)   ==  8", itk::Math::Round<IndexValueType>(8.4999)   ==  8);
  ok &= math_test_helper("rnd(8.4999f)  ==  8", itk::Math::Round<IndexValueType>(8.4999f)  ==  8);
  ok &= math_test_helper("rnd(8.50)     ==  9", itk::Math::Round<IndexValueType>(8.50)     ==  9);
  ok &= math_test_helper("rnd(8.50f)    ==  9", itk::Math::Round<IndexValueType>(8.50f)    ==  9);
  ok &= math_test_helper("rnd(8.5001)   ==  9", itk::Math::Round<IndexValueType>(8.5001)   ==  9);
  ok &= math_test_helper("rnd(8.5001f)  ==  9", itk::Math::Round<IndexValueType>(8.5001f)  ==  9);

  ok &= math_test_helper("rnd(-9.4999)  == -9 ", itk::Math::Round<IndexValueType>(-9.4999)   == -9);
  ok &= math_test_helper("rnd(-9.4999f) == -9 ", itk::Math::Round<IndexValueType>(-9.4999f)  == -9);
  ok &= math_test_helper("rnd(-9.50)    == -9 ", itk::Math::Round<IndexValueType>(-9.50)     == -9);
  ok &= math_test_helper("rnd(-9.50f)   == -9 ", itk::Math::Round<IndexValueType>(-9.50f)    == -9);
  ok &= math_test_helper("rnd(-9.5001)  == -10", itk::Math::Round<IndexValueType>(-9.5001)   == -10);
  ok &= math_test_helper("rnd(-9.5001f) == -10", itk::Math::Round<IndexValueType>(-9.5001f)  == -10);
  ok &= math_test_helper("rnd(9.4999)   ==  9 ", itk::Math::Round<IndexValueType>(9.4999)    ==  9);
  ok &= math_test_helper("rnd(9.4999f)  ==  9 ", itk::Math::Round<IndexValueType>(9.4999f)   ==  9);
  ok &= math_test_helper("rnd(9.50)     ==  10", itk::Math::Round<IndexValueType>(9.50)      ==  10);
  ok &= math_test_helper("rnd(9.50f)    ==  10", itk::Math::Round<IndexValueType>(9.50f)     ==  10);
  ok &= math_test_helper("rnd(9.5001)   ==  10", itk::Math::Round<IndexValueType>(9.5001)    ==  10);
  ok &= math_test_helper("rnd(9.5001f)  ==  10", itk::Math::Round<IndexValueType>(9.5001f)   ==  10);

  ok &= math_test_helper("rnd_halfinttoeven(-8.4999)  == -8", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-8.4999) == -8);
  ok &= math_test_helper("rnd_halfinttoeven(-8.4999f) == -8", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-8.4999f)== -8);
  ok &= math_test_helper("rnd_halfinttoeven(-8.50)    == -8", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-8.50)   == -8);
  ok &= math_test_helper("rnd_halfinttoeven(-8.50f)   == -8", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-8.50f)  == -8);
  ok &= math_test_helper("rnd_halfinttoeven(-8.5001)  == -9", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-8.5001) == -9);
  ok &= math_test_helper("rnd_halfinttoeven(-8.5001f) == -9", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-8.5001f)== -9);
  ok &= math_test_helper("rnd_halfinttoeven(8.4999)   ==  8", itk::Math::RoundHalfIntegerToEven<IndexValueType>(8.4999)  ==  8);
  ok &= math_test_helper("rnd_halfinttoeven(8.4999f)  ==  8", itk::Math::RoundHalfIntegerToEven<IndexValueType>(8.4999f) ==  8);
  ok &= math_test_helper("rnd_halfinttoeven(8.50)     ==  8", itk::Math::RoundHalfIntegerToEven<IndexValueType>(8.50)    ==  8);
  ok &= math_test_helper("rnd_halfinttoeven(8.50f)    ==  8", itk::Math::RoundHalfIntegerToEven<IndexValueType>(8.50f)   ==  8);
  ok &= math_test_helper("rnd_halfinttoeven(8.5001)   ==  9", itk::Math::RoundHalfIntegerToEven<IndexValueType>(8.5001)  ==  9);
  ok &= math_test_helper("rnd_halfinttoeven(8.5001f)  ==  9", itk::Math::RoundHalfIntegerToEven<IndexValueType>(8.5001f) ==  9);

  ok &= math_test_helper("rnd_halfinttoeven(-9.4999)  == -9 ", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-9.4999) == -9);
  ok &= math_test_helper("rnd_halfinttoeven(-9.4999f) == -9 ", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-9.4999f)== -9);
  ok &= math_test_helper("rnd_halfinttoeven(-9.50)    == -10", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-9.50)   == -10);
  ok &= math_test_helper("rnd_halfinttoeven(-9.50f)   == -10", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-9.50f)  == -10);
  ok &= math_test_helper("rnd_halfinttoeven(-9.5001)  == -10", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-9.5001) == -10);
  ok &= math_test_helper("rnd_halfinttoeven(-9.5001f) == -10", itk::Math::RoundHalfIntegerToEven<IndexValueType>(-9.5001f)== -10);
  ok &= math_test_helper("rnd_halfinttoeven(9.4999)   ==  9 ", itk::Math::RoundHalfIntegerToEven<IndexValueType>(9.4999)  ==  9);
  ok &= math_test_helper("rnd_halfinttoeven(9.4999f)  ==  9 ", itk::Math::RoundHalfIntegerToEven<IndexValueType>(9.4999f) ==  9);
  ok &= math_test_helper("rnd_halfinttoeven(9.50)     ==  10", itk::Math::RoundHalfIntegerToEven<IndexValueType>(9.50)    ==  10);
  ok &= math_test_helper("rnd_halfinttoeven(9.50f)    ==  10", itk::Math::RoundHalfIntegerToEven<IndexValueType>(9.50f)   ==  10);
  ok &= math_test_helper("rnd_halfinttoeven(9.5001)   ==  10", itk::Math::RoundHalfIntegerToEven<IndexValueType>(9.5001)  ==  10);
  ok &= math_test_helper("rnd_halfinttoeven(9.5001f)  ==  10", itk::Math::RoundHalfIntegerToEven<IndexValueType>(9.5001f) ==  10);

  ok &= math_test_helper("rnd_halfintup(-8.4999)  == -8", itk::Math::RoundHalfIntegerUp<IndexValueType>(-8.4999) == -8);
  ok &= math_test_helper("rnd_halfintup(-8.4999f) == -8", itk::Math::RoundHalfIntegerUp<IndexValueType>(-8.4999f)== -8);
  ok &= math_test_helper("rnd_halfintup(-8.50)    == -8", itk::Math::RoundHalfIntegerUp<IndexValueType>(-8.50)   == -8);
  ok &= math_test_helper("rnd_halfintup(-8.50f)   == -8", itk::Math::RoundHalfIntegerUp<IndexValueType>(-8.50f)  == -8);
  ok &= math_test_helper("rnd_halfintup(-8.5001)  == -9", itk::Math::RoundHalfIntegerUp<IndexValueType>(-8.5001) == -9);
  ok &= math_test_helper("rnd_halfintup(-8.5001f) == -9", itk::Math::RoundHalfIntegerUp<IndexValueType>(-8.5001f)== -9);
  ok &= math_test_helper("rnd_halfintup(8.4999)   ==  8", itk::Math::RoundHalfIntegerUp<IndexValueType>(8.4999)  ==  8);
  ok &= math_test_helper("rnd_halfintup(8.4999f)  ==  8", itk::Math::RoundHalfIntegerUp<IndexValueType>(8.4999f) ==  8);
  ok &= math_test_helper("rnd_halfintup(8.50)     ==  9", itk::Math::RoundHalfIntegerUp<IndexValueType>(8.50)    ==  9);
  ok &= math_test_helper("rnd_halfintup(8.50f)    ==  9", itk::Math::RoundHalfIntegerUp<IndexValueType>(8.50f)   ==  9);
  ok &= math_test_helper("rnd_halfintup(8.5001)   ==  9", itk::Math::RoundHalfIntegerUp<IndexValueType>(8.5001)  ==  9);
  ok &= math_test_helper("rnd_halfintup(8.5001f)  ==  9", itk::Math::RoundHalfIntegerUp<IndexValueType>(8.5001f) ==  9);

  ok &= math_test_helper("rnd_halfintup(-9.4999)  == -9 ", itk::Math::RoundHalfIntegerUp<IndexValueType>(-9.4999) == -9);
  ok &= math_test_helper("rnd_halfintup(-9.4999f) == -9 ", itk::Math::RoundHalfIntegerUp<IndexValueType>(-9.4999f)== -9);
  ok &= math_test_helper("rnd_halfintup(-9.50)    == -9 ", itk::Math::RoundHalfIntegerUp<IndexValueType>(-9.50)   == -9);
  ok &= math_test_helper("rnd_halfintup(-9.50f)   == -9 ", itk::Math::RoundHalfIntegerUp<IndexValueType>(-9.50f)  == -9);
  ok &= math_test_helper("rnd_halfintup(-9.5001)  == -10", itk::Math::RoundHalfIntegerUp<IndexValueType>(-9.5001) == -10);
  ok &= math_test_helper("rnd_halfintup(-9.5001f) == -10", itk::Math::RoundHalfIntegerUp<IndexValueType>(-9.5001f)== -10);
  ok &= math_test_helper("rnd_halfintup(9.4999)   ==  9 ", itk::Math::RoundHalfIntegerUp<IndexValueType>(9.4999)  ==  9);
  ok &= math_test_helper("rnd_halfintup(9.4999f)  ==  9 ", itk::Math::RoundHalfIntegerUp<IndexValueType>(9.4999f) ==  9);
  ok &= math_test_helper("rnd_halfintup(9.50)     ==  10", itk::Math::RoundHalfIntegerUp<IndexValueType>(9.50)    ==  10);
  ok &= math_test_helper("rnd_halfintup(9.50f)    ==  10", itk::Math::RoundHalfIntegerUp<IndexValueType>(9.50f)   ==  10);
  ok &= math_test_helper("rnd_halfintup(9.5001)   ==  10", itk::Math::RoundHalfIntegerUp<IndexValueType>(9.5001)  ==  10);
  ok &= math_test_helper("rnd_halfintup(9.5001f)  ==  10", itk::Math::RoundHalfIntegerUp<IndexValueType>(9.5001f) ==  10);

  ok &= math_test_helper("floor(8.0)      ==  8", itk::Math::Floor<IndexValueType>(8.0)      ==  8);
  ok &= math_test_helper("floor(8.0f)     ==  8", itk::Math::Floor<IndexValueType>(8.0f)     ==  8);
  ok &= math_test_helper("floor(8.9999)   ==  8", itk::Math::Floor<IndexValueType>(8.9999)   ==  8);
  ok &= math_test_helper("floor(8.9999f)  ==  8", itk::Math::Floor<IndexValueType>(8.9999f)  ==  8);
  ok &= math_test_helper("floor(8.0001)   ==  8", itk::Math::Floor<IndexValueType>(8.0001)   ==  8);
  ok &= math_test_helper("floor(8.0001f)  ==  8", itk::Math::Floor<IndexValueType>(8.0001f)  ==  8);
  ok &= math_test_helper("floor(-8.0)     == -8", itk::Math::Floor<IndexValueType>(-8.0)     == -8);
  ok &= math_test_helper("floor(-8.0f)    == -8", itk::Math::Floor<IndexValueType>(-8.0f)    == -8);
  ok &= math_test_helper("floor(-8.9999)  == -9", itk::Math::Floor<IndexValueType>(-8.9999)  == -9);
  ok &= math_test_helper("floor(-8.9999f) == -9", itk::Math::Floor<IndexValueType>(-8.9999f) == -9);
  ok &= math_test_helper("floor(-8.0001)  == -9", itk::Math::Floor<IndexValueType>(-8.0001)  == -9);
  ok &= math_test_helper("floor(-8.0001f) == -9", itk::Math::Floor<IndexValueType>(-8.0001f) == -9);

  ok &= math_test_helper("floor(9.0)      ==  9 ", itk::Math::Floor<IndexValueType>(9.0)      ==  9);
  ok &= math_test_helper("floor(9.0f)     ==  9 ", itk::Math::Floor<IndexValueType>(9.0f)     ==  9);
  ok &= math_test_helper("floor(9.9999)   ==  9 ", itk::Math::Floor<IndexValueType>(9.9999)   ==  9);
  ok &= math_test_helper("floor(9.9999f)  ==  9 ", itk::Math::Floor<IndexValueType>(9.9999f)  ==  9);
  ok &= math_test_helper("floor(9.0001)   ==  9 ", itk::Math::Floor<IndexValueType>(9.0001)   ==  9);
  ok &= math_test_helper("floor(9.0001f)  ==  9 ", itk::Math::Floor<IndexValueType>(9.0001f)  ==  9);
  ok &= math_test_helper("floor(-9.0)     == -9 ", itk::Math::Floor<IndexValueType>(-9.0)     == -9);
  ok &= math_test_helper("floor(-9.0f)    == -9 ", itk::Math::Floor<IndexValueType>(-9.0f)    == -9);
  ok &= math_test_helper("floor(-9.9999)  == -10", itk::Math::Floor<IndexValueType>(-9.9999)  == -10);
  ok &= math_test_helper("floor(-9.9999f) == -10", itk::Math::Floor<IndexValueType>(-9.9999f) == -10);
  ok &= math_test_helper("floor(-9.0001)  == -10", itk::Math::Floor<IndexValueType>(-9.0001)  == -10);
  ok &= math_test_helper("floor(-9.0001f) == -10", itk::Math::Floor<IndexValueType>(-9.0001f) == -10);

  ok &= math_test_helper("ceil(8.0)      ==  8", itk::Math::Ceil<IndexValueType>(8.0)      ==  8);
  ok &= math_test_helper("ceil(8.0f)     ==  8", itk::Math::Ceil<IndexValueType>(8.0f)     ==  8);
  ok &= math_test_helper("ceil(8.9999)   ==  9", itk::Math::Ceil<IndexValueType>(8.9999)   ==  9);
  ok &= math_test_helper("ceil(8.9999f)  ==  9", itk::Math::Ceil<IndexValueType>(8.9999f)  ==  9);
  ok &= math_test_helper("ceil(8.0001)   ==  9", itk::Math::Ceil<IndexValueType>(8.0001)   ==  9);
  ok &= math_test_helper("ceil(8.0001f)  ==  9", itk::Math::Ceil<IndexValueType>(8.0001f)  ==  9);
  ok &= math_test_helper("ceil(-8.0)     == -8", itk::Math::Ceil<IndexValueType>(-8.0)     == -8);
  ok &= math_test_helper("ceil(-8.0f)    == -8", itk::Math::Ceil<IndexValueType>(-8.0f)    == -8);
  ok &= math_test_helper("ceil(-8.9999)  == -8", itk::Math::Ceil<IndexValueType>(-8.9999)  == -8);
  ok &= math_test_helper("ceil(-8.9999f) == -8", itk::Math::Ceil<IndexValueType>(-8.9999f) == -8);
  ok &= math_test_helper("ceil(-8.0001)  == -8", itk::Math::Ceil<IndexValueType>(-8.0001)  == -8);
  ok &= math_test_helper("ceil(-8.0001f) == -8", itk::Math::Ceil<IndexValueType>(-8.0001f) == -8);

  ok &= math_test_helper("ceil(9.0)      ==  9 ", itk::Math::Ceil<IndexValueType>(9.0)      ==  9);
  ok &= math_test_helper("ceil(9.0f)     ==  9 ", itk::Math::Ceil<IndexValueType>(9.0f)     ==  9);
  ok &= math_test_helper("ceil(9.9999)   ==  10", itk::Math::Ceil<IndexValueType>(9.9999)   ==  10);
  ok &= math_test_helper("ceil(9.9999f)  ==  10", itk::Math::Ceil<IndexValueType>(9.9999f)  ==  10);
  ok &= math_test_helper("ceil(9.0001)   ==  10", itk::Math::Ceil<IndexValueType>(9.0001)   ==  10);
  ok &= math_test_helper("ceil(9.0001f)  ==  10", itk::Math::Ceil<IndexValueType>(9.0001f)  ==  10);
  ok &= math_test_helper("ceil(-9.0)     == -9 ", itk::Math::Ceil<IndexValueType>(-9.0)     == -9);
  ok &= math_test_helper("ceil(-9.0f)    == -9 ", itk::Math::Ceil<IndexValueType>(-9.0f)    == -9);
  ok &= math_test_helper("ceil(-9.9999)  == -9 ", itk::Math::Ceil<IndexValueType>(-9.9999)  == -9);
  ok &= math_test_helper("ceil(-9.9999f) == -9 ", itk::Math::Ceil<IndexValueType>(-9.9999f) == -9);
  ok &= math_test_helper("ceil(-9.0001)  == -9 ", itk::Math::Ceil<IndexValueType>(-9.0001)  == -9);
  ok &= math_test_helper("ceil(-9.0001f) == -9 ", itk::Math::Ceil<IndexValueType>(-9.0001f) == -9);

  if (!ok)
    {
    return EXIT_FAILURE;
    }
  else
    {
    std::cout<<"Test passed"<<std::endl;
    return EXIT_SUCCESS;
    }
}
