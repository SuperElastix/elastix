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
#include "itkGPUKernelManager.h"
#include <iostream>
#include <fstream>
#include "itkTimeProbe.h"

namespace itk
{

GPUKernelManager::GPUKernelManager()
{
  m_Manager = GPUContextManager::GetInstance();

  if(m_Manager->GetNumberOfCommandQueues() > 0) m_CommandQueueId = 0;   // default
                                                                  // command
                                                                  // queue
}

bool GPUKernelManager::LoadProgramFromFile(const char* filename, const char* preamble)
{
  if(filename == NULL)
  {
    itkWarningMacro( << "The filename must be specified.");
    return false;
  }
  
  // open the file
  std::ifstream inputFile( filename, std::ifstream::in | std::ifstream::binary );
  if ( inputFile.is_open() == false )
  {
    itkWarningMacro( << "Cannot open OpenCL source file: " << filename );
    return false;
  }

  std::stringstream sstream;
  if(preamble != "")
    sstream << preamble << std::endl << inputFile.rdbuf();
  else
    sstream << inputFile.rdbuf();

  inputFile.close();

  const std::string oclSource = sstream.str();
  const std::size_t oclSourceSize = oclSource.size();

  if(oclSourceSize == 0)
  {
    itkWarningMacro( << "Cannot build OpenCL source file: " << filename << " is empty." );
    return false;
  }

  std::string fileName(filename);
  bool created = false;

  // Intel OpenCL Debugging is enabled only when the target device is a CPU. 
  // If you target your code to run on Intel Processor Graphics, you can debug 
  // it on the CPU device during development phase and when ready to change the target device.
#if defined( ITK_USE_INTEL_CPU_OPENCL ) && defined( _DEBUG )
  // To work with the Intel SDK for OpenCL* - Debugger plug-in, the OpenCL* 
  // kernel code must exist in a text file separate from the code of the host. 
  if(preamble != "")
  {
    fileName = fileName.substr(0, fileName.rfind( "." )) + "-Debug.cl";
    std::ofstream debugfile( fileName.c_str() );
    if ( debugfile.is_open() == false )
    {
      itkWarningMacro(<< "Cannot create OpenCL debug source file: " << fileName );
      return false;
    }
    debugfile << oclSource;
    debugfile.close();

    itkWarningMacro(<< "For Debugging your OpenCL kernel use :" << fileName << " , not original .cl file.");
  }
#endif

#ifdef _DEBUG
  std::cout << "Creating OpenCL program from : " << fileName << std::endl;
#endif

  // Create 
  created = CreateOpenCLProgram(fileName, oclSource, oclSourceSize);
  return created;
}

bool GPUKernelManager::LoadProgramFromString(const char* source, const char* preamble)
{
  if(source == NULL)
  {
    itkWarningMacro( << "The source must be specified.");
    return false;
  }

  std::stringstream sstream;
  if(preamble != "")
    sstream << preamble << std::endl << source;
  else
    sstream << source;

  const std::string oclSource = sstream.str();
  const std::size_t oclSourceSize = oclSource.size();

  if(oclSourceSize == 0)
  {
    itkWarningMacro( << "Cannot build empty OpenCL source." );
    return false;
  }

#ifdef _DEBUG
  std::cout << "Creating OpenCL program from source." << std::endl;
#endif

  // Create 
  const bool created = CreateOpenCLProgram(std::string(), oclSource, oclSourceSize);
  return created;
}

bool GPUKernelManager::CreateOpenCLProgram(const std::string &filename,
  const std::string &source, const std::size_t sourceSize)
{
#ifdef OPENCL_PROFILING
  itk::TimeProbe buildtimer;
  buildtimer.Start();
#endif

  cl_int errid;
#ifdef _DEBUG
  std::cout<<"clCreateProgramWithSource" << "..." << std::endl;
#endif
  const char *code = source.c_str();
  m_Program = clCreateProgramWithSource(m_Manager->GetCurrentContext(), 1, 
    &code, &sourceSize, &errid);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  if(errid != CL_SUCCESS)
  {
    itkWarningMacro("Cannot create GPU program");
    return false;
  }

  // Get OpenCL math and optimization options
  std::string oclMathAndOptimization;
  const bool oclMathAndOptimizationEnabled = 
    GetOpenCLMathAndOptimizationOptions(oclMathAndOptimization);

#ifdef _DEBUG
  if(filename.size() > 0)
    std::cout<<"clBuildProgram '" << filename << "' ..." << std::endl;
  else
    std::cout<<"clBuildProgram from source ..." << std::endl;
#endif

#if defined( ITK_USE_INTEL_CPU_OPENCL ) && defined( _DEBUG )
  // Enable debugging mode in the Intel OpenCL runtime
  if(filename.size() > 0)
  {
    std::string oclDebugOptions = "-g -s \"" + filename + "\"";
    if(oclMathAndOptimizationEnabled)
    {
      oclDebugOptions = oclDebugOptions + " " + oclMathAndOptimization;
      errid = clBuildProgram(m_Program, 1,
        &m_Manager->GetDevices()[m_Manager->GetTargetDeviceId()],
        oclDebugOptions.c_str(), NULL, NULL);
    }
    else
    {
      errid = clBuildProgram(m_Program, 1,
        &m_Manager->GetDevices()[m_Manager->GetTargetDeviceId()],
        oclDebugOptions.c_str(), NULL, NULL);
    }
  }
#else
  if(oclMathAndOptimizationEnabled)
  {
    errid = clBuildProgram(m_Program, 1,
      &m_Manager->GetDevices()[m_Manager->GetTargetDeviceId()],
      oclMathAndOptimization.c_str(), NULL, NULL);
  }
  else
  {
    errid = clBuildProgram(m_Program, 1,
      &m_Manager->GetDevices()[m_Manager->GetTargetDeviceId()],
      NULL, NULL, NULL);
  }
#endif

  if(errid != CL_SUCCESS)
  {
    //itkWarningMacro("OpenCL program build error");

    // print out build error
    size_t paramValueSize = 0;

    // get error message size
    clGetProgramBuildInfo(m_Program, m_Manager->GetDeviceId(0), CL_PROGRAM_BUILD_LOG, 0, NULL, &paramValueSize);

    char *paramValue;
    paramValue = (char*)malloc(paramValueSize);

    // get error message
    clGetProgramBuildInfo(m_Program, m_Manager->GetDeviceId(0), CL_PROGRAM_BUILD_LOG, paramValueSize, paramValue, NULL);

    /*
    std::ostringstream itkmsg;
    itkmsg << "ERROR: In " __FILE__ ", line " << __LINE__ << "\n"
    << this->GetNameOfClass() << " (" << this << "): "
    << "OpenCL program build error:" << paramValue
    << "\n\n";
    ::itk::OutputWindowDisplayErrorText( itkmsg.str().c_str() );
    */

    std::cerr << paramValue << std::endl;

    free( paramValue );

    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

    return false;
  }

#ifdef OPENCL_PROFILING
  buildtimer.Stop();
  std::cout << "GPU Build program took " << buildtimer.GetMeanTime() << " seconds." << std::endl;
#endif

  return true;
}

int GPUKernelManager::CreateKernel(const char* kernelName)
{
  cl_int errid;

  // create kernel
#ifdef _DEBUG
  std::cout<<"clCreateKernel" << "..." << std::endl;
#endif
  cl_kernel newKernel = clCreateKernel(m_Program, kernelName, &errid);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  if(errid != CL_SUCCESS) // failed
  {
    itkWarningMacro("Fail to create GPU kernel");
    return -1;
  }

  m_KernelContainer.push_back( newKernel );

  // argument list
  m_KernelArgumentReady.push_back( std::vector< KernelArgumentList >() );
  cl_uint nArg;

#ifdef _DEBUG
  std::cout<<"clGetKernelInfo" << "..." << std::endl;
#endif

  errid = clGetKernelInfo( newKernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &nArg, NULL);
  (m_KernelArgumentReady.back()).resize( nArg );

  ResetArguments( (int)m_KernelContainer.size()-1 );

  return (int)m_KernelContainer.size()-1;
}

cl_int GPUKernelManager::GetKernelWorkGroupInfo(int kernelIdx,
  cl_kernel_work_group_info paramName, void *value)
{
  size_t valueSize, valueSizeRet;

  switch (paramName)
  {
  case CL_KERNEL_WORK_GROUP_SIZE:
    valueSize = sizeof(size_t);
    break;
  case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
    valueSize = 3 * sizeof(size_t);
    break;
  case CL_KERNEL_LOCAL_MEM_SIZE:
    valueSize = sizeof(cl_ulong);
    break;
  default:
    itkGenericExceptionMacro (<< "Unknown type of work goup information");
    break;
  }
#ifdef _DEBUG
  std::cout<<"clGetKernelWorkGroupInfo" << "..." << std::endl;
#endif
  cl_int errid = clGetKernelWorkGroupInfo(m_KernelContainer[kernelIdx], m_Manager->GetDeviceId(0),
    paramName, valueSize, value, &valueSizeRet);

  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  return errid;
}

bool GPUKernelManager::SetKernelArg(int kernelIdx, cl_uint argIdx, size_t argSize, const void* argVal)
{
  if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  cl_int errid;
#ifdef _DEBUG
  std::cout<<"clSetKernelArg" << "..." << std::endl;
#endif
  errid = clSetKernelArg(m_KernelContainer[kernelIdx], argIdx, argSize, argVal);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  m_KernelArgumentReady[kernelIdx][argIdx].m_IsReady = true;
  m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = (GPUDataManager::Pointer)NULL;

  return true;
}

bool GPUKernelManager::SetKernelArgWithImage(int kernelIdx, cl_uint argIdx, GPUDataManager::Pointer manager)
{
  if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  cl_int errid;
#ifdef _DEBUG
  std::cout<<"clSetKernelArg" << "..." << std::endl;
#endif
  errid = clSetKernelArg(m_KernelContainer[kernelIdx], argIdx, sizeof(cl_mem), manager->GetGPUBufferPointer());
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  m_KernelArgumentReady[kernelIdx][argIdx].m_IsReady = true;
  m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = manager;

  return true;
}

// this function must be called right before GPU kernel is launched
bool GPUKernelManager::CheckArgumentReady(int kernelIdx)
{
  int nArg = m_KernelArgumentReady[kernelIdx].size();

  for(int i=0; i<nArg; i++)
  {
    if(!(m_KernelArgumentReady[kernelIdx][i].m_IsReady)) return false;

    // automatic synchronization before kernel launch
    if(m_KernelArgumentReady[kernelIdx][i].m_GPUDataManager != (GPUDataManager::Pointer)NULL)
    {
      m_KernelArgumentReady[kernelIdx][i].m_GPUDataManager->SetCPUBufferDirty();
    }
  }
  return true;
}

void GPUKernelManager::ResetArguments(int kernelIdx)
{
  int nArg = m_KernelArgumentReady[kernelIdx].size();

  for(int i=0; i<nArg; i++)
  {
    m_KernelArgumentReady[kernelIdx][i].m_IsReady = false;
    m_KernelArgumentReady[kernelIdx][i].m_GPUDataManager = (GPUDataManager::Pointer)NULL;
  }
}

bool GPUKernelManager::LaunchKernel1D(int kernelIdx, size_t globalWorkSize, size_t localWorkSize)
{
  if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  if(!CheckArgumentReady(kernelIdx))
  {
    itkWarningMacro("GPU kernel arguments are not completely assigned");
    return false;
  }

  cl_int errid;
#ifdef _DEBUG
  std::cout<<"clEnqueueNDRangeKernel" << "..." << std::endl;
#endif
  errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId), m_KernelContainer[kernelIdx], 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  if(errid != CL_SUCCESS)
  {
    itkWarningMacro("GPU kernel launch failed");
    return false;
  }

  return true;
}

bool GPUKernelManager::LaunchKernel1D(int kernelIdx, size_t globalWorkSize)
{
  if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  if(!CheckArgumentReady(kernelIdx))
  {
    itkWarningMacro("GPU kernel arguments are not completely assigned");
    return false;
  }

  cl_int errid;
#ifdef _DEBUG
  std::cout<<"clEnqueueNDRangeKernel" << "..." << std::endl;
#endif
  errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId), m_KernelContainer[kernelIdx], 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  if(errid != CL_SUCCESS)
  {
    itkWarningMacro("GPU kernel launch failed");
    return false;
  }

  return true;
}

bool GPUKernelManager::LaunchKernel2D(int kernelIdx,
  size_t globalWorkSizeX, size_t globalWorkSizeY,
  size_t localWorkSizeX,  size_t localWorkSizeY )
{
  if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  if(!CheckArgumentReady(kernelIdx))
  {
    itkWarningMacro("GPU kernel arguments are not completely assigned");
    return false;
  }

  size_t gws[2], lws[2];

  gws[0] = globalWorkSizeX;
  gws[1] = globalWorkSizeY;

  lws[0] = localWorkSizeX;
  lws[1] = localWorkSizeY;

  cl_int errid;
#ifdef _DEBUG
  std::cout<<"clEnqueueNDRangeKernel" << "..." << std::endl;
#endif
  errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId), m_KernelContainer[kernelIdx], 2, NULL, gws, lws, 0, NULL, NULL);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  if(errid != CL_SUCCESS)
  {
    itkWarningMacro("GPU kernel launch failed");
    return false;
  }

  return true;
}

bool GPUKernelManager::LaunchKernel2D(int kernelIdx,
  size_t globalWorkSizeX, size_t globalWorkSizeY)
{
  if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  if(!CheckArgumentReady(kernelIdx))
  {
    itkWarningMacro("GPU kernel arguments are not completely assigned");
    return false;
  }

  size_t gws[2];

  gws[0] = globalWorkSizeX;
  gws[1] = globalWorkSizeY;

  cl_int errid;
#ifdef _DEBUG
  std::cout<<"clEnqueueNDRangeKernel" << "..." << std::endl;
#endif
  errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId), m_KernelContainer[kernelIdx], 2, NULL, gws, NULL, 0, NULL, NULL);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  if(errid != CL_SUCCESS)
  {
    itkWarningMacro("GPU kernel launch failed");
    return false;
  }

  return true;
}

bool GPUKernelManager::LaunchKernel3D(int kernelIdx,
  size_t globalWorkSizeX, size_t globalWorkSizeY, size_t globalWorkSizeZ,
  size_t localWorkSizeX,  size_t localWorkSizeY, size_t localWorkSizeZ )
{
  if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  if(!CheckArgumentReady(kernelIdx))
  {
    itkWarningMacro("GPU kernel arguments are not completely assigned");
    return false;
  }

  size_t gws[3], lws[3];

  gws[0] = globalWorkSizeX;
  gws[1] = globalWorkSizeY;
  gws[2] = globalWorkSizeZ;

  lws[0] = localWorkSizeX;
  lws[1] = localWorkSizeY;
  lws[2] = localWorkSizeZ;

  cl_int errid;
#ifdef _DEBUG
  std::cout<<"clEnqueueNDRangeKernel" << "..." << std::endl;
#endif
  errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId), m_KernelContainer[kernelIdx], 3, NULL, gws, lws, 0, NULL, NULL);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  if(errid != CL_SUCCESS)
  {
    itkWarningMacro("GPU kernel launch failed");
    return false;
  }

  return true;
}

bool GPUKernelManager::LaunchKernel3D(int kernelIdx,
  size_t globalWorkSizeX, size_t globalWorkSizeY, size_t globalWorkSizeZ)
{
  if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  if(!CheckArgumentReady(kernelIdx))
  {
    itkWarningMacro("GPU kernel arguments are not completely assigned");
    return false;
  }

  size_t gws[3];
  gws[0] = globalWorkSizeX;
  gws[1] = globalWorkSizeY;
  gws[2] = globalWorkSizeZ;

  cl_int errid;
#ifdef _DEBUG
  std::cout<<"clEnqueueNDRangeKernel" << "..." << std::endl;
#endif
  errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId), m_KernelContainer[kernelIdx], 3, NULL, gws, NULL, 0, NULL, NULL);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  if(errid != CL_SUCCESS)
  {
    itkWarningMacro("GPU kernel launch failed");
    return false;
  }

  return true;
}

bool GPUKernelManager::LaunchKernel(int kernelIdx, int dim, size_t *globalWorkSize, size_t *localWorkSize)
{
  if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  if(!CheckArgumentReady(kernelIdx))
  {
    itkWarningMacro("GPU kernel arguments are not completely assigned");
    return false;
  }

  cl_int errid;
  cl_event clEvent = NULL;

#ifdef _DEBUG
  std::cout<<"clEnqueueNDRangeKernel" << "..." << std::endl;
#endif

#ifdef OPENCL_PROFILING
  errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId),
    m_KernelContainer[kernelIdx], dim, NULL, globalWorkSize, localWorkSize,
    0, NULL, &clEvent);
#else
  errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId),
    m_KernelContainer[kernelIdx], dim, NULL, globalWorkSize, localWorkSize,
    0, NULL, NULL);
#endif

  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
  m_Manager->OpenCLProfile(clEvent, "clEnqueueNDRangeKernel");

  if(errid != CL_SUCCESS)
  {
    itkWarningMacro("GPU kernel launch failed");
    return false;
  }

  return true;
}

bool GPUKernelManager::LaunchKernel(int kernelIdx, int dim, size_t *globalWorkSize)
{
	if(kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

	if(!CheckArgumentReady(kernelIdx))
	{
		itkWarningMacro("GPU kernel arguments are not completely assigned");
		return false;
	}

	cl_int errid;
	cl_event clEvent = NULL;

#ifdef _DEBUG
	std::cout<<"clEnqueueNDRangeKernel" << "..." << std::endl;
#endif

#ifdef OPENCL_PROFILING
	errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId),
		m_KernelContainer[kernelIdx], dim, NULL, globalWorkSize, NULL,
		0, NULL, &clEvent);
#else
	errid = clEnqueueNDRangeKernel(m_Manager->GetCommandQueue(m_CommandQueueId),
		m_KernelContainer[kernelIdx], dim, NULL, globalWorkSize, NULL,
		0, NULL, NULL);
#endif

	OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
	m_Manager->OpenCLProfile(clEvent, "clEnqueueNDRangeKernel");

	if(errid != CL_SUCCESS)
	{
		itkWarningMacro("GPU kernel launch failed");
		return false;
	}

	return true;
}

void GPUKernelManager::SetCurrentCommandQueue( int queueid )
{
  if( queueid >= 0 && queueid < (int)m_Manager->GetNumberOfCommandQueues() )
  {
    // Assumption: different command queue is assigned to different device
    m_CommandQueueId = queueid;
  }
  else
  {
    itkWarningMacro("Not a valid command queue id");
  }
}

int GPUKernelManager::GetCurrentCommandQueueID()
{
  return m_CommandQueueId;
}

} // end namespace itk
