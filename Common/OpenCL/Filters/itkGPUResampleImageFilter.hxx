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
#ifndef itkGPUResampleImageFilter_hxx
#define itkGPUResampleImageFilter_hxx

#include "itkGPUResampleImageFilter.h"

#include "itkGPUKernelManagerHelperFunctions.h"
#include "itkGPUMath.h"
#include "itkGPUImageBase.h"

#include "itkImageLinearIteratorWithIndex.h"
#include "itkTimeProbe.h"
#include "itkImageRegionSplitterSlowDimension.h"

#include "itkOpenCLUtil.h"
#include "itkOpenCLKernelToImageBridge.h"

namespace
{
typedef struct
{
  cl_float2 min_max;
  cl_float2 min_max_output;
  cl_float  default_value;
  cl_float  dummy_for_alignment;
} FilterParameters;
} // end of unnamed namespace

namespace itk
{
/**
 * ***************** Constructor ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  GPUResampleImageFilter()
{
  this->m_PreKernelManager = OpenCLKernelManager::New();
  this->m_LoopKernelManager = OpenCLKernelManager::New();
  this->m_PostKernelManager = OpenCLKernelManager::New();

  this->m_InputGPUImageBase = GPUDataManager::New();
  this->m_OutputGPUImageBase = GPUDataManager::New();

  this->m_FilterParameters = GPUDataManager::New();
  this->m_FilterParameters->Initialize();
  this->m_FilterParameters->SetBufferFlag(CL_MEM_READ_ONLY);
  this->m_FilterParameters->SetBufferSize(sizeof(FilterParameters));
  this->m_FilterParameters->Allocate();

  this->m_DeformationFieldBuffer = GPUDataManager::New();

  this->m_InterpolatorSourceLoadedIndex = 0;
  this->m_TransformSourceLoadedIndex = 0;

  this->m_InterpolatorIsBSpline = false; // make it protected in base class
  this->m_TransformIsCombo = false;

  // Set all handlers to -1;
  this->m_FilterPreGPUKernelHandle = -1;
  this->m_FilterPostGPUKernelHandle = -1;

  this->m_InterpolatorBase = nullptr;
  this->m_TransformBase = nullptr;

  this->m_RequestedNumberOfSplits = 5;

  std::ostringstream defines;
  if (TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1)
  {
    itkExceptionMacro("GPUResampleImageFilter supports 1/2/3D image.");
  }

  defines << "#define DIM_" << int(TInputImage::ImageDimension) << "\n";
  defines << "#define INPIXELTYPE ";
  GetTypenameInString(typeid(typename TInputImage::PixelType), defines);
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString(typeid(typename TOutputImage::PixelType), defines);
  // defines << "#define INTERPOLATOR_PRECISION_TYPE ";
  // GetTypenameInString( typeid( TInterpolatorPrecisionType ), defines );

  // Resize m_Sources according to the number of source files
  // Defines source code for GPUMath, GPUImageBase, GPUResampleImageFilter
  const unsigned int numberOfIncludes = 4;
  // Defines source code for GPUInterpolator, GPUTransform,
  const unsigned int numberOfSources = 2;

  this->m_Sources.resize(numberOfIncludes + numberOfSources);
  this->m_SourceIndex = 0;

  // Add defines
  this->m_Sources[this->m_SourceIndex++] = defines.str();

  // Get GPUMath source
  const std::string oclMathSource(GPUMathKernel::GetOpenCLSource());
  this->m_Sources[this->m_SourceIndex++] = oclMathSource;

  // Get GPUImageBase source
  const std::string oclImageBaseSource(GPUImageBaseKernel::GetOpenCLSource());
  this->m_Sources[this->m_SourceIndex++] = oclImageBaseSource;

  // Get GPUResampleImageFilter source
  const std::string oclResampleImageFilterSource(GPUResampleImageFilterKernel::GetOpenCLSource());
  this->m_Sources[this->m_SourceIndex++] = oclResampleImageFilterSource;

  // Construct ResampleImageFilter Pre code
  std::ostringstream resamplePreSource;
  resamplePreSource << "#define RESAMPLE_PRE\n";
  resamplePreSource << this->m_Sources[1]; // GPUMath source
  resamplePreSource << this->m_Sources[2]; // GPUImageBase source
  resamplePreSource << this->m_Sources[3]; // GPUResampleImageFilter source

  // Build and create kernel
  const OpenCLProgram program =
    this->m_PreKernelManager->BuildProgramFromSourceCode(resamplePreSource.str(), defines.str());
  if (program.IsNull())
  {
    itkExceptionMacro(<< "Kernel has not been loaded from string:\n"
                      << defines.str() << '\n'
                      << resamplePreSource.str());
  }
  this->m_FilterPreGPUKernelHandle = this->m_PreKernelManager->CreateKernel(program, "ResampleImageFilterPre");
} // end Constructor


/**
 * ***************** SetInterpolator ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::SetInterpolator(
  InterpolatorType * _arg)
{
  itkDebugMacro("setting Interpolator to " << _arg);
  CPUSuperclass::SetInterpolator(_arg);

  /** Test for a supported GPU interpolator. */
  const GPUInterpolatorBase * interpolatorBase = dynamic_cast<const GPUInterpolatorBase *>(_arg);
  if (!interpolatorBase)
  {
    itkExceptionMacro("Setting unsupported GPU interpolator to " << _arg);
  }
  this->m_InterpolatorBase = (GPUInterpolatorBase *)interpolatorBase;

  // Test for a GPU B-spline interpolator
  const GPUBSplineInterpolatorType * GPUBSplineInterpolator = dynamic_cast<const GPUBSplineInterpolatorType *>(_arg);
  this->m_InterpolatorIsBSpline = false;
  if (GPUBSplineInterpolator)
  {
    this->m_InterpolatorIsBSpline = true;
  }

  // Get interpolator source
  std::string interpolatorSource;
  if (!interpolatorBase->GetSourceCode(interpolatorSource))
  {
    itkExceptionMacro(<< "Unable to get interpolator source code.");
  }

  // Construct ResampleImageFilter Post code
  const std::string  defines = m_Sources[0];
  std::ostringstream resamplePostSource;
  resamplePostSource << "#define RESAMPLE_POST\n";

  if (this->m_InterpolatorIsBSpline)
  {
    resamplePostSource << "#define BSPLINE_INTERPOLATOR\n";
  }

  resamplePostSource << this->m_Sources[1]; // GPUMath source
  resamplePostSource << this->m_Sources[2]; // GPUImageBase source
  resamplePostSource << interpolatorSource;
  resamplePostSource << this->m_Sources[3]; // GPUResampleImageFilter source

  // Build and create kernel
  const OpenCLProgram program =
    this->m_PostKernelManager->BuildProgramFromSourceCode(resamplePostSource.str(), defines);
  if (program.IsNull())
  {
    itkExceptionMacro(<< "Kernel has not been loaded from string:\n" << defines << '\n' << resamplePostSource.str());
  }

  if (this->m_InterpolatorIsBSpline)
  {
    this->m_FilterPostGPUKernelHandle =
      this->m_PostKernelManager->CreateKernel(program, "ResampleImageFilterPost_BSplineInterpolator");
  }
  else
  {
    this->m_FilterPostGPUKernelHandle = this->m_PostKernelManager->CreateKernel(program, "ResampleImageFilterPost");
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetInterpolator() finished");
} // end SetInterpolator()


/**
 * ***************** SetExtrapolator ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::SetExtrapolator(
  ExtrapolatorType * _arg)
{
  // CPUSuperclass::SetExtrapolator( _arg );
  itkWarningMacro(<< "Setting Extrapolator for GPUResampleImageFilter not supported yet.");
} // end SetExtrapolator()


/**
 * ***************** SetTransform ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::SetTransform(
  const TransformType * _arg)
{
  itkDebugMacro("setting Transform to " << _arg);
  CPUSuperclass::SetTransform(_arg);

  /** Test for a supported GPU transform. */
  const GPUTransformBase * transformBase = dynamic_cast<const GPUTransformBase *>(_arg);
  if (!transformBase)
  {
    itkExceptionMacro("Setting unsupported GPU transform to " << _arg);
  }
  this->m_TransformBase = (GPUTransformBase *)transformBase;

  // Erase map of supported transforms
  this->m_FilterLoopGPUKernelHandle.clear();

  // Test for a GPU combo transform
  using CompositeTransformType = GPUCompositeTransformBase<InterpolatorPrecisionType, InputImageDimension>;
  const CompositeTransformType * compositeTransformBase = dynamic_cast<const CompositeTransformType *>(_arg);

  if (compositeTransformBase)
  {
    this->m_TransformIsCombo = true;

    // Construct m_FilterLoopGPUKernelHandle
    TransformHandle identitytransform(-1, compositeTransformBase->HasIdentityTransform());
    TransformHandle matrixoffsettransform(-1, compositeTransformBase->HasMatrixOffsetTransform());
    TransformHandle translationtransform(-1, compositeTransformBase->HasTranslationTransform());
    TransformHandle bsplinetransform(-1, compositeTransformBase->HasBSplineTransform());

    this->m_FilterLoopGPUKernelHandle[IdentityTransform] = identitytransform;
    this->m_FilterLoopGPUKernelHandle[MatrixOffsetTransform] = matrixoffsettransform;
    this->m_FilterLoopGPUKernelHandle[TranslationTransform] = translationtransform;
    this->m_FilterLoopGPUKernelHandle[BSplineTransform] = bsplinetransform;
  }
  else
  {
    this->m_TransformIsCombo = false;

    // Construct m_FilterLoopGPUKernelHandle
    TransformHandle identitytransform(-1, transformBase->IsIdentityTransform());
    TransformHandle matrixoffsettransform(-1, transformBase->IsMatrixOffsetTransform());
    TransformHandle translationtransform(-1, transformBase->IsTranslationTransform());
    TransformHandle bsplinetransform(-1, transformBase->IsBSplineTransform());

    this->m_FilterLoopGPUKernelHandle[IdentityTransform] = identitytransform;
    this->m_FilterLoopGPUKernelHandle[MatrixOffsetTransform] = matrixoffsettransform;
    this->m_FilterLoopGPUKernelHandle[TranslationTransform] = translationtransform;
    this->m_FilterLoopGPUKernelHandle[BSplineTransform] = bsplinetransform;
  }

  // Get transform source
  std::string transformSource;
  if (!transformBase->GetSourceCode(transformSource))
  {
    itkExceptionMacro(<< "Unable to get transform source code.");
  }

  // Construct ResampleImageFilter Loop code
  const std::string  defines = this->m_Sources[0];
  std::ostringstream resampleLoopSource;
  resampleLoopSource << "#define RESAMPLE_LOOP\n";

  // todo: can we clean this up
  // like: for all transforms that exist print #define GetTransformName()
  if (this->HasTransform(IdentityTransform))
  {
    resampleLoopSource << "#define IDENTITY_TRANSFORM\n";
  }
  if (this->HasTransform(MatrixOffsetTransform))
  {
    resampleLoopSource << "#define MATRIX_OFFSET_TRANSFORM\n";
  }
  if (this->HasTransform(TranslationTransform))
  {
    resampleLoopSource << "#define TRANSLATION_TRANSFORM\n";
  }
  if (this->HasTransform(BSplineTransform))
  {
    resampleLoopSource << "#define BSPLINE_TRANSFORM\n";
  }

  resampleLoopSource << this->m_Sources[1]; // GPUMath source
  resampleLoopSource << this->m_Sources[2]; // GPUImageBase source
  resampleLoopSource << transformSource;
  resampleLoopSource << this->m_Sources[3]; // GPUResampleImageFilter source

  // Build and create kernel
  const OpenCLProgram program =
    this->m_LoopKernelManager->BuildProgramFromSourceCode(resampleLoopSource.str(), defines);
  if (program.IsNull())
  {
    itkExceptionMacro(<< "Kernel has not been loaded from string:\n" << defines << '\n' << resampleLoopSource.str());
  }

  // \todo: can we clean this up?
  // like: for all transforms that exist create the correct kernel
  // Loop over all supported transform types
#if 0
  typename TransformsHandle::const_iterator it = this->m_FilterLoopGPUKernelHandle.begin();
  for(; it != this->m_FilterLoopGPUKernelHandle.end(); ++it )
  {
    // Skip transform, that means that it is not present
    if( !it->second.second ) { continue; }

    const GPUInputTransformType this->GetTransformType

    this->m_FilterLoopGPUKernelHandle[ IdentityTransform ].first
      = this->m_LoopKernelManager->CreateKernel( program, "ResampleImageFilterLoop_IdentityTransform" );

  }
#endif

  if (this->HasTransform(IdentityTransform))
  {
    this->m_FilterLoopGPUKernelHandle[IdentityTransform].first =
      this->m_LoopKernelManager->CreateKernel(program, "ResampleImageFilterLoop_IdentityTransform");
  }
  if (this->HasTransform(MatrixOffsetTransform))
  {
    this->m_FilterLoopGPUKernelHandle[MatrixOffsetTransform].first =
      this->m_LoopKernelManager->CreateKernel(program, "ResampleImageFilterLoop_MatrixOffsetTransform");
  }
  if (this->HasTransform(TranslationTransform))
  {
    this->m_FilterLoopGPUKernelHandle[TranslationTransform].first =
      this->m_LoopKernelManager->CreateKernel(program, "ResampleImageFilterLoop_TranslationTransform");
  }
  if (this->HasTransform(BSplineTransform))
  {
    this->m_FilterLoopGPUKernelHandle[BSplineTransform].first =
      this->m_LoopKernelManager->CreateKernel(program, "ResampleImageFilterLoop_BSplineTransform");
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetTransform() finished");
} // end SetTransform()


/**
 * ***************** GPUGenerateData ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  GPUGenerateData()
{
  itkDebugMacro(<< "GPUResampleImageFilter::GPUGenerateData() called");

  // Profiling
#ifdef OPENCL_PROFILING
  TimeProbe gputimer;
  gputimer.Start();
#endif

  // Get handles to the input and output images
  const typename GPUInputImage::Pointer inPtr = dynamic_cast<GPUInputImage *>(this->ProcessObject::GetInput(0));
  typename GPUOutputImage::Pointer      outPtr = dynamic_cast<GPUOutputImage *>(this->ProcessObject::GetOutput(0));

  // Perform safety checks
  if (inPtr.IsNull())
  {
    itkExceptionMacro(<< "The GPU InputImage is NULL. Filter unable to perform.");
    return;
  }
  if (outPtr.IsNull())
  {
    itkExceptionMacro(<< "The GPU OutputImage is NULL. Filter unable to perform.");
    return;
  }

  // Get the largest possible output region.
  const OutputImageRegionType outputLargestRegion = outPtr->GetLargestPossibleRegion();
  if (outputLargestRegion.GetNumberOfPixels() == 0)
  {
    itkExceptionMacro(<< "GPUResampleImageFilter has not been properly initialized. Filter unable to perform.");
    return;
  }

  // Define filter parameters:
  // defaultValue, minValue/maxValue, minOutputValue/maxOutputValue
  FilterParameters parameters;
  parameters.default_value = static_cast<float>(this->GetDefaultPixelValue());
  parameters.min_max.s[0] = static_cast<float>(NumericTraits<OutputImagePixelType>::NonpositiveMin());
  parameters.min_max.s[1] = static_cast<float>(NumericTraits<OutputImagePixelType>::max());
  parameters.min_max_output.s[0] = parameters.min_max.s[0];
  parameters.min_max_output.s[1] = parameters.min_max.s[1];

  // Set them to the GPU
  this->m_FilterParameters->SetCPUBufferPointer(&parameters);
  this->m_FilterParameters->SetGPUDirtyFlag(true);
  this->m_FilterParameters->UpdateGPUBuffer();

  // Define the number of chunks in which we process the image.
  // For now we fix it to a constant value, later we can support user defined
  // splits in manual or auto mode. For auto definition, all GPU memory
  // allocation within ITK OpenCL has to be tracked and the number of splits
  // have to be computed based on the remaining global GPU memory.
  // Splitting is not used for low-dimensional images.
  unsigned int requestedNumberOfSplits = this->m_RequestedNumberOfSplits;
  if (InputImageDimension < 3)
  {
    requestedNumberOfSplits = 1;
  }

  using RegionSplitterType = ImageRegionSplitterSlowDimension;
  auto               splitter = RegionSplitterType::New();
  const unsigned int numberOfChunks = splitter->GetNumberOfSplits(outputLargestRegion, requestedNumberOfSplits);

  // Get the maximum chunk size
  SizeType maxChunkSize;
  maxChunkSize.Fill(0);
  for (unsigned int i = 0; i < numberOfChunks; ++i)
  {
    OutputImageRegionType currentChunkRegion = outputLargestRegion;
    splitter->GetSplit(i, numberOfChunks, currentChunkRegion);

    const SizeType currentChunkSize = currentChunkRegion.GetSize();
    std::size_t    cSize = 1, mSize = 1;
    for (unsigned int i = 0; i < OutputImageDimension; ++i)
    {
      cSize *= currentChunkSize[i];
      mSize *= maxChunkSize[i];
    }

    if (cSize > mSize)
    {
      maxChunkSize = currentChunkSize;
    }
  }

  // Create and allocate the deformation field buffer
  // The deformation field size equals the maximum chunk size
  std::size_t totalDFSize = 1;
  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    totalDFSize *= maxChunkSize[i];
  }

  unsigned int mem_size_DF = 0;
  switch (static_cast<unsigned int>(OutputImageDimension))
  {
    case 1:
      mem_size_DF = totalDFSize * sizeof(cl_float);
      break;
    case 2:
      mem_size_DF = totalDFSize * sizeof(cl_float2);
      break;
    case 3:
      mem_size_DF = totalDFSize * sizeof(cl_float3);
      break;
    default:
      break;
  }

  this->m_DeformationFieldBuffer->Initialize();
  this->m_DeformationFieldBuffer->SetBufferFlag(CL_MEM_READ_WRITE);
  this->m_DeformationFieldBuffer->SetBufferSize(mem_size_DF);
  this->m_DeformationFieldBuffer->Allocate();

  // Set arguments for pre kernel
  this->SetArgumentsForPreKernelManager(outPtr);

  // Set arguments for loop kernel
  this->SetArgumentsForLoopKernelManager(inPtr, outPtr);
  if (!this->m_TransformIsCombo) // move below
  {
    this->SetTransformParametersForLoopKernelManager(0);
  }

  // Set arguments for post kernel
  this->SetArgumentsForPostKernelManager(inPtr, outPtr);

  // Define global and local work size
  const OpenCLSize localWorkSize =
    OpenCLSize::GetLocalWorkSize(this->m_PreKernelManager->GetContext()->GetDefaultDevice());
  std::size_t local3D[3], local2D[2], local1D;

  local3D[0] = local2D[0] = local1D = localWorkSize[0];
  local3D[1] = local2D[1] = localWorkSize[1];
  local3D[2] = localWorkSize[2];

  cl_uint3    dfsize3D;
  cl_uint2    dfsize2D;
  cl_uint     dfsize1D;
  std::size_t global3D[3], global2D[2], global1D;
  std::size_t offset3D[3], offset2D[2], offset1D;

  // Some temporaries
  OpenCLEventList eventList;
  unsigned int    piece;
  OpenCLSize      global_work_size;
  OpenCLSize      global_work_offset;

  /** Loop over the chunks. */
  for (piece = 0; piece < numberOfChunks && !this->GetAbortGenerateData(); ++piece)
  {
    // Get the current chunk region.
    OutputImageRegionType currentChunkRegion = outputLargestRegion;
    splitter->GetSplit(piece, numberOfChunks, currentChunkRegion);

    // define and set deformation field size, global_work_size and global_work_offset
    // The deformation field size is the second argument in the
    // pre/loop/post kernel, i.e. index is 1.
    const cl_uint dfSizeKernelIndex = 1;
    switch (static_cast<unsigned int>(OutputImageDimension))
    {
      case 1:
      {
        dfsize1D = currentChunkRegion.GetSize(0);
        global1D = local1D * (unsigned int)ceil((float)dfsize1D / (float)local1D);
        offset1D = currentChunkRegion.GetIndex(0);

        // set dfsize argument
        this->m_PreKernelManager->SetKernelArgForAllKernels(dfSizeKernelIndex, sizeof(cl_uint), (void *)&dfsize1D);

        this->m_LoopKernelManager->SetKernelArgForAllKernels(dfSizeKernelIndex, sizeof(cl_uint), (void *)&dfsize1D);

        this->m_PostKernelManager->SetKernelArgForAllKernels(dfSizeKernelIndex, sizeof(cl_uint), (void *)&dfsize1D);

        global_work_size = OpenCLSize(global1D);
        global_work_offset = OpenCLSize(offset1D);
      }
      break;
      case 2:
      {
        for (unsigned int i = 0; i < 2; ++i)
        {
          dfsize2D.s[i] = currentChunkRegion.GetSize(i);
          global2D[i] = local2D[i] * (unsigned int)ceil((float)dfsize2D.s[i] / (float)local2D[i]);
          offset2D[i] = currentChunkRegion.GetIndex(i);
        }

        // set dfsize argument
        this->m_PreKernelManager->SetKernelArgForAllKernels(dfSizeKernelIndex, sizeof(cl_uint2), (void *)&dfsize2D);

        this->m_LoopKernelManager->SetKernelArgForAllKernels(dfSizeKernelIndex, sizeof(cl_uint2), (void *)&dfsize2D);

        this->m_PostKernelManager->SetKernelArgForAllKernels(dfSizeKernelIndex, sizeof(cl_uint2), (void *)&dfsize2D);

        global_work_size = OpenCLSize(global2D[0], global2D[1]);
        global_work_offset = OpenCLSize(offset2D[0], offset2D[1]);
      }
      break;
      case 3:
      {
        for (unsigned int i = 0; i < 3; ++i)
        {
          dfsize3D.s[i] = currentChunkRegion.GetSize(i);
          global3D[i] = local3D[i] * (unsigned int)ceil((float)dfsize3D.s[i] / (float)local3D[i]);
          offset3D[i] = currentChunkRegion.GetIndex(i);
        }
        dfsize3D.s[3] = 0;

        // set dfsize argument
        this->m_PreKernelManager->SetKernelArgForAllKernels(dfSizeKernelIndex, sizeof(cl_uint3), (void *)&dfsize3D);

        this->m_LoopKernelManager->SetKernelArgForAllKernels(dfSizeKernelIndex, sizeof(cl_uint3), (void *)&dfsize3D);

        this->m_PostKernelManager->SetKernelArgForAllKernels(dfSizeKernelIndex, sizeof(cl_uint3), (void *)&dfsize3D);

        global_work_size = OpenCLSize(global3D[0], global3D[1], global3D[2]);
        global_work_offset = OpenCLSize(offset3D[0], offset3D[1], offset3D[2]);
      }
      break;
      default:
        break;
    }

    // Set global work size and offset for all kernels
    this->m_PreKernelManager->SetGlobalWorkSizeForAllKernels(global_work_size);
    this->m_PreKernelManager->SetGlobalWorkOffsetForAllKernels(global_work_offset);

    this->m_LoopKernelManager->SetGlobalWorkSizeForAllKernels(global_work_size);
    this->m_LoopKernelManager->SetGlobalWorkOffsetForAllKernels(global_work_offset);

    this->m_PostKernelManager->SetGlobalWorkSizeForAllKernels(global_work_size);
    this->m_PostKernelManager->SetGlobalWorkOffsetForAllKernels(global_work_offset);

    // Launch pre kernel
#if 0 // this should work in theory but doesn't
    OpenCLEvent preEvent = this->m_PreKernelManager->LaunchKernel( this->m_FilterPreGPUKernelHandle, eventList );
    eventList.Append( preEvent );
#else
    if (eventList.GetSize() == 0)
    {
      OpenCLEvent preEvent = this->m_PreKernelManager->LaunchKernel(this->m_FilterPreGPUKernelHandle);
      eventList.Append(preEvent);
    }
    else
    {
      OpenCLEvent preEvent = this->m_PreKernelManager->LaunchKernel(this->m_FilterPreGPUKernelHandle, eventList);
      eventList.Append(preEvent);
    }
#endif

    // Launch all the loop kernels
    if (this->m_TransformIsCombo)
    {
      using CompositeTransformType = GPUCompositeTransformBase<InterpolatorPrecisionType, InputImageDimension>;
      const CompositeTransformType * compositeTransform =
        dynamic_cast<const CompositeTransformType *>(this->m_TransformBase);

      for (int i = compositeTransform->GetNumberOfTransforms() - 1; i >= 0; i--)
      {
        /** Set the transform parameters to the loop kernel. */
        this->SetTransformParametersForLoopKernelManager(i);

        /** Get the kernel id for this transform and launch it. */
        std::size_t kernelId = 1e10;
        this->GetKernelIdFromTransformId(i, kernelId);
        OpenCLEvent loopEvent = this->m_LoopKernelManager->LaunchKernel(kernelId, eventList);
        eventList.Append(loopEvent);

      } // end loop over the list of transforms
    }   // end if is combo
    else
    {
      /** Get the kernel id for this transform and launch it. */
      std::size_t kernelId = 1e10;
      this->GetKernelIdFromTransformId(0, kernelId); // 0 is dummy for non-combo transform
      OpenCLEvent loopEvent = this->m_LoopKernelManager->LaunchKernel(kernelId, eventList);
      eventList.Append(loopEvent);
    }

    // Launch the post kernel
    OpenCLEvent postEvent = this->m_PostKernelManager->LaunchKernel(this->m_FilterPostGPUKernelHandle, eventList);
    eventList.Append(postEvent);
  }

  eventList.WaitForFinished();

  itkDebugMacro(<< "GPUResampleImageFilter::GPUGenerateData() finished");
} // end GPUGenerateData()


/**
 * ***************** SetArgumentsForPreKernelManager ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  SetArgumentsForPreKernelManager(const typename GPUOutputImage::Pointer & outputImage)
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForPreKernelManager called");

  // Get a handle to the pre kernel
  cl_uint        argidx = 0;
  OpenCLKernel & preKernel = this->m_PreKernelManager->GetKernel(this->m_FilterPreGPUKernelHandle);

  // Set deformation field to the kernel
  this->m_PreKernelManager->SetKernelArgWithImage(
    this->m_FilterPreGPUKernelHandle, argidx++, this->m_DeformationFieldBuffer);

  ++argidx; // skip deformation field size for now

  // Set output image index_to_physical_point to the kernel
  OpenCLKernelToImageBridge<OutputImageType>::SetDirection(preKernel, argidx++, outputImage->GetIndexToPhysicalPoint());

  // Set output image origin to the kernel
  OpenCLKernelToImageBridge<OutputImageType>::SetOrigin(preKernel, argidx++, outputImage->GetOrigin());

  // Set output image size to the kernel
  OpenCLKernelToImageBridge<OutputImageType>::SetSize(
    preKernel, argidx++, outputImage->GetLargestPossibleRegion().GetSize());

  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForPreKernelManager() finished");
} // end SetArgumentsForPreKernelManager()


/**
 * ***************** SetArgumentsForLoopKernelManager ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  SetArgumentsForLoopKernelManager(const typename GPUInputImage::Pointer &  input,
                                   const typename GPUOutputImage::Pointer & output)
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForLoopKernelManager(" << input->GetNameOfClass() << ", "
                << output->GetNameOfClass() << ") called");

  // Loop over all supported transform types
  typename TransformsHandle::const_iterator it = this->m_FilterLoopGPUKernelHandle.begin();
  for (; it != this->m_FilterLoopGPUKernelHandle.end(); ++it)
  {
    // Skip transform, that means that it is not present
    if (!it->second.second)
    {
      continue;
    }

    // Get handle to the kernel
    cl_uint   argidx = 0;
    const int handleId = it->second.first;

    // Get the loop kernel
    OpenCLKernel & loopKernel = this->m_LoopKernelManager->GetKernel(handleId);

    // Set deformation field buffer to the kernel
    this->m_LoopKernelManager->SetKernelArgWithImage(handleId, argidx++, this->m_DeformationFieldBuffer);

    ++argidx; // skip deformation field size for now

    // Set output image size to the kernel
    OpenCLKernelToImageBridge<OutputImageType>::SetSize(
      loopKernel, argidx++, output->GetLargestPossibleRegion().GetSize());
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForLoopKernelManager() finished");
} // end SetArgumentsForLoopKernelManager()


/**
 * ***************** SetTransformArgumentsForLoopKernelManager ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  SetTransformParametersForLoopKernelManager(const std::size_t transformIndex)
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetTransformArgumentsForLoopKernelManager(" << transformIndex
                << ") called");

  const GPUTransformTypeEnum transformType = this->GetTransformType(transformIndex);

  if (transformType == GPUResampleImageFilter::TranslationTransform ||
      transformType == GPUResampleImageFilter::MatrixOffsetTransform)
  {
    const cl_uint transformParametersLoopKernelIndex = 3;
    // Set the transform parameters
    std::size_t kernelId = 1e10;
    this->GetKernelIdFromTransformId(transformIndex, kernelId);
    this->m_LoopKernelManager->SetKernelArgWithImage(
      kernelId, transformParametersLoopKernelIndex, this->m_TransformBase->GetParametersDataManager(transformIndex));
  }
  else if (transformType == GPUResampleImageFilter::BSplineTransform)
  {
    const cl_uint bsplineTransformOrderLoopKernelIndex = 3;

    // Set the B-spline transform spline order
    std::size_t kernelId = 1e10;
    this->GetKernelIdFromTransformId(transformIndex, kernelId);

    GPUBSplineBaseTransformType * GPUBSplineTransformBase = this->GetGPUBSplineBaseTransform(transformIndex);

    const cl_uint splineOrder = GPUBSplineTransformBase->GetSplineOrder();
    this->m_LoopKernelManager->SetKernelArg(
      kernelId, bsplineTransformOrderLoopKernelIndex, sizeof(cl_uint), (void *)&splineOrder);

    // Set the B-spline coefficient images
    this->SetBSplineTransformCoefficientsToGPU(transformIndex);
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetTransformArgumentsForLoopKernelManager() finished");
} // end SetTransformArgumentsForLoopKernelManager()


/**
 * ***************** SetBSplineTransformCoefficientsToGPU ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  SetBSplineTransformCoefficientsToGPU(const std::size_t transformIndex)
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetBSplineTransformCoefficientsToGPU(" << transformIndex << ") called");

  // Typedefs
  using GPUBSplineTransformType = GPUBSplineBaseTransform<InterpolatorPrecisionType, InputImageDimension>;

  using GPUCoefficientImageType = typename GPUBSplineTransformType::GPUCoefficientImageType;
  using GPUCoefficientImageArray = typename GPUBSplineTransformType::GPUCoefficientImageArray;
  using GPUCoefficientImageBaseArray = typename GPUBSplineTransformType::GPUCoefficientImageBaseArray;
  using GPUCoefficientImagePointer = typename GPUBSplineTransformType::GPUCoefficientImagePointer;
  using GPUDataManagerPointer = typename GPUBSplineTransformType::GPUDataManagerPointer;

  // Local variables
  const cl_uint coefficientsImageLoopKernelIndex = 4;
  cl_uint       argidx = coefficientsImageLoopKernelIndex;

  GPUBSplineBaseTransformType * GPUBSplineTransformBase = this->GetGPUBSplineBaseTransform(transformIndex);

  // Get all coefficient images.
  GPUCoefficientImageArray     gpuCoefficientImages = GPUBSplineTransformBase->GetGPUCoefficientImages();
  GPUCoefficientImageBaseArray gpuCoefficientImagesBases = GPUBSplineTransformBase->GetGPUCoefficientImagesBases();

  // Get a handle to the B-spline transform kernel.
  this->m_LoopKernelManager->GetKernel(this->GetTransformHandle(BSplineTransform));

  // Set the B-spline coefficient image meta information to the kernel.
  GPUCoefficientImagePointer coefficient = gpuCoefficientImages[0];
  GPUDataManagerPointer      coefficientbase = gpuCoefficientImagesBases[0];

  SetKernelWithITKImage<GPUCoefficientImageType>(this->m_LoopKernelManager,
                                                 this->GetTransformHandle(BSplineTransform),
                                                 argidx,
                                                 coefficient,
                                                 coefficientbase,
                                                 false,
                                                 true);

  // Set the B-spline coefficient images to the kernel.
  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    coefficient = gpuCoefficientImages[i];
    coefficientbase = gpuCoefficientImagesBases[i];

    // Set output image to the kernel
    SetKernelWithITKImage<GPUCoefficientImageType>(this->m_LoopKernelManager,
                                                   this->GetTransformHandle(BSplineTransform),
                                                   argidx,
                                                   coefficient,
                                                   coefficientbase,
                                                   true,
                                                   false);
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetBSplineTransformCoefficientsToGPU() finished");
} // end SetBSplineTransformCoefficientsToGPU()


/**
 * ***************** SetArgumentsForPostKernelManager ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  SetArgumentsForPostKernelManager(const typename GPUInputImage::Pointer &  input,
                                   const typename GPUOutputImage::Pointer & output)
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForPostKernelManager(" << input->GetNameOfClass() << ", "
                << output->GetNameOfClass() << ") called");

  // Get a handle to the post kernel
  OpenCLKernel & postKernel = this->m_PostKernelManager->GetKernel(this->m_FilterPostGPUKernelHandle);

  cl_uint argidx = 0;

  // Set deformation field buffer to the kernel
  this->m_PostKernelManager->SetKernelArgWithImage(
    this->m_FilterPostGPUKernelHandle, argidx++, this->m_DeformationFieldBuffer);

  ++argidx; // skip deformation field size for now

  // Most interpolators work on the input image.
  // The B-spline interpolator however, works on the coefficients image,
  // previously generated by the BSplineDecompositionImageFilter.
  if (!this->m_InterpolatorIsBSpline)
  {
    SetKernelWithITKImage<GPUInputImage>(this->m_PostKernelManager,
                                         this->m_FilterPostGPUKernelHandle,
                                         argidx,
                                         input,
                                         this->m_InputGPUImageBase,
                                         true,
                                         true);
  }
  else
  {
    // Get a handle to the B-spline interpolator.
    const GPUBSplineInterpolatorType * gpuBSplineInterpolator =
      dynamic_cast<const GPUBSplineInterpolatorType *>(this->m_InterpolatorBase);

    // Get a handle to the B-spline interpolator coefficient image.
    GPUBSplineInterpolatorCoefficientImagePointer coefficient = gpuBSplineInterpolator->GetGPUCoefficients();
    GPUBSplineInterpolatorDataManagerPointer coefficientbase = gpuBSplineInterpolator->GetGPUCoefficientsImageBase();

    SetKernelWithITKImage<GPUBSplineInterpolatorCoefficientImageType>(
      this->m_PostKernelManager, this->m_FilterPostGPUKernelHandle, argidx, coefficient, coefficientbase, true, true);

    // Set the B-spline interpolator spline order
    const cl_uint splineOrder = gpuBSplineInterpolator->GetSplineOrder();
    this->m_PostKernelManager->SetKernelArg(
      this->m_FilterPostGPUKernelHandle, argidx++, sizeof(cl_uint), (void *)&splineOrder);
  }

  // Set output image to the kernel
  GPUDataManager::Pointer dummy;
  SetKernelWithITKImage<GPUOutputImage>(
    this->m_PostKernelManager, this->m_FilterPostGPUKernelHandle, argidx, output, dummy, true, false);

  // Set output image size to the kernel
  OpenCLKernelToImageBridge<OutputImageType>::SetSize(
    postKernel, argidx++, output->GetLargestPossibleRegion().GetSize());

  // Set the parameters struct to the kernel
  this->m_PostKernelManager->SetKernelArgWithImage(
    this->m_FilterPostGPUKernelHandle, argidx++, this->m_FilterParameters);

  // Set the image function to the kernel
  this->m_PostKernelManager->SetKernelArgWithImage(
    this->m_FilterPostGPUKernelHandle, argidx++, this->m_InterpolatorBase->GetParametersDataManager());

  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForPostKernelManager() finished");
} // end SetArgumentsForPostKernelManager()


/**
 * ***************** GetTransformType ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
auto
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  GetTransformType(const int & transformIndex) const -> const GPUTransformTypeEnum
{
  if (this->m_TransformIsCombo)
  {
    const CompositeTransformBaseType * compositeTransform =
      dynamic_cast<const CompositeTransformBaseType *>(this->m_TransformBase);

    if (compositeTransform->IsIdentityTransform(transformIndex))
    {
      return GPUResampleImageFilter::IdentityTransform;
    }
    else if (compositeTransform->IsMatrixOffsetTransform(transformIndex))
    {
      return GPUResampleImageFilter::MatrixOffsetTransform;
    }
    else if (compositeTransform->IsTranslationTransform(transformIndex))
    {
      return GPUResampleImageFilter::TranslationTransform;
    }
    else if (compositeTransform->IsBSplineTransform(transformIndex))
    {
      return GPUResampleImageFilter::BSplineTransform;
    }
  } // end if combo
  else
  {
    if (this->m_TransformBase->IsIdentityTransform())
    {
      return GPUResampleImageFilter::IdentityTransform;
    }
    else if (this->m_TransformBase->IsMatrixOffsetTransform())
    {
      return GPUResampleImageFilter::MatrixOffsetTransform;
    }
    else if (this->m_TransformBase->IsTranslationTransform())
    {
      return GPUResampleImageFilter::TranslationTransform;
    }
    else if (this->m_TransformBase->IsBSplineTransform())
    {
      return GPUResampleImageFilter::BSplineTransform;
    }
  }

  return GPUResampleImageFilter::Else;
} // end GetTransformType()


/**
 * ***************** HasTransform ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
bool
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::HasTransform(
  const GPUTransformTypeEnum type) const
{
  if (this->m_FilterLoopGPUKernelHandle.empty())
  {
    return false;
  }

  typename TransformsHandle::const_iterator it = this->m_FilterLoopGPUKernelHandle.find(type);
  if (it == this->m_FilterLoopGPUKernelHandle.end())
  {
    return false;
  }

  return it->second.second;
} // end HasTransform()


/**
 * ***************** GetTransformHandle ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
int
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  GetTransformHandle(const GPUTransformTypeEnum type) const
{
  if (this->m_FilterLoopGPUKernelHandle.empty())
  {
    return -1;
  }

  typename TransformsHandle::const_iterator it = this->m_FilterLoopGPUKernelHandle.find(type);
  if (it == this->m_FilterLoopGPUKernelHandle.end())
  {
    return -1;
  }

  return it->second.first;
} // end GetTransformHandle()


/**
 * ***************** GetKernelIdFromTransformId ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
bool
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  GetKernelIdFromTransformId(const std::size_t & transformIndex, std::size_t & kernelId) const
{
  if (this->m_TransformIsCombo)
  {
    const CompositeTransformBaseType * compositeTransform =
      dynamic_cast<const CompositeTransformBaseType *>(this->m_TransformBase);

    if (compositeTransform->IsIdentityTransform(transformIndex))
    {
      kernelId = this->GetTransformHandle(IdentityTransform);
      return true;
    }
    else if (compositeTransform->IsMatrixOffsetTransform(transformIndex))
    {
      kernelId = this->GetTransformHandle(MatrixOffsetTransform);
      return true;
    }
    else if (compositeTransform->IsTranslationTransform(transformIndex))
    {
      kernelId = this->GetTransformHandle(TranslationTransform);
      return true;
    }
    else if (compositeTransform->IsBSplineTransform(transformIndex))
    {
      kernelId = this->GetTransformHandle(BSplineTransform);
      return true;
    }
  } // end if combo
  else
  {
    if (this->HasTransform(IdentityTransform))
    {
      kernelId = this->GetTransformHandle(IdentityTransform);
      return true;
    }
    else if (this->HasTransform(MatrixOffsetTransform))
    {
      kernelId = this->GetTransformHandle(MatrixOffsetTransform);
      return true;
    }
    else if (this->HasTransform(TranslationTransform))
    {
      kernelId = this->GetTransformHandle(TranslationTransform);
      return true;
    }
    else if (this->HasTransform(BSplineTransform))
    {
      kernelId = this->GetTransformHandle(BSplineTransform);
      return true;
    }
  }

  return false;
} // end GetKernelIdFromTransformId()


/**
 * ***************** GetGPUBSplineBaseTransform ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
auto
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::
  GetGPUBSplineBaseTransform(const std::size_t transformIndex) -> GPUBSplineBaseTransformType *
{
  GPUBSplineBaseTransformType * GPUBSplineTransformBase = nullptr;

  // Get GPUBSplineTransformBase
  if (this->m_TransformIsCombo)
  {
    using CompositeTransformType = GPUCompositeTransformBase<InterpolatorPrecisionType, InputImageDimension>;
    CompositeTransformType * compositeTransform = dynamic_cast<CompositeTransformType *>(this->m_TransformBase);

    GPUBSplineTransformBase =
      dynamic_cast<GPUBSplineBaseTransformType *>(compositeTransform->GetNthTransform(transformIndex).GetPointer());
  }
  else
  {
    GPUBSplineTransformBase = dynamic_cast<GPUBSplineBaseTransformType *>(this->m_TransformBase);
  }

  if (!GPUBSplineTransformBase)
  {
    itkExceptionMacro(<< "Could not get coefficients from GPU BSpline transform.");
  }

  return GPUBSplineTransformBase;
} // end GetGPUBSplineBaseTransform()

/**
 * ***************** PrintSelf ***********************
 */

template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType,
          typename TTransformPrecisionType>
void
GPUResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::PrintSelf(
  std::ostream & os,
  Indent         indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
} // end PrintSelf()


} // end namespace itk

#endif /* itkGPUResampleImageFilter_hxx */
