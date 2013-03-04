/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUResampleImageFilter_hxx
#define __itkGPUResampleImageFilter_hxx

#include "itkGPUResampleImageFilter.h"
#include "itkGPUKernelManagerHelperFunctions.h"
#include "itkGPUMath.h"
#include "itkGPUImageBase.h"
#include "itkGPUCompositeTransformBase.h"
#include "itkGPUBSplineBaseTransform.h"
#include "itkGPUBSplineInterpolateImageFunction.h"

#include "itkImageLinearIteratorWithIndex.h"
#include "itkTimeProbe.h"
#include "itkImageRegionSplitter.h"

namespace
{
typedef struct {
  cl_int transform_linear;
  cl_int interpolator_is_bspline;
  cl_int transform_is_bspline; // has to be removed, but then this struct has to be align, see CL_ALIGNED()
  cl_float default_value;
  cl_float2 min_max;
  cl_float2 min_max_output;
  cl_float3 delta;
} FilterParameters;
//} CL_ALIGNED(16) FilterParameters;
} // end of unnamed namespace

namespace itk
{
//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::GPUResampleImageFilter()
{
  itkDebugMacro(<< "GPUResampleImageFilter constructor called");

  this->m_PreKernelManager  = GPUKernelManager::New();
  this->m_LoopKernelManager = GPUKernelManager::New();
  this->m_PostKernelManager = GPUKernelManager::New();

  this->m_InputGPUImageBase = GPUDataManager::New();
  this->m_OutputGPUImageBase = GPUDataManager::New();

  this->m_Parameters = GPUDataManager::New();
  this->m_Parameters->Initialize();
  this->m_Parameters->SetBufferFlag( CL_MEM_READ_ONLY );
  this->m_Parameters->SetBufferSize( sizeof( FilterParameters ) );
  this->m_Parameters->Allocate();

  m_DeformationFieldBuffer = GPUDataManager::New();

  this->m_InterpolatorSourceLoadedIndex = 0;
  this->m_TransformSourceLoadedIndex = 0;

  this->m_InterpolatorIsBSpline = false; // make it protected in base class
  this->m_TransformIsCombo = false;

  // Set all handlers to -1;
  this->m_FilterPreGPUKernelHandle = -1;
  this->m_FilterPostGPUKernelHandle = -1;

  this->m_InterpolatorBase = NULL;
  this->m_TransformBase = NULL;

  std::ostringstream defines;
  if ( TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1 )
  {
    itkExceptionMacro( "GPUResampleImageFilter supports 1/2/3D image." );
  }

  defines << "#define DIM_" << int(TInputImage::ImageDimension) << "\n";
  defines << "#define INPIXELTYPE ";
  GetTypenameInString( typeid( typename TInputImage::PixelType ), defines );
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString( typeid( typename TOutputImage::PixelType ), defines );
  defines << "#define INTERPOLATOR_PRECISION_TYPE ";
  GetTypenameInString( typeid( TInterpolatorPrecisionType ), defines );

  // Resize m_Sources
  const unsigned int numberOfIncludes = 4; // Defines, GPUMath, GPUImageBase,
                                           // GPUResampleImageFilter
  const unsigned int numberOfSources  = 2; // GPUInterpolator, GPUTransform,
                                           //
  m_Sources.resize( numberOfIncludes + numberOfSources );
  m_SourceIndex = 0;

  // Add defines
  m_Sources[m_SourceIndex++] = defines.str();

  // Get GPUMath source
  const std::string oclMathSource( GPUMathKernel::GetOpenCLSource() );
  m_Sources[m_SourceIndex++] = oclMathSource;

  // Get GPUImageBase source
  const std::string oclImageBaseSource( GPUImageBaseKernel::GetOpenCLSource() );
  m_Sources[m_SourceIndex++] = oclImageBaseSource;

  // Get GPUResampleImageFilter source
  const std::string oclResampleImageFilterSource( GPUResampleImageFilterKernel::GetOpenCLSource() );
  m_Sources[m_SourceIndex++] = oclResampleImageFilterSource;

  // Construct ResampleImageFilter Pre code
  std::ostringstream resamplePreSource;
  resamplePreSource << "#define RESAMPLE_PRE\n";
  resamplePreSource << m_Sources[1]; // GPUMath source
  resamplePreSource << m_Sources[2]; // GPUImageBase source
  resamplePreSource << m_Sources[3]; // GPUResampleImageFilter source

  // Load and create kernel
  const bool loaded =
    m_PreKernelManager->LoadProgramFromString( resamplePreSource.str().c_str(), defines.str().c_str() );
  if ( loaded )
  {
    this->m_FilterPreGPUKernelHandle = m_PreKernelManager->CreateKernel( "ResampleImageFilterPre" );
  }
  else
  {
    itkExceptionMacro( << "Kernel has not been loaded from string:\n" << resamplePreSource.str() );
  }

  itkDebugMacro(<< "GPUResampleImageFilter constructor finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::SetInterpolator( InterpolatorType *_arg )
{
  itkDebugMacro( "setting Interpolator to " << _arg );
  CPUSuperclass::SetInterpolator( _arg );

  const GPUInterpolatorBase *interpolatorBase =
    dynamic_cast< const GPUInterpolatorBase * >( _arg );

  if ( interpolatorBase )
  {
    this->m_InterpolatorBase = (GPUInterpolatorBase *)interpolatorBase;

    // Test for a GPU BSpline interpolator
    typedef GPUBSplineInterpolateImageFunction< InputImageType,
                                                TInterpolatorPrecisionType > GPUBSplineInterpolatorType;
    const GPUBSplineInterpolatorType *GPUBSplineInterpolator =
      dynamic_cast< const GPUBSplineInterpolatorType * >( _arg );
    if ( GPUBSplineInterpolator )
    {
      m_InterpolatorIsBSpline = true;
    }
    else
    {
      m_InterpolatorIsBSpline = false;
    }

    // Get transform source
    std::string source;
    if ( !interpolatorBase->GetSourceCode( source ) )
    {
      itkExceptionMacro( << "Unable to get interpolator source code." );
    }
    else
    {
      // Construct ResampleImageFilter Post code
      const std::string  defines = m_Sources[0];
      std::ostringstream resamplePostSource;
      resamplePostSource << "#define RESAMPLE_POST\n";
      resamplePostSource << m_Sources[1]; // GPUMath source
      resamplePostSource << m_Sources[2]; // GPUImageBase source
      resamplePostSource << source;
      resamplePostSource << m_Sources[3]; // GPUResampleImageFilter source

      // Load and create kernel
      const bool loaded =
        m_PostKernelManager->LoadProgramFromString( resamplePostSource.str().c_str(), defines.c_str() );
      if ( loaded )
      {
        if ( m_InterpolatorIsBSpline )
        {
          this->m_FilterPostGPUKernelHandle = m_PostKernelManager->CreateKernel(
            "ResampleImageFilterPost_InterpolatorBSpline" );
        }
        else
        {
          this->m_FilterPostGPUKernelHandle = m_PostKernelManager->CreateKernel( "ResampleImageFilterPost" );
        }
      }
      else
      {
        itkExceptionMacro( << "Kernel has not been loaded from string:\n" << resamplePostSource.str() );
      }
    }
  }
  else
  {
    itkExceptionMacro( "Setting unsupported GPU interpolator to " << _arg );
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetInterpolator() finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
  ::SetExtrapolator( ExtrapolatorType *_arg )
{
  itkDebugMacro( "setting Extrapolator to " << _arg );
  CPUSuperclass::SetExtrapolator( _arg );

  itkWarningMacro(<< "Setting Extrapolator for GPUResampleImageFilter not supported yet.");

  itkDebugMacro(<< "GPUResampleImageFilter::SetExtrapolator() finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::SetTransform( const TransformType *_arg )
{
  itkDebugMacro( "setting Transform to " << _arg );
  CPUSuperclass::SetTransform( _arg );

  const GPUTransformBase *transformBase =
    dynamic_cast< const GPUTransformBase * >( _arg );

  if ( transformBase )
  {
    this->m_TransformBase = (GPUTransformBase *)transformBase;

    // Test for a GPU Combo transform
    typedef GPUCompositeTransformBase< TInterpolatorPrecisionType,
                                       InputImageDimension > CompositeTransformType;
    const CompositeTransformType *compositeTransformBase =
      dynamic_cast< const CompositeTransformType * >( _arg );

    // Erase map of supported transforms
    m_FilterLoopGPUKernelHandle.clear();

    if ( compositeTransformBase )
    {
      m_TransformIsCombo = true;

      // Construct m_FilterLoopGPUKernelHandle
      TransformHandle identitytransform( -1, compositeTransformBase->HasIdentityTransform() );
      TransformHandle matrixoffsettransform( -1, compositeTransformBase->HasMatrixOffsetTransform() );
      TransformHandle translationtransform( -1, compositeTransformBase->HasTranslationTransform() );
      TransformHandle bsplinetransform( -1, compositeTransformBase->HasBSplineTransform() );

      m_FilterLoopGPUKernelHandle[IdentityTransform] = identitytransform;
      m_FilterLoopGPUKernelHandle[MatrixOffsetTransform] = matrixoffsettransform;
      m_FilterLoopGPUKernelHandle[TranslationTransform] = translationtransform;
      m_FilterLoopGPUKernelHandle[BSplineTransform] = bsplinetransform;
    }
    else
    {
      m_TransformIsCombo = false;

      // Construct m_FilterLoopGPUKernelHandle
      TransformHandle identitytransform( -1, transformBase->IsIdentityTransform() );
      TransformHandle matrixoffsettransform( -1, transformBase->IsMatrixOffsetTransform() );
      TransformHandle translationtransform( -1, transformBase->IsTranslationTransform() );
      TransformHandle bsplinetransform( -1, transformBase->IsBSplineTransform() );

      m_FilterLoopGPUKernelHandle[IdentityTransform] = identitytransform;
      m_FilterLoopGPUKernelHandle[MatrixOffsetTransform] = matrixoffsettransform;
      m_FilterLoopGPUKernelHandle[TranslationTransform] = translationtransform;
      m_FilterLoopGPUKernelHandle[BSplineTransform] = bsplinetransform;
    }

    // Get transform source
    std::string source;
    if ( !transformBase->GetSourceCode( source ) )
    {
      itkExceptionMacro( << "Unable to get transform source code." );
    }
    else
    {
      // Construct ResampleImageFilter Loop code
      const std::string  defines = m_Sources[0];
      std::ostringstream resampleLoopSource;
      resampleLoopSource << "#define RESAMPLE_LOOP\n";

      if ( HasTransform( IdentityTransform ) )
      {
        resampleLoopSource << "#define IDENTITY_TRANSFORM\n";
      }
      if ( HasTransform( MatrixOffsetTransform ) )
      {
        resampleLoopSource << "#define MATRIX_OFFSET_TRANSFORM\n";
      }
      if ( HasTransform( TranslationTransform ) )
      {
        resampleLoopSource << "#define TRANSLATION_TRANSFORM\n";
      }
      if ( HasTransform( BSplineTransform ) )
      {
        resampleLoopSource << "#define BSPLINE_TRANSFORM\n";
      }

      resampleLoopSource << m_Sources[1]; // GPUMath source
      resampleLoopSource << m_Sources[2]; // GPUImageBase source
      resampleLoopSource << source;
      resampleLoopSource << m_Sources[3]; // GPUResampleImageFilter source

      // Load and create kernel
      const bool loaded =
        m_LoopKernelManager->LoadProgramFromString( resampleLoopSource.str().c_str(), defines.c_str() );
      if ( loaded )
      {
        if ( HasTransform( IdentityTransform ) )
        {
          m_FilterLoopGPUKernelHandle[IdentityTransform].first =
            m_LoopKernelManager->CreateKernel( "ResampleImageFilterLoop_IdentityTransform" );
        }
        if ( HasTransform( MatrixOffsetTransform ) )
        {
          m_FilterLoopGPUKernelHandle[MatrixOffsetTransform].first =
            m_LoopKernelManager->CreateKernel( "ResampleImageFilterLoop_MatrixOffsetTransform" );
        }
        if ( HasTransform( TranslationTransform ) )
        {
          m_FilterLoopGPUKernelHandle[TranslationTransform].first =
            m_LoopKernelManager->CreateKernel( "ResampleImageFilterLoop_TranslationTransform" );
        }
        if ( HasTransform( BSplineTransform ) )
        {
          m_FilterLoopGPUKernelHandle[BSplineTransform].first =
            m_LoopKernelManager->CreateKernel( "ResampleImageFilterLoop_BSplineTransform" );
        }
      }
      else
      {
        itkExceptionMacro( << "Kernel has not been loaded from string:\n" << resampleLoopSource.str() );
      }
    }
  }
  else
  {
    itkExceptionMacro( "Setting unsupported GPU transform to " << _arg );
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetTransform() finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >::GPUGenerateData()
{
  itkDebugMacro(<< "GPUResampleImageFilter::GPUGenerateData() called");

  // Profiling
#ifdef OPENCL_PROFILING
  itk::TimeProbe gputimer;
  gputimer.Start();
#endif

  const typename GPUInputImage::Pointer inPtr = dynamic_cast< GPUInputImage * >( this->ProcessObject::GetInput( 0 ) );
  const typename GPUOutputImage::Pointer outPtr = dynamic_cast< GPUOutputImage * >( this->ProcessObject::GetOutput( 0 ) );

  // Perform the safe check
  if ( inPtr.IsNull() )
  {
    itkExceptionMacro( << "The GPU InputImage is NULL. Filter unable to perform." );
    return;
  }
  if ( outPtr.IsNull() )
  {
    itkExceptionMacro( << "The GPU OutputImage is NULL. Filter unable to perform." );
    return;
  }

  const typename GPUOutputImage::SizeType outSize = outPtr->GetLargestPossibleRegion().GetSize();
  const unsigned int ImageDim = (unsigned int)TInputImage::ImageDimension;

  // Define parameters
  FilterParameters parameters;
  // Set transform linear
  parameters.transform_linear = static_cast< int >( this->GetTransform()->IsLinear() );
  // Set interpolator is BSpline
  parameters.interpolator_is_bspline = static_cast< int >( this->m_InterpolatorIsBSpline );
  // Set defaultValue, minValue/maxValue, minOutputValue/maxOutputValue
  //typedef typename InterpolatorType::OutputType OutputType;
  parameters.default_value = static_cast< float >( this->GetDefaultPixelValue() );
  // Min/max values of the output pixel type AND these values
  // represented as the output type of the interpolator
  parameters.min_max.s[0] = static_cast< float >( NumericTraits< OutputImagePixelType >::NonpositiveMin() );
  parameters.min_max.s[1] = static_cast< float >( NumericTraits< OutputImagePixelType >::max() );
  parameters.min_max_output.s[0] = parameters.min_max.s[0];
  parameters.min_max_output.s[1] = parameters.min_max.s[1];

  // Calculate delta
  float delta[3];
  CalculateDelta( inPtr, outPtr, &delta[0] );
  for ( unsigned int i = 0; i < ImageDim; i++ )
  {
    parameters.delta.s[i] = delta[i];
  }

  this->m_Parameters->SetCPUBufferPointer( &parameters );
  this->m_Parameters->SetGPUDirtyFlag( true );
  this->m_Parameters->UpdateGPUBuffer();

  //
  unsigned int imgSize[3];
  imgSize[0] = imgSize[1] = imgSize[2] = 1;
  for ( unsigned int i = 0; i < ImageDim; i++ )
  {
    imgSize[i] = outSize[i];
  }

  std::size_t localSize[3], globalSize[3];
  localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize( ImageDim );

  for ( unsigned int i = 0; i < ImageDim; i++ )
  {
    // total # of threads
    globalSize[i] = localSize[i] * (unsigned int)ceil( (float)outSize[i] / (float)localSize[i] );
  }

  // For now let's fix it, later we could support user defined splits in manual or auto mode.
  // For auto definition, all GPU memory allocation within ITK OpenCL has to be tracked and
  // the number of splits has to be computed based on remaining global GPU memory.
  unsigned int requestedNumberOfSplits = 5;
  if( ImageDim < 3 )
    requestedNumberOfSplits = 1;

  typedef ImageRegionSplitter< TInputImage::ImageDimension > RegionSplitterType;
  typename RegionSplitterType::RegionType splitRegion;
  splitRegion.SetSize( outSize );
  typename RegionSplitterType::Pointer splitter = RegionSplitterType::New();
  const unsigned int numberOfSplits = splitter->GetNumberOfSplits( splitRegion, requestedNumberOfSplits );

  // maxSize
  typename RegionSplitterType::SizeType maxSize;
  maxSize.Fill( 0 );
  for ( unsigned int i = 0; i < numberOfSplits; i++ )
  {
    const typename RegionSplitterType::RegionType currentRegion = splitter->GetSplit( i, numberOfSplits, splitRegion );
    const typename RegionSplitterType::SizeType currentSize = currentRegion.GetSize();
    std::size_t cSize = 1, mSize = 1;
    for ( unsigned int i = 0; i < ImageDim; i++ )
    {
      cSize *= currentSize[i];
      mSize *= maxSize[i];
    }

    if ( cSize > mSize )
    {
      maxSize = currentSize;
    }
    //std::cout<< currentRegion;
  }

  //std::cout<< "ImageSize: " << splitRegion.GetSize() << std::endl;
  //std::cout<< "MaxSize for requested number of splits("<<
  //requestedNumberOfSplits <<"): " << maxSize << std::endl << std::endl;

  std::size_t sizeDF = 1;
  for ( unsigned int i = 0; i < ImageDim; i++ )
  {
    sizeDF *= maxSize[i];
  }

  // Create T
  unsigned int mem_size_DF = 0;
  switch ( ImageDim )
  {
  case 1: mem_size_DF = sizeDF * sizeof( cl_float ); break;
  case 2: mem_size_DF = sizeDF * sizeof( cl_float2 ); break;
  case 3: mem_size_DF = sizeDF * sizeof( cl_float3 ); break;
  }

  this->m_DeformationFieldBuffer->Initialize();
  this->m_DeformationFieldBuffer->SetBufferFlag( CL_MEM_READ_WRITE );
  this->m_DeformationFieldBuffer->SetBufferSize( mem_size_DF );
  this->m_DeformationFieldBuffer->Allocate();

  //std::cout<<" mem_size_T = " << mem_size_T << std::endl;

  // arguments set up
  cl_uint argidx = 0;

  cl_uint dtsizePreIntex = 0;
  SetArgumentsForPreKernelManager( outPtr, dtsizePreIntex );

  // Set arguments for loop kernel
  cl_uint dfsizeLoopIntex = 0, comboIndex = 0, transformIndex = 0;
  SetArgumentsForLoopKernelManager( inPtr, outPtr, dfsizeLoopIntex, comboIndex, transformIndex );
  if ( !m_TransformIsCombo )
  {
    SetTransformArgumentsForLoopKernelManager( 0, comboIndex, transformIndex );
  }

  // Set arguments for post kernel
  cl_uint dfsizePostIntex = 0;
  SetArgumentsForPostKernelManager( inPtr, outPtr, dfsizePostIntex );

  //
  std::size_t local3D[3], local2D[2], local1D;

#if ITK_USE_INTEL_OPENCL
  local3D[0] = local3D[1] = local3D[2] = 1;
  local2D[0] = local2D[1] = 2;
  local1D = 2;
#else
  local3D[0] = local3D[1] = local3D[2] = OpenCLGetLocalBlockSize( InputImageDimension );
  local2D[0] = local2D[1] = OpenCLGetLocalBlockSize( InputImageDimension );
  local1D = OpenCLGetLocalBlockSize( InputImageDimension );
#endif

  // Start
  OpenCLEventList    eventList;
  const unsigned int numDivisions = numberOfSplits; // numberOfSplits
  unsigned int       piece;

  cl_uint3    dfsize3D; cl_uint2 dfsize2D; cl_uint dfsize1D;
  std::size_t global3D[3], global2D[2], global1D;
  std::size_t offset3D[3], offset2D[2], offset1D;

  OpenCLSize global;
  OpenCLSize offset;

  for ( piece = 0;
        piece < numDivisions && !this->GetAbortGenerateData();
        piece++ )
  {
    const typename RegionSplitterType::RegionType currentRegion =
      splitter->GetSplit( piece, numDivisions, splitRegion );

    // define and set dfsize, global and offset
    switch ( ImageDim )
    {
      case 1:
      {
        dfsize1D = currentRegion.GetSize( 0 );
        global1D = local1D * (unsigned int)ceil( (float)dfsize1D / (float)local1D );
        offset1D = currentRegion.GetIndex( 0 );

        // set dfsize argument
        m_PreKernelManager->SetKernelArg( m_FilterPreGPUKernelHandle,
                                          dtsizePreIntex, sizeof( cl_uint ), (void *)&dfsize1D );

        if ( HasTransform( IdentityTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( IdentityTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint ), (void *)&dfsize1D );
        }
        if ( HasTransform( MatrixOffsetTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( MatrixOffsetTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint ), (void *)&dfsize1D );
        }
        if ( HasTransform( TranslationTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( TranslationTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint ), (void *)&dfsize1D );
        }
        if ( HasTransform( BSplineTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( BSplineTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint ), (void *)&dfsize1D );
        }

        m_PostKernelManager->SetKernelArg( m_FilterPostGPUKernelHandle,
                                           dfsizePostIntex, sizeof( cl_uint ), (void *)&dfsize1D );

        global = OpenCLSize( global1D );
        offset = OpenCLSize( offset1D );
      }
      break;
      case 2:
      {
        for ( unsigned int i = 0; i < 2; i++ )
        {
          dfsize2D.s[i] = currentRegion.GetSize( i );
          global2D[i] = local2D[i] * (unsigned int)ceil( (float)dfsize2D.s[i] / (float)local2D[i] );
          offset2D[i] = currentRegion.GetIndex( i );
        }

        // set dfsize argument
        m_PreKernelManager->SetKernelArg( m_FilterPreGPUKernelHandle,
                                          dtsizePreIntex, sizeof( cl_uint2 ), (void *)&dfsize2D );

        if ( HasTransform( IdentityTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( IdentityTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint2 ), (void *)&dfsize2D );
        }
        if ( HasTransform( MatrixOffsetTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( MatrixOffsetTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint2 ), (void *)&dfsize2D );
        }
        if ( HasTransform( TranslationTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( TranslationTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint2 ), (void *)&dfsize2D );
        }
        if ( HasTransform( BSplineTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( BSplineTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint2 ), (void *)&dfsize2D );
        }

        m_PostKernelManager->SetKernelArg( m_FilterPostGPUKernelHandle,
                                           dfsizePostIntex, sizeof( cl_uint2 ), (void *)&dfsize2D );

        global = OpenCLSize( global2D[0], global2D[1] );
        offset = OpenCLSize( offset2D[0], offset2D[1] );
      }
      break;
      case 3:
      {
        for ( unsigned int i = 0; i < 3; i++ )
        {
          dfsize3D.s[i] = currentRegion.GetSize( i );
          global3D[i] = local3D[i] * (unsigned int)ceil( (float)dfsize3D.s[i] / (float)local3D[i] );
          offset3D[i] = currentRegion.GetIndex( i );
        }
        dfsize3D.s[3] = 0;

        // set dfsize argument
        m_PreKernelManager->SetKernelArg( m_FilterPreGPUKernelHandle,
                                          dtsizePreIntex, sizeof( cl_uint3 ), (void *)&dfsize3D );

        if ( HasTransform( IdentityTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( IdentityTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint3 ), (void *)&dfsize3D );
        }
        if ( HasTransform( MatrixOffsetTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( MatrixOffsetTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint3 ), (void *)&dfsize3D );
        }
        if ( HasTransform( TranslationTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( TranslationTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint3 ), (void *)&dfsize3D );
        }
        if ( HasTransform( BSplineTransform ) )
        {
          m_LoopKernelManager->SetKernelArg( GetTransformHandle( BSplineTransform ),
                                             dfsizeLoopIntex, sizeof( cl_uint3 ), (void *)&dfsize3D );
        }

        m_PostKernelManager->SetKernelArg( m_FilterPostGPUKernelHandle,
                                           dfsizePostIntex, sizeof( cl_uint3 ), (void *)&dfsize3D );

        global = OpenCLSize( global3D[0], global3D[1], global3D[2] );
        offset = OpenCLSize( offset3D[0], offset3D[1], offset3D[2] );
      }
      break;
    }

    m_PreKernelManager->SetGlobalWorkSize( 0, global );
    m_PreKernelManager->SetGlobalWorkOffset( 0, offset );

    if ( HasTransform( IdentityTransform ) )
    {
      const int kernelId = GetTransformHandle( IdentityTransform );
      m_LoopKernelManager->SetGlobalWorkSize( kernelId, global );
      m_LoopKernelManager->SetGlobalWorkOffset( kernelId, offset );
    }
    if ( HasTransform( MatrixOffsetTransform ) )
    {
      const int kernelId = GetTransformHandle( MatrixOffsetTransform );
      m_LoopKernelManager->SetGlobalWorkSize( kernelId, global );
      m_LoopKernelManager->SetGlobalWorkOffset( kernelId, offset );
    }
    if ( HasTransform( TranslationTransform ) )
    {
      const int kernelId = GetTransformHandle( TranslationTransform );
      m_LoopKernelManager->SetGlobalWorkSize( kernelId, global );
      m_LoopKernelManager->SetGlobalWorkOffset( kernelId, offset );
    }
    if ( HasTransform( BSplineTransform ) )
    {
      const int kernelId = GetTransformHandle( BSplineTransform );
      m_LoopKernelManager->SetGlobalWorkSize( kernelId, global );
      m_LoopKernelManager->SetGlobalWorkOffset( kernelId, offset );
    }

    m_PostKernelManager->SetGlobalWorkSize( 0, global );
    m_PostKernelManager->SetGlobalWorkOffset( 0, offset );

    if ( eventList.GetSize() == 0 )
    {
      itkDebugMacro(<< "Calling m_PreKernelManager->LaunchKernel("
        << m_FilterPreGPUKernelHandle << ")");

      OpenCLEvent preEvent = m_PreKernelManager->LaunchKernel( m_FilterPreGPUKernelHandle );
      eventList.Append( preEvent );
    }
    else
    {
      itkDebugMacro(<< "Calling m_PreKernelManager->LaunchKernel("
        << m_FilterPreGPUKernelHandle << ",\n" << eventList << ")");

      OpenCLEvent preEvent = m_PreKernelManager->LaunchKernel( m_FilterPreGPUKernelHandle, eventList );
      eventList.Append( preEvent );
    }

    if ( m_TransformIsCombo )
    {
      typedef GPUCompositeTransformBase< TInterpolatorPrecisionType,
                                         InputImageDimension > CompositeTransformType;
      const CompositeTransformType *compositeTransform =
        dynamic_cast< const CompositeTransformType * >( m_TransformBase );

      const cl_uint withCombo = 1;
      const cl_uint withoutCombo = 0;
      if ( compositeTransform )
      {
        for ( int i = compositeTransform->GetNumberOfTransforms() - 1; i >= 0; i-- )
        {
          SetTransformArgumentsForLoopKernelManager( i, comboIndex, transformIndex );

          if ( compositeTransform->IsIdentityTransform( i ) )
          {
            if ( i != 0 )
            {
              m_LoopKernelManager->SetKernelArg( GetTransformHandle( IdentityTransform ),
                                                 comboIndex, sizeof( cl_uint ), (const void *)&withCombo );
            }
            else
            {
              m_LoopKernelManager->SetKernelArg( GetTransformHandle( IdentityTransform ),
                                                 comboIndex, sizeof( cl_uint ), (const void *)&withoutCombo );
            }

            itkDebugMacro(<< "Calling m_LoopKernelManager->LaunchKernel("
              << GetTransformHandle( IdentityTransform ) << ",\n" << eventList << ")");

            OpenCLEvent loopEvent =
              m_LoopKernelManager->LaunchKernel( GetTransformHandle( IdentityTransform ), eventList );
            eventList.Append( loopEvent );
          }
          else if ( compositeTransform->IsMatrixOffsetTransform( i ) )
          {
            if ( i != 0 )
            {
              m_LoopKernelManager->SetKernelArg( GetTransformHandle( MatrixOffsetTransform ),
                                                 comboIndex, sizeof( cl_uint ), (const void *)&withCombo );
            }
            else
            {
              m_LoopKernelManager->SetKernelArg( GetTransformHandle( MatrixOffsetTransform ),
                                                 comboIndex, sizeof( cl_uint ), (const void *)&withoutCombo );
            }

            itkDebugMacro(<< "Calling m_LoopKernelManager->LaunchKernel("
              << GetTransformHandle( MatrixOffsetTransform ) << ",\n" << eventList << ")");

            OpenCLEvent loopEvent =
              m_LoopKernelManager->LaunchKernel( GetTransformHandle( MatrixOffsetTransform ), eventList );
            eventList.Append( loopEvent );
          }
          else if ( compositeTransform->IsTranslationTransform( i ) )
          {
            if ( i != 0 )
            {
              m_LoopKernelManager->SetKernelArg( GetTransformHandle( TranslationTransform ),
                                                 comboIndex, sizeof( cl_uint ), (const void *)&withCombo );
            }
            else
            {
              m_LoopKernelManager->SetKernelArg( GetTransformHandle( TranslationTransform ),
                                                 comboIndex, sizeof( cl_uint ), (const void *)&withoutCombo );
            }

            itkDebugMacro(<< "Calling m_LoopKernelManager->LaunchKernel("
              << GetTransformHandle( TranslationTransform ) << ",\n" << eventList << ")");

            OpenCLEvent loopEvent =
              m_LoopKernelManager->LaunchKernel( GetTransformHandle( TranslationTransform ), eventList );
            eventList.Append( loopEvent );
          }
          else if ( compositeTransform->IsBSplineTransform( i ) )
          {
            if ( i != 0 )
            {
              m_LoopKernelManager->SetKernelArg( GetTransformHandle( BSplineTransform ),
                                                 comboIndex, sizeof( cl_uint ), (const void *)&withCombo );
            }
            else
            {
              m_LoopKernelManager->SetKernelArg( GetTransformHandle( BSplineTransform ),
                                                 comboIndex, sizeof( cl_uint ), (const void *)&withoutCombo );
            }

            itkDebugMacro(<< "Calling m_LoopKernelManager->LaunchKernel("
              << GetTransformHandle( BSplineTransform ) << ",\n" << eventList << ")");

            OpenCLEvent loopEvent =
              m_LoopKernelManager->LaunchKernel( GetTransformHandle( BSplineTransform ), eventList );
            eventList.Append( loopEvent );
          }
        }
      }
      else
      {
        itkExceptionMacro( << "Could not get GPU composite transform." );
      }
    }
    else
    {
      if ( HasTransform( IdentityTransform ) )
      {
        itkDebugMacro(<< "Calling m_LoopKernelManager->LaunchKernel("
          << GetTransformHandle( IdentityTransform ) << ",\n" << eventList << ")");

        OpenCLEvent loopEvent =
          m_LoopKernelManager->LaunchKernel( GetTransformHandle( IdentityTransform ), eventList );
        eventList.Append( loopEvent );
      }
      else if ( HasTransform( MatrixOffsetTransform ) )
      {
        itkDebugMacro(<< "Calling m_LoopKernelManager->LaunchKernel("
          << GetTransformHandle( MatrixOffsetTransform ) << ",\n" << eventList << ")");

        OpenCLEvent loopEvent =
          m_LoopKernelManager->LaunchKernel( GetTransformHandle( MatrixOffsetTransform ), eventList );
        eventList.Append( loopEvent );
      }
      else if ( HasTransform( TranslationTransform ) )
      {
        itkDebugMacro(<< "Calling m_LoopKernelManager->LaunchKernel("
          << GetTransformHandle( TranslationTransform ) << ",\n" << eventList << ")");

        OpenCLEvent loopEvent =
          m_LoopKernelManager->LaunchKernel( GetTransformHandle( TranslationTransform ), eventList );
        eventList.Append( loopEvent );
      }
      else if ( HasTransform( BSplineTransform ) )
      {
        itkDebugMacro(<< "Calling m_LoopKernelManager->LaunchKernel("
          << GetTransformHandle( BSplineTransform ) << ",\n" << eventList << ")");

        OpenCLEvent loopEvent =
          m_LoopKernelManager->LaunchKernel( GetTransformHandle( BSplineTransform ), eventList );
        eventList.Append( loopEvent );
      }
    }

    itkDebugMacro(<< "Calling m_PostKernelManager->LaunchKernel("
      << m_FilterPostGPUKernelHandle << ",\n" << eventList << ")");

    OpenCLEvent postEvent = m_PostKernelManager->LaunchKernel( m_FilterPostGPUKernelHandle, eventList );
    eventList.Append( postEvent );
  }

  eventList.WaitForFinished();

  itkDebugMacro(<< "GPUResampleImageFilter::GPUGenerateData() finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::SetArgumentsForPreKernelManager( const typename GPUOutputImage::Pointer & output,
                                   cl_uint & index )
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForPreKernelManager("<< index <<") called");

  cl_uint argidx = 0;
  itk::SetKernelWithITKImage< GPUOutputImage >( m_PreKernelManager,
                                                m_FilterPreGPUKernelHandle,
                                                argidx,
                                                output,
                                                m_OutputGPUImageBase,
                                                true );

  m_PreKernelManager->SetKernelArgWithImage( m_FilterPreGPUKernelHandle, argidx++,
                                             this->m_DeformationFieldBuffer );
  index = argidx;

  //// dummy null
  //cl_uint3 dummyNull;
  //for(unsigned int i=0; i<4; i++)
  //{
  //  dummyNull.s[i] = 0;
  //}
  //m_PreKernelManager->SetKernelArg( m_FilterPreGPUKernelHandle, index, sizeof( cl_uint3 ), (void *)&dummyNull );

  argidx++;
  m_PreKernelManager->SetKernelArgWithImage( m_FilterPreGPUKernelHandle, argidx++,
                                             this->m_Parameters );

  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForPreKernelManager() finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::SetArgumentsForLoopKernelManager( const typename GPUInputImage::Pointer & input,
                                    const typename GPUOutputImage::Pointer & output,
                                    cl_uint & tsizeLoopIntex,
                                    cl_uint & comboIntex,
                                    cl_uint & transformIndex )
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForLoopKernelManager("
    << input->GetNameOfClass() << ", " << output->GetNameOfClass() << ", "
    << tsizeLoopIntex << ", " << comboIntex << ", " << transformIndex << ") called");

  const cl_uint combo_default = 0;

  if ( HasTransform( IdentityTransform ) )
  {
    cl_uint   argidx = 0;
    const int handleId = GetTransformHandle( IdentityTransform );

    // Set input image to the kernel
    itk::SetKernelWithITKImage< GPUInputImage >( m_LoopKernelManager,
                                                 handleId,
                                                 argidx,
                                                 input,
                                                 m_InputGPUImageBase,
                                                 true );

    // Set output image to the kernel
    itk::SetKernelWithITKImage< GPUOutputImage >( m_LoopKernelManager,
                                                  handleId,
                                                  argidx,
                                                  output,
                                                  m_OutputGPUImageBase,
                                                  true );

    m_LoopKernelManager->SetKernelArgWithImage( handleId, argidx++, this->m_DeformationFieldBuffer );
    tsizeLoopIntex = argidx;
    argidx++;
    comboIntex = argidx;
    m_LoopKernelManager->SetKernelArg( handleId, comboIntex, sizeof( cl_uint ), (const void *)&combo_default );
    argidx++;
    m_LoopKernelManager->SetKernelArgWithImage( handleId, argidx++, this->m_Parameters );
    transformIndex = argidx;
  }

  if ( HasTransform( MatrixOffsetTransform ) )
  {
    cl_uint   argidx = 0;
    const int handleId = GetTransformHandle( MatrixOffsetTransform );

    // Set input image to the kernel
    itk::SetKernelWithITKImage< GPUInputImage >( m_LoopKernelManager,
                                                 handleId,
                                                 argidx,
                                                 input,
                                                 m_InputGPUImageBase,
                                                 true );

    // Set output image to the kernel
    itk::SetKernelWithITKImage< GPUOutputImage >( m_LoopKernelManager,
                                                  handleId,
                                                  argidx,
                                                  output,
                                                  m_OutputGPUImageBase,
                                                  true );

    m_LoopKernelManager->SetKernelArgWithImage( handleId, argidx++, this->m_DeformationFieldBuffer );
    tsizeLoopIntex = argidx;
    argidx++;
    comboIntex = argidx;
    m_LoopKernelManager->SetKernelArg( handleId, comboIntex, sizeof( cl_uint ), (const void *)&combo_default );
    argidx++;
    m_LoopKernelManager->SetKernelArgWithImage( handleId, argidx++, this->m_Parameters );
    transformIndex = argidx;
  }

  if ( HasTransform( TranslationTransform ) )
  {
    cl_uint   argidx = 0;
    const int handleId = GetTransformHandle( TranslationTransform );

    // Set input image to the kernel
    itk::SetKernelWithITKImage< GPUInputImage >( m_LoopKernelManager,
                                                 handleId,
                                                 argidx,
                                                 input,
                                                 m_InputGPUImageBase,
                                                 true );

    // Set output image to the kernel
    itk::SetKernelWithITKImage< GPUOutputImage >( m_LoopKernelManager,
                                                  handleId,
                                                  argidx,
                                                  output,
                                                  m_OutputGPUImageBase,
                                                  true );

    m_LoopKernelManager->SetKernelArgWithImage( handleId, argidx++, this->m_DeformationFieldBuffer );
    tsizeLoopIntex = argidx;
    argidx++;
    comboIntex = argidx;
    m_LoopKernelManager->SetKernelArg( handleId, comboIntex, sizeof( cl_uint ), (const void *)&combo_default );
    argidx++;
    m_LoopKernelManager->SetKernelArgWithImage( handleId, argidx++, this->m_Parameters );
    transformIndex = argidx;
  }

  if ( HasTransform( BSplineTransform ) )
  {
    cl_uint   argidx = 0;
    const int handleId = GetTransformHandle( BSplineTransform );

    // Set input image to the kernel
    itk::SetKernelWithITKImage< GPUInputImage >( m_LoopKernelManager,
                                                 handleId,
                                                 argidx,
                                                 input,
                                                 m_InputGPUImageBase,
                                                 true );

    // Set output image to the kernel
    itk::SetKernelWithITKImage< GPUOutputImage >( m_LoopKernelManager,
                                                  handleId,
                                                  argidx,
                                                  output,
                                                  m_OutputGPUImageBase,
                                                  true );

    m_LoopKernelManager->SetKernelArgWithImage( handleId, argidx++, this->m_DeformationFieldBuffer );
    tsizeLoopIntex = argidx;
    argidx++;
    comboIntex = argidx;
    m_LoopKernelManager->SetKernelArg( handleId, comboIntex, sizeof( cl_uint ), (const void *)&combo_default );
    argidx++;
    m_LoopKernelManager->SetKernelArgWithImage( handleId, argidx++, this->m_Parameters );
    transformIndex = argidx;
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForLoopKernelManager() finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::SetTransformArgumentsForLoopKernelManager( const std::size_t index,
                                             const cl_uint comboIndex,
                                             const cl_uint transformIndex )
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetTransformArgumentsForLoopKernelManager("
    << index << ", " << comboIndex << ", " << transformIndex << ") called");

  cl_uint argidx = transformIndex;

  if ( !m_TransformIsCombo )
  {
    if ( m_TransformBase->IsMatrixOffsetTransform() )
    {
      m_LoopKernelManager->SetKernelArgWithImage( GetTransformHandle( MatrixOffsetTransform ), argidx++,
                                                  this->m_TransformBase->GetParametersDataManager() );
    }
    else if ( m_TransformBase->IsTranslationTransform() )
    {
      m_LoopKernelManager->SetKernelArgWithImage( GetTransformHandle( TranslationTransform ), argidx++,
                                                  this->m_TransformBase->GetParametersDataManager() );
    }
    else if ( m_TransformBase->IsBSplineTransform() )
    {
      SetGPUCoefficients( 0, transformIndex );
    }
  }
  else
  {
    typedef GPUCompositeTransformBase< TInterpolatorPrecisionType,
                                       InputImageDimension > CompositeTransformType;
    const CompositeTransformType *compositeTransform =
      dynamic_cast< const CompositeTransformType * >( m_TransformBase );

    if ( compositeTransform )
    {
      if ( compositeTransform->IsMatrixOffsetTransform( index ) )
      {
        m_LoopKernelManager->SetKernelArgWithImage( GetTransformHandle( MatrixOffsetTransform ), argidx++,
                                                    this->m_TransformBase->GetParametersDataManager( index ) );
      }
      else if ( compositeTransform->IsTranslationTransform( index ) )
      {
        m_LoopKernelManager->SetKernelArgWithImage( GetTransformHandle( TranslationTransform ), argidx++,
                                                    this->m_TransformBase->GetParametersDataManager( index ) );
      }
      else if ( compositeTransform->IsBSplineTransform( index ) )
      {
        SetGPUCoefficients( index, transformIndex );
      }
    }
    else
    {
      itkExceptionMacro( << "Could not get GPU composite transform." );
    }
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetTransformArgumentsForLoopKernelManager() finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::SetGPUCoefficients( const std::size_t index, const cl_uint transformindex )
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetGPUCoefficients("
    << index << ", " << transformindex << ") called");

  // Typedefs
  typedef GPUBSplineBaseTransform< TInterpolatorPrecisionType,
                                   InputImageDimension > GPUBSplineTransformType;

  typedef typename GPUBSplineTransformType::GPUCoefficientImageType      GPUCoefficientImageType;
  typedef typename GPUBSplineTransformType::GPUCoefficientImageArray     GPUCoefficientImageArray;
  typedef typename GPUBSplineTransformType::GPUCoefficientImageBaseArray GPUCoefficientImageBaseArray;
  typedef typename GPUBSplineTransformType::GPUCoefficientImagePointer   GPUCoefficientImagePointer;
  typedef typename GPUBSplineTransformType::GPUDataManagerPointer        GPUDataManagerPointer;

  // Local variables
  cl_uint                  argidx = transformindex;
  GPUBSplineTransformType *GPUBSplineTransformBase = NULL;

  if ( m_TransformIsCombo )
  {
    typedef GPUCompositeTransformBase< TInterpolatorPrecisionType,
                                       InputImageDimension > CompositeTransformType;
    CompositeTransformType *compositeTransform =
      dynamic_cast< CompositeTransformType * >( m_TransformBase );

    if ( compositeTransform )
    {
      GPUBSplineTransformBase =
        dynamic_cast< GPUBSplineTransformType * >(
          compositeTransform->GetNthTransform( index ).GetPointer() );
    }
    else
    {
      itkExceptionMacro( << "Could not get GPU composite transform." );
    }
  }
  else
  {
    GPUBSplineTransformBase =
      dynamic_cast< GPUBSplineTransformType * >( m_TransformBase );
  }

  if ( GPUBSplineTransformBase )
  {
    GPUCoefficientImageArray GPUCoefficientImages =
      GPUBSplineTransformBase->GetGPUCoefficientImages();
    GPUCoefficientImageBaseArray GPUCoefficientImagesBases =
      GPUBSplineTransformBase->GetGPUCoefficientImagesBases();

    for ( unsigned int i = 0; i < InputImageDimension; i++ )
    {
      GPUCoefficientImagePointer coefficient = GPUCoefficientImages[i];
      GPUDataManagerPointer      coefficientbase = GPUCoefficientImagesBases[i];

      itk::SetKernelWithITKImage< GPUCoefficientImageType >(
        m_LoopKernelManager,
        GetTransformHandle( BSplineTransform ), argidx, coefficient, coefficientbase );
    }
  }
  else
  {
    itkExceptionMacro( << "Could not get coefficients from GPU BSpline transform." );
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetGPUCoefficients() finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::SetArgumentsForPostKernelManager( const typename GPUOutputImage::Pointer & input,
                                    const typename GPUOutputImage::Pointer & output,
                                    cl_uint & index )
{
  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForPostKernelManager("
    << input->GetNameOfClass() << ", " << output->GetNameOfClass() << ", " << index << ") called");

  cl_uint argidx = 0;

  // Set input image to the kernel
  itk::SetKernelWithITKImage< GPUInputImage >( m_PostKernelManager,
                                               m_FilterPostGPUKernelHandle,
                                               argidx,
                                               input,
                                               m_InputGPUImageBase );

  // Set output image to the kernel
  itk::SetKernelWithITKImage< GPUOutputImage >( m_PostKernelManager,
                                                m_FilterPostGPUKernelHandle,
                                                argidx,
                                                output,
                                                m_OutputGPUImageBase );

  m_PostKernelManager->SetKernelArgWithImage( m_FilterPostGPUKernelHandle, argidx++, this->m_DeformationFieldBuffer );
  index = argidx;
  argidx++;
  m_PostKernelManager->SetKernelArgWithImage( m_FilterPostGPUKernelHandle, argidx++, this->m_Parameters );

  // Set image function
  m_PostKernelManager->SetKernelArgWithImage( m_FilterPostGPUKernelHandle, argidx++,
                                              this->m_InterpolatorBase->GetParametersDataManager() );

  if ( m_InterpolatorIsBSpline )
  {
    typedef GPUBSplineInterpolateImageFunction< InputImageType,
                                                TInterpolatorPrecisionType > GPUBSplineInterpolatorType;
    typedef typename GPUBSplineInterpolatorType::GPUCoefficientImageType    GPUCoefficientImageType;
    typedef typename GPUBSplineInterpolatorType::GPUCoefficientImagePointer GPUCoefficientImagePointer;
    typedef typename GPUBSplineInterpolatorType::GPUDataManagerPointer      GPUDataManagerPointer;

    const GPUBSplineInterpolatorType *GPUBSplineInterpolator =
      dynamic_cast< const GPUBSplineInterpolatorType * >( m_InterpolatorBase );

    if ( GPUBSplineInterpolator )
    {
      GPUCoefficientImagePointer coefficient =
        GPUBSplineInterpolator->GetGPUCoefficients();
      GPUDataManagerPointer coefficientbase =
        GPUBSplineInterpolator->GetGPUCoefficientsImageBase();

      SetKernelWithITKImage< GPUCoefficientImageType >(
        m_PostKernelManager,
        m_FilterPostGPUKernelHandle, argidx, coefficient, coefficientbase );
    }
    else
    {
      itkExceptionMacro( << "Could not get coefficients from GPU BSpline interpolator." );
    }
  }

  itkDebugMacro(<< "GPUResampleImageFilter::SetArgumentsForPostKernelManager() finished");
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
bool GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::HasTransform( const GPUInputTransformType type )
{
  if ( m_FilterLoopGPUKernelHandle.size() == 0 )
  {
    return false;
  }

  typename TransformsHandle::iterator it = m_FilterLoopGPUKernelHandle.find( type );
  if ( it == m_FilterLoopGPUKernelHandle.end() )
  {
    return false;
  }
  else
  {
    return it->second.second;
  }
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
int GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::GetTransformHandle( const GPUInputTransformType type )
{
  if ( m_FilterLoopGPUKernelHandle.size() == 0 )
  {
    return -1;
  }

  typename TransformsHandle::iterator it = m_FilterLoopGPUKernelHandle.find( type );
  if ( it == m_FilterLoopGPUKernelHandle.end() )
  {
    return -1;
  }
  else
  {
    return it->second.first;
  }
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::CalculateDelta( const typename GPUInputImage::Pointer & _inputPtr,
                  const typename GPUOutputImage::Pointer & _outputPtr,
                  float *_delta )
{
  if ( !this->GetTransform()->IsLinear() )
  {
    for ( unsigned int i = 0; i < OutputImageType::ImageDimension; i++ )
    {
      _delta[i] = 0.0;
    }
    return;
  }

  // Create an iterator that will walk the output region for this thread.
  typedef ImageLinearIteratorWithIndex< TOutputImage > OutputIterator;
  OutputIterator outIt( _outputPtr, _outputPtr->GetLargestPossibleRegion() );
  outIt.SetDirection( 0 );

  // Define a few indices that will be used to translate from an input pixel
  // to an output pixel
  typedef Point< float, OutputImageType::ImageDimension >           FloatPointType;
  typedef ContinuousIndex< float, OutputImageType::ImageDimension > FloatContinuousInputIndexType;
  FloatPointType outputPoint;         // Coordinates of current output pixel
  FloatPointType inputPoint;          // Coordinates of current input pixel
  FloatPointType tmpOutputPoint;
  FloatPointType tmpInputPoint;

  FloatContinuousInputIndexType inputIndex;
  FloatContinuousInputIndexType tmpInputIndex;

  //typedef typename PointType::VectorType VectorType;
  typedef Vector< float, OutputImageType::ImageDimension > FloatVectorType;
  FloatVectorType delta;          // delta in input continuous index coordinate
                                  // frame

  IndexType index;

  // Determine the position of the first pixel in the scanline
  index = outIt.GetIndex();
  _outputPtr->TransformIndexToPhysicalPoint( index, outputPoint );

  // Compute corresponding input pixel position
  inputPoint = this->GetTransform()->TransformPoint( outputPoint );
  _inputPtr->TransformPhysicalPointToContinuousIndex( inputPoint, inputIndex );

  // As we walk across a scan line in the output image, we trace
  // an oriented/scaled/translated line in the input image.  Cache
  // the delta along this line in continuous index space of the input
  // image. This allows us to use vector addition to model the
  // transformation.
  //
  // To find delta, we take two pixels adjacent in a scanline
  // and determine the continuous indices of these pixels when
  // mapped to the input coordinate frame. We use the difference
  // between these two continuous indices as the delta to apply
  // to an index to trace line in the input image as we move
  // across the scanline of the output image.
  //
  // We determine delta in this manner so that Images
  // are both handled properly (taking into account the direction cosines).
  //

  ++index[0];
  _outputPtr->TransformIndexToPhysicalPoint( index, tmpOutputPoint );
  tmpInputPoint = this->GetTransform()->TransformPoint( tmpOutputPoint );
  _inputPtr->TransformPhysicalPointToContinuousIndex( tmpInputPoint, tmpInputIndex );

  delta = tmpInputIndex - inputIndex;

  for ( unsigned int i = 0; i < OutputImageType::ImageDimension; i++ )
  {
    _delta[i] = delta[i];
  }
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage, class TInterpolatorPrecisionType >
void GPUResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
  GPUSuperclass::PrintSelf( os, indent );
}
} // end namespace itk

#endif /* __itkGPUResampleImageFilter_hxx */
