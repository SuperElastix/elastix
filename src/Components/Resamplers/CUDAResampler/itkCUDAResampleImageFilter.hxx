/* $Id$ */
#if !defined(ITKCUDARESAMPLEFILTER_HXX)
#define ITKCUDARESAMPLEFILTER_HXX
#include <cuda_runtime.h>

#include "itkCUDAResampleImageFilter.h"

namespace itk
{
	template <typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType>
	itkCUDAResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType>
		::itkCUDAResampleImageFilter()
		: m_UseCuda(true)
		, m_PreFilter(false)
		, m_InternalCUDATransform(false)
	{
	}

	template <typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType>
	itkCUDAResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType>
		::~itkCUDAResampleImageFilter()
	{
		if (m_UseCuda) m_cuda.cudaUnInit();
	}

	template <typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType>
	void
		itkCUDAResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType>
		::copyParameters()
	{
		/* copy parameters to the GPU memory space */
		const typename SizeType        ITKoutputsize    = GetSize();
		const typename SizeType        ITKinputsize     = GetInput()->GetLargestPossibleRegion().GetSize();
		const typename SpacingType     ITKoutputSpacing = GetOutputSpacing();
		const typename OriginPointType ITKoutputOrigin  = GetOutputOrigin();
		const typename SpacingType     ITKinputSpacing  = GetInput()->GetSpacing();
		const typename OriginPointType ITKinputOrigin   = GetInput()->GetOrigin();

		int3 inputsize             = make_int3(ITKinputsize[0],  ITKinputsize[1],  ITKinputsize[2]);
		int3 outputsize            = make_int3(ITKoutputsize[0], ITKoutputsize[1], ITKoutputsize[2]);
		const InputPixelType* data = GetInput()->GetBufferPointer();
		m_cuda.cudaMallocImageData(inputsize, outputsize, data, m_PreFilter);

		float3 outputimageSpacing = make_float3(ITKoutputSpacing[0], ITKoutputSpacing[1], ITKoutputSpacing[2]);
		float3 outputimageOrigin  = make_float3(ITKoutputOrigin[0],  ITKoutputOrigin[1],  ITKoutputOrigin[2]);
		float3 inputimageSpacing  = make_float3(ITKinputSpacing[0],  ITKinputSpacing[1],  ITKinputSpacing[2]);
		float3 inputimageOrigin   = make_float3(ITKinputOrigin[0],   ITKinputOrigin[1],   ITKinputOrigin[2]);

		float defaultPixelValue   = GetDefaultPixelValue();
		m_cuda.cudaCopyImageSymbols(inputimageSpacing, inputimageOrigin, outputimageSpacing, outputimageOrigin, defaultPixelValue);

		const typename InternalBSplineTransformType::OriginType  ITKgridOrigin  = m_InternalCUDATransform->GetGridOrigin();
		const typename InternalBSplineTransformType::SpacingType ITKgridSpacing = m_InternalCUDATransform->GetGridSpacing();
		const typename InternalBSplineTransformType::SizeType    ITKgridSize    = m_InternalCUDATransform->GetGridRegion().GetSize();

		float3 gridSpacing        = make_float3(ITKgridSpacing[0],   ITKgridSpacing[1],   ITKgridSpacing[2]);
		float3 gridOrigin         = make_float3(ITKgridOrigin[0],    ITKgridOrigin[1],    ITKgridOrigin[2]);
		int3   gridSize           = make_int3  (ITKgridSize[0],      ITKgridSize[1],      ITKgridSize[2]);
		m_cuda.cudaCopyGridSymbols(gridSpacing, gridOrigin, gridSize);

		const InternalBSplineTransformType::ParametersType params
      = m_InternalCUDATransform->GetParameters();

		m_cuda.cudaMallocTransformationData( gridSize, params.data_block() );
	}

	template <typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType>
	void
		itkCUDAResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType>
		::GenerateData( void )
	{
    /** If we are not using CUDA simply use the CPU implementation. */
		if ( !m_UseCuda )
    {
      return Superclass::GenerateData();
    }

		try
		{
      /** First check if the Transform is valid for CUDA. */
      InternalBSplineTransformType * testPtr1
        = const_cast<InternalBSplineTransformType *>(
        dynamic_cast<const InternalBSplineTransformType *>( this->GetTransform() ) );
      InternalComboTransformType * testPtr2a
        = const_cast<InternalComboTransformType *>(
        dynamic_cast<const InternalComboTransformType *>( this->GetTransform() ) );

      bool transformIsValid = false;
      if ( testPtr1 )
      {
        /** The transform is of type AdvancedBSplineDeformableTransform. */
        transformIsValid = true;
        m_InternalCUDATransform = testPtr1;
      }
      else if ( testPtr2a )
      {
        // check that the comboT has no initial transform and that current = b-spline
        // and that b-spline = 3rd order

        /** The transform is of type AdvancedCombinationTransform. */
        if ( !testPtr2a->GetInitialTransform() )
        {
          InternalBSplineTransformType * testPtr2b
            = dynamic_cast<InternalBSplineTransformType *>(
            testPtr2a->GetCurrentTransform() );
          if ( testPtr2b )
          {
            /** The current transform is of type AdvancedBSplineDeformableTransform. */
            transformIsValid = true;
            m_InternalCUDATransform = testPtr2b;
          }
        }
      }

      if ( !transformIsValid )
      {
        itkWarningMacro( << "Using CPU (no B-spline transform set)" );
      }

      /** Check if proper CUDA device. */
			bool cuda_device = (Cudaclass::checkExecutionParameters() == 0);
			if ( !cuda_device )
      {
        itkWarningMacro( << "Using CPU (no CUDA capable GPU found, and/or update driver)" );
      }

      m_UseCuda = m_InternalCUDATransform.IsNotNull() && cuda_device;
		}
		catch ( itk::ExceptionObject & excep )
		{
      // FIXME: no printing
			std::cerr << excep << std::endl;
			m_UseCuda = false;
		}

		if ( !m_UseCuda )
    {
      return Superclass::GenerateData();
    }

		/** Initialize CUDA device. */
		m_cuda.cudaInit();

		/** Copy the parameters to the GPU. */
		copyParameters();

		/** Allocate host memory for the output and copy/cast the result back to the host. */
		AllocateOutputs();
		InputPixelType* data = GetOutput()->GetBufferPointer();

		/** Run the resampler. */
		m_cuda.GenerateData( data );

	} // end GenerateData()


}; /* namespace itk */

#endif /* ITKCUDARESAMPLEFILTER_HXX */
