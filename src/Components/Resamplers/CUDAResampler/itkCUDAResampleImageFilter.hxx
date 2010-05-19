/* $Id$ */
#pragma once
#include <cuda_runtime.h>

#include "itkCUDAResampleImageFilter.h"

namespace itk
{
	template <typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType>
	itkCUDAResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType>
		::itkCUDAResampleImageFilter()
		: m_UseCuda(true)
		, m_PreFilter(false)
		, m_Transform(false)
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

		const typename BSplineTransformType::OriginType  ITKgridOrigin  = m_Transform->GetGridOrigin();
		const typename BSplineTransformType::SpacingType ITKgridSpacing = m_Transform->GetGridSpacing();
		const typename BSplineTransformType::SizeType    ITKgridSize    = m_Transform->GetGridRegion().GetSize();

		float3 gridSpacing        = make_float3(ITKgridSpacing[0],   ITKgridSpacing[1],   ITKgridSpacing[2]);
		float3 gridOrigin         = make_float3(ITKgridOrigin[0],    ITKgridOrigin[1],    ITKgridOrigin[2]);
		int3   gridSize           = make_int3  (ITKgridSize[0],      ITKgridSize[1],      ITKgridSize[2]);
		m_cuda.cudaCopyGridSymbols(gridSpacing, gridOrigin, gridSize);

		const BSplineTransformType::ParametersType params = m_Transform->GetParameters();

		m_cuda.cudaMallocTransformationData(gridSize, params.data_block());
	}

	template <typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType>
	void
		itkCUDAResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType>
		::GenerateData()
	{
		if (!m_UseCuda) return Superclass::GenerateData();

		try
		{
			m_Transform = dynamic_cast<const BSplineTransformType*>(GetTransform());
			if (m_Transform == NULL) itkWarningMacro("using CPU (no B-spline transform set)");
			bool cuda_device = (Cudaclass::checkExecutionParameters() == 0);
			if (!cuda_device) itkWarningMacro("using CPU (no CUDA capable GPU found, update driver)");
			m_UseCuda = (m_Transform != NULL) && cuda_device;
		}
		catch (itk::ExceptionObject& excep)
		{
			std::cerr << excep << std::endl;
			m_UseCuda = false;
		}

		if (!m_UseCuda) return Superclass::GenerateData();			

		/* initialise cuda device */
		m_cuda.cudaInit();

		/* copy the parameters to the GPU */
		copyParameters();

		/* allocate host memory for the output and copy/cast the result back to the host */
		AllocateOutputs();
		InputPixelType* data = GetOutput()->GetBufferPointer();

		/* run the resampler */
		m_cuda.GenerateData(data);
	}
}; /* namespace itk */
