/* $Id$ */
#pragma once

#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineDeformableTransform.h"
#include "cudaResampleImageFilter.cuh"

namespace itk
{

template <typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType = float>
class ITK_EXPORT itkCUDAResampleImageFilter:
	public ResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType>
{
public:
	/** Standard class typedefs. */
	typedef itkCUDAResampleImageFilter                                               Self;
	typedef ResampleImageFilter<TInputImage,TOutputImage,TInterpolatorPrecisionType> Superclass;
	typedef SmartPointer<Self>                                                       Pointer;
	typedef SmartPointer<const Self>                                                 ConstPointer;

	typedef BSplineDeformableTransform<TInterpolatorPrecisionType, 3, 3> BSplineTransformType;
	typedef cuda::CUDAResampleImageFilter<typename BSplineTransformType::ParametersValueType, typename TInputImage::PixelType, float> Cudaclass;

	itkCUDAResampleImageFilter();
	~itkCUDAResampleImageFilter();
	virtual void GenerateData();

	itkNewMacro(Self); 

	itkSetMacro(UseCuda, bool);
	itkBooleanMacro(UseCuda);
	itkGetConstMacro(UseCuda, bool);

	itkSetMacro(PreFilter, bool);
	itkBooleanMacro(PreFilter);
	itkGetConstMacro(PreFilter, bool);

private:
	bool m_UseCuda;
	bool m_PreFilter;
	const BSplineTransformType* m_Transform;
	Cudaclass m_cuda;

	void copyParameters();
};

}; /* namespace itk */

#include "itkCUDAResampleImageFilter.hxx"
