/* $Id$ */
#if !defined(ITKCUDARESAMPLEFILTER_H)
#define ITKCUDARESAMPLEFILTER_H

#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "cudaResampleImageFilter.cuh"

namespace itk
{

template <typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType = float>
class ITK_EXPORT itkCUDAResampleImageFilter:
	public ResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType>
{
public:
	/** Standard class typedefs. */
	typedef itkCUDAResampleImageFilter                          Self;
	typedef ResampleImageFilter<
    TInputImage,TOutputImage,TInterpolatorPrecisionType>      Superclass;
	typedef SmartPointer<Self>                                  Pointer;
	typedef SmartPointer<const Self>                            ConstPointer;

  typedef AdvancedCombinationTransform<
    TInterpolatorPrecisionType, 3 >                           InternalComboTransformType;
	typedef AdvancedBSplineDeformableTransform<
    TInterpolatorPrecisionType, 3, 3>                         InternalBSplineTransformType;
	typedef cuda::CUDAResampleImageFilter<
    typename InternalBSplineTransformType::ParametersValueType,
    typename TInputImage::PixelType, float>                   Cudaclass;

	itkCUDAResampleImageFilter();
	~itkCUDAResampleImageFilter();
	virtual void GenerateData( void );

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
  typename InternalBSplineTransformType::Pointer m_InternalCUDATransform;
	Cudaclass m_cuda;

	void copyParameters();
};

}; /* namespace itk */

#include "itkCUDAResampleImageFilter.hxx"

#endif /* ITKCUDARESAMPLEFILTER_H */
